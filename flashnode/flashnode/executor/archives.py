"""Unpacking an untrusted archive onto a volunteer's machine.

This is the one place in flashnode that writes attacker-chosen *paths* to a
host owner's disk. The bytes arrive as a task input: any signed-in user
points the cloud API at any repo they like, the API stages the original
archive as an ``artifact://`` input, and this agent downloads it. Member
names, symlink targets, member types and declared sizes are all chosen by
whoever submitted the job — never by us.

The cloud API has an equivalent guard on its own copy of the same bytes
(``flashml_cloud_api.repo.extract_safely``). That one is *not* reusable
here: flashnode may import ``flashruntime.protocol`` and nothing else (see
AGENTS.md hard rule 2), and a private control-plane module must never be a
dependency of the public host agent. So the guard is deliberately
duplicated, and the drift that duplication invites is caught by a parity
test in the workspace ``e2e/`` suite — the only place allowed to import
both repos — which feeds one attack corpus to both extractors.

What is refused, and why each one matters on someone else's laptop:

- **zip-slip**, relative (``../../.ssh/authorized_keys``) or absolute
  (``/etc/cron.d/evil``). Both are caught by the same check rather than
  special-cased, because a naive join-then-compare misses the absolute
  form entirely: ``Path("/work/inputs/code") / "/etc/evil"`` is
  ``/etc/evil`` — pathlib silently discards the left operand when the
  right side is absolute. Resolving first and then requiring containment
  via ``relative_to`` has no such blind spot.
- **escaping symlinks**. A member linking to ``/`` or ``../..`` turns a
  later, individually-innocent write into an escape. Contained symlinks
  are validated and then *not created at all*: nothing downstream needs a
  link, and not materialising one removes the whole TOCTOU class (member
  ``a`` is a symlink to a safe dir; member ``a/b`` then resolves through
  it) for free.
- **decompression bombs**, checked *during* extraction and in bounded
  chunks. A 40 KB archive can hold petabytes; a total measured after the
  fact has already filled the disk it was meant to protect.
- **member count**, because millions of empty files exhaust inodes without
  ever tripping a byte cap.
- **device, fifo and hard-link members**, which have no legitimate place in
  a code archive and every place in an attack.

Nothing is executed, nothing is made executable (members are rewritten as
plain files with the process umask, so an archive cannot smuggle in a
setuid bit), and on any refusal everything already written is removed —
a rejected archive must not leave half a payload behind for the next stage
to trip over.
"""

from __future__ import annotations

import shutil
import stat
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

#: Read/write in bounded chunks so an oversized member is caught partway
#: through rather than only once it is fully on disk.
_CHUNK_SIZE = 1 << 16  # 64 KiB

#: Defaults for a task input. Generous enough for a real repo (the cloud
#: API caps an extracted repo at 128 MiB), small enough that a hostile job
#: cannot fill a volunteer's disk before the cap fires.
DEFAULT_MAX_BYTES = 128 * 1024 * 1024
DEFAULT_MAX_MEMBERS = 20_000


class ArchiveError(Exception):
    """Raised for any archive this agent refuses to unpack: a malformed
    container, a member whose path or symlink target escapes the
    destination, an unsupported member type, or a cap (bytes, members)
    exceeded. The executor turns it into a task failure, never into a
    crashed agent — a hostile input must cost the submitter their task, not
    the volunteer their node."""


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def _safe_target(name: str, dest_root: Path) -> Path:
    """Resolve a member name against the extraction root, refusing anything
    that lands outside it. Handles the relative and the absolute escape with
    one check — see the module docstring on why joining first is not enough
    on its own but resolving after it is."""
    resolved = (dest_root / name).resolve()
    if not _is_within(resolved, dest_root):
        raise ArchiveError(f"refusing archive member with unsafe path: {name!r}")
    return resolved


def _check_symlink(member_path: Path, link_name: str, dest_root: Path) -> None:
    """A symlink target resolves against the link's own directory unless it
    is absolute — standard symlink semantics, and the semantics an attacker
    is counting on."""
    if PurePosixPath(link_name).is_absolute():
        target = Path(link_name).resolve()
    else:
        target = (member_path.parent / link_name).resolve()
    if not _is_within(target, dest_root):
        raise ArchiveError(
            f"refusing symlink member {member_path.name!r} whose target "
            f"{link_name!r} escapes the destination"
        )


def _top_level(name: str) -> str | None:
    """First path component of a member name, or None for a name that has
    none (``.``/``./``, which GNU tar emits for the archive root). The
    None case is not hypothetical: indexing ``parts[0]`` unguarded turns a
    one-member archive into an IndexError instead of a clean refusal."""
    parts = PurePosixPath(name).parts
    return parts[0] if parts else None


def _write_stream(src, out_path: Path, budget: list[int]) -> None:
    """Copy a member's bytes in chunks, decrementing a shared budget as we
    go. ``budget`` is a one-element list so the running total survives
    across members without a class or a global."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as out:
        while True:
            chunk = src.read(_CHUNK_SIZE)
            if not chunk:
                return
            budget[0] -= len(chunk)
            if budget[0] < 0:
                raise ArchiveError(
                    "archive exceeds the uncompressed size cap during "
                    "extraction — refusing the rest"
                )
            out.write(chunk)


def _extract_tar(src: Path, dest_root: Path, budget: list[int], max_members: int) -> set[str]:
    try:
        tar = tarfile.open(src, mode="r:*")
    except tarfile.TarError as exc:
        raise ArchiveError(f"not a valid tar archive: {exc}") from exc

    tops: set[str] = set()
    seen = 0
    with tar:
        for member in tar:
            seen += 1
            if seen > max_members:
                raise ArchiveError(
                    f"archive has more than {max_members} members — refusing the rest"
                )
            top = _top_level(member.name)
            if top is not None:
                tops.add(top)
            member_path = _safe_target(member.name, dest_root)

            if member.issym():
                _check_symlink(member_path, member.linkname, dest_root)
                continue  # validated, deliberately not created — see docstring
            if member.islnk():
                raise ArchiveError(
                    f"refusing hard-link member: {member.name!r}"
                )
            if member.isdir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                # chr/blk/fifo — nothing a code archive needs.
                raise ArchiveError(
                    f"refusing archive member of unsupported type: {member.name!r}"
                )

            stream = tar.extractfile(member)
            if stream is None:
                continue
            _write_stream(stream, member_path, budget)
    return tops


def _extract_zip(src: Path, dest_root: Path, budget: list[int], max_members: int) -> set[str]:
    try:
        zf = zipfile.ZipFile(src)
    except zipfile.BadZipFile as exc:
        raise ArchiveError(f"not a valid zip archive: {exc}") from exc

    tops: set[str] = set()
    with zf:
        infos = zf.infolist()
        if len(infos) > max_members:
            raise ArchiveError(
                f"archive has more than {max_members} members — refusing it"
            )
        for info in infos:
            top = _top_level(info.filename)
            if top is not None:
                tops.add(top)
            member_path = _safe_target(info.filename, dest_root)

            # Zip stores unix mode in the high 16 bits of external_attr, but
            # only when it was created on a unix host AND the writer bothered
            # to set the file-type bits — plenty of legitimate zips carry
            # bare permission bits with no S_IFREG. Treating a missing type
            # as "unsupported" would refuse ordinary archives, so the type
            # check only runs when there is a type to check.
            mode = info.external_attr >> 16
            if info.create_system == 3 and stat.S_IFMT(mode):
                if stat.S_ISLNK(mode):
                    # A zip symlink stores its target as the member's own
                    # content, so reading it is a read of attacker bytes:
                    # cap it rather than pulling an arbitrary "target" into
                    # memory to validate a link we are not going to create.
                    if info.file_size > 4096:
                        raise ArchiveError(
                            f"refusing symlink member {info.filename!r} with an "
                            f"implausible {info.file_size}-byte target"
                        )
                    _check_symlink(
                        member_path,
                        zf.read(info).decode("utf-8", errors="replace"),
                        dest_root,
                    )
                    continue
                if not (stat.S_ISREG(mode) or stat.S_ISDIR(mode)):
                    raise ArchiveError(
                        f"refusing archive member of unsupported type: "
                        f"{info.filename!r}"
                    )
            if info.is_dir():
                member_path.mkdir(parents=True, exist_ok=True)
                continue
            with zf.open(info) as stream:
                _write_stream(stream, member_path, budget)
    return tops


def extract_archive_safely(
    src: Path,
    dest: Path,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_members: int = DEFAULT_MAX_MEMBERS,
) -> Path:
    """Unpack the archive at ``src`` under ``dest`` and return the path the
    *content* actually lives at.

    That return value is not always ``dest``: an archive whose members all
    sit under a single top-level directory — which is exactly what GitHub's
    ``codeload`` tarballs are, wrapping every repo in ``owner-name-<sha>/``
    — has that wrapper as its content root. Returning it is what lets the
    caller present the repo at the path the job's argv was compiled
    against; returning ``dest`` blindly would leave every such job pointing
    one directory too high, which is the same "file not found" the missing
    unpacking caused in the first place.

    The container format is detected from the bytes, never from the file
    name: an attacker supplies the name, so trusting it would let them pick
    which parser reads their bytes. Supported: tar with any of gzip/bzip2/xz
    compression or none, and zip.

    Raises ``ArchiveError`` for every refusal. On refusal ``dest`` is
    removed entirely.
    """
    src = Path(src)
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    dest_root = dest.resolve()
    budget = [int(max_bytes)]

    try:
        if zipfile.is_zipfile(src):
            tops = _extract_zip(src, dest_root, budget, max_members)
        else:
            tops = _extract_tar(src, dest_root, budget, max_members)
    except ArchiveError:
        shutil.rmtree(dest_root, ignore_errors=True)
        raise
    except Exception as exc:
        # Anything the stdlib raises from hostile bytes (an OSError on an
        # absurd name, a zlib error mid-stream) becomes the same typed
        # refusal: the caller must never have to catch the union of every
        # decompressor's private exception tree.
        shutil.rmtree(dest_root, ignore_errors=True)
        raise ArchiveError(f"could not extract archive: {exc}") from exc

    if len(tops) == 1:
        only = dest_root / next(iter(tops))
        if only.is_dir():
            return only
    return dest_root
