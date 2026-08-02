"""Adversarial tests for the untrusted-archive extractor.

Every malicious archive here is built in-process with ``tarfile`` /
``zipfile``. Nothing touches the network: an attack corpus that has to be
downloaded is an attack corpus that silently stops being tested the first
time CI runs offline.

The positive control at the bottom matters as much as the refusals — an
extractor that rejects everything would pass every test above it.
"""

from __future__ import annotations

import io
import os
import tarfile
import zipfile
from pathlib import Path

import pytest

from flashnode.executor.archives import ArchiveError, extract_archive_safely

TOP = "repo-abc123"


# -- corpus builders ---------------------------------------------------------

def _tar(members: list[tuple[tarfile.TarInfo, bytes | None]]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for info, body in members:
            tar.addfile(info, io.BytesIO(body) if body is not None else None)
    return buf.getvalue()


def _file(name: str, content: bytes = b"x") -> tuple[tarfile.TarInfo, bytes]:
    info = tarfile.TarInfo(name=name)
    info.size = len(content)
    info.type = tarfile.REGTYPE
    return info, content


def _dir(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name=name.rstrip("/") + "/")
    info.type = tarfile.DIRTYPE
    return info, None


def _symlink(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.SYMTYPE
    info.linkname = target
    return info, None


def _hardlink(name: str, target: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.LNKTYPE
    info.linkname = target
    return info, None


def _fifo(name: str) -> tuple[tarfile.TarInfo, None]:
    info = tarfile.TarInfo(name=name)
    info.type = tarfile.FIFOTYPE
    return info, None


def _write(tmp_path: Path, data: bytes, name: str = "a.tar.gz") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def _extract(tmp_path: Path, data: bytes, **kw) -> Path:
    return extract_archive_safely(
        _write(tmp_path, data), tmp_path / "dest",
        kw.get("max_bytes", 1 << 20), kw.get("max_members", 1000),
    )


# -- zip-slip ----------------------------------------------------------------

def test_relative_path_escape_is_refused(tmp_path):
    data = _tar([_dir(TOP), _file(f"{TOP}/../../pwned.txt")])
    with pytest.raises(ArchiveError, match="unsafe path"):
        _extract(tmp_path, data)
    assert not (tmp_path.parent / "pwned.txt").exists()


def test_absolute_path_escape_is_refused(tmp_path):
    """The form a join-then-check misses entirely: ``Path(dest) / "/etc/x"``
    is ``/etc/x`` — pathlib drops the left operand when the right side is
    absolute, so the naive guard compares a path that was never inside
    ``dest`` in the first place and finds nothing wrong with it."""
    outside = tmp_path / "outside.txt"
    data = _tar([_dir(TOP), _file(str(outside))])
    with pytest.raises(ArchiveError, match="unsafe path"):
        _extract(tmp_path, data)
    assert not outside.exists()


def test_deep_traversal_inside_a_subdirectory_is_refused(tmp_path):
    data = _tar([_dir(TOP), _file(f"{TOP}/a/b/../../../../../escape.txt")])
    with pytest.raises(ArchiveError, match="unsafe path"):
        _extract(tmp_path, data)


# -- symlinks ----------------------------------------------------------------

def test_symlink_escaping_the_destination_is_refused(tmp_path):
    data = _tar([_dir(TOP), _symlink(f"{TOP}/link", "../../../../etc/passwd")])
    with pytest.raises(ArchiveError, match="escapes the destination"):
        _extract(tmp_path, data)


def test_absolute_symlink_is_refused(tmp_path):
    data = _tar([_dir(TOP), _symlink(f"{TOP}/link", "/etc/passwd")])
    with pytest.raises(ArchiveError, match="escapes the destination"):
        _extract(tmp_path, data)


def test_contained_symlink_is_accepted_but_never_materialized(tmp_path):
    """The link is validated and then simply not created. That is what makes
    the follow-up attack — a later member writing *through* the link — a
    non-event rather than a race we have to win."""
    data = _tar([_dir(TOP), _file(f"{TOP}/real.txt"), _symlink(f"{TOP}/link", "real.txt")])
    root = _extract(tmp_path, data)
    assert (root / "real.txt").is_file()
    assert not (root / "link").exists()
    assert not (root / "link").is_symlink()


def test_write_through_a_symlink_member_cannot_escape(tmp_path):
    """Member 1 is a symlink to /tmp; member 2 writes "through" it. Because
    member 1 was never created, member 2 is just a plain nested file inside
    the destination."""
    data = _tar([
        _dir(TOP),
        _symlink(f"{TOP}/link", str(tmp_path / "elsewhere")),
        _file(f"{TOP}/link/planted.txt", b"owned"),
    ])
    with pytest.raises(ArchiveError, match="escapes the destination"):
        _extract(tmp_path, data)
    assert not (tmp_path / "elsewhere" / "planted.txt").exists()


# -- member types ------------------------------------------------------------

def test_hardlink_member_is_refused(tmp_path):
    data = _tar([_dir(TOP), _file(f"{TOP}/a.txt"), _hardlink(f"{TOP}/b.txt", f"{TOP}/a.txt")])
    with pytest.raises(ArchiveError, match="hard-link"):
        _extract(tmp_path, data)


def test_fifo_member_is_refused(tmp_path):
    data = _tar([_dir(TOP), _fifo(f"{TOP}/pipe")])
    with pytest.raises(ArchiveError, match="unsupported type"):
        _extract(tmp_path, data)


def test_device_member_is_refused(tmp_path):
    info = tarfile.TarInfo(name=f"{TOP}/dev")
    info.type = tarfile.CHRTYPE
    data = _tar([_dir(TOP), (info, None)])
    with pytest.raises(ArchiveError, match="unsupported type"):
        _extract(tmp_path, data)


# -- caps --------------------------------------------------------------------

def test_decompression_bomb_is_refused_during_extraction(tmp_path):
    """4 MiB of zeros compresses to a few KB. The refusal has to happen
    while the member is streaming, not from a total computed afterwards —
    by then the disk this cap protects is already full."""
    data = _tar([_dir(TOP), _file(f"{TOP}/bomb.bin", b"\0" * (4 * 1024 * 1024))])
    assert len(data) < 100 * 1024, "the bomb should be small on the wire"
    with pytest.raises(ArchiveError, match="size cap"):
        _extract(tmp_path, data, max_bytes=64 * 1024)


def test_size_cap_counts_across_members_not_per_member(tmp_path):
    members = [_dir(TOP)] + [_file(f"{TOP}/f{i}.bin", b"\0" * 4096) for i in range(64)]
    with pytest.raises(ArchiveError, match="size cap"):
        _extract(tmp_path, _tar(members), max_bytes=32 * 1024)


def test_member_count_cap_is_enforced(tmp_path):
    """Millions of empty files exhaust inodes without ever moving a byte
    counter, so the count is capped independently of the size."""
    members = [_dir(TOP)] + [_file(f"{TOP}/f{i}", b"") for i in range(50)]
    with pytest.raises(ArchiveError, match="more than 10 members"):
        _extract(tmp_path, _tar(members), max_members=10)


# -- refusal hygiene ---------------------------------------------------------

def test_refusal_leaves_nothing_behind(tmp_path):
    """A rejected archive must not leave a partial tree at the destination
    for the next stage to mistake for a successful unpack."""
    data = _tar([
        _dir(TOP),
        _file(f"{TOP}/innocent.txt", b"already written"),
        _file(f"{TOP}/../../pwned.txt"),
    ])
    with pytest.raises(ArchiveError):
        _extract(tmp_path, data)
    assert not (tmp_path / "dest").exists()


def test_garbage_bytes_raise_the_typed_error(tmp_path):
    with pytest.raises(ArchiveError):
        _extract(tmp_path, b"this is not an archive at all")


def test_archive_root_member_does_not_crash(tmp_path):
    """GNU tar emits a ``./`` member for the archive root. Indexing its
    (empty) path components unguarded turns that into an IndexError — a
    crash where a clean result was due."""
    data = _tar([_dir("."), _file("./file.txt", b"hi")])
    root = _extract(tmp_path, data)
    assert (root / "file.txt").read_bytes() == b"hi"


# -- zip ---------------------------------------------------------------------

def test_zip_traversal_is_refused(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../pwned.txt", "owned")
    with pytest.raises(ArchiveError, match="unsafe path"):
        _extract(tmp_path, buf.getvalue(), )


def test_zip_symlink_escape_is_refused(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        info = zipfile.ZipInfo("link")
        info.create_system = 3
        info.external_attr = (0o120777 << 16)
        zf.writestr(info, "/etc/passwd")
    with pytest.raises(ArchiveError, match="escapes the destination"):
        _extract(tmp_path, buf.getvalue())


def test_zip_bomb_is_refused(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("big.bin", b"\0" * (4 * 1024 * 1024))
    with pytest.raises(ArchiveError, match="size cap"):
        _extract(tmp_path, buf.getvalue(), max_bytes=64 * 1024)


# -- format detection --------------------------------------------------------

def test_format_comes_from_the_bytes_not_the_filename(tmp_path):
    """The submitter names the artifact. If the name picked the parser, they
    would pick which code reads their bytes."""
    path = _write(tmp_path, _tar([_dir(TOP), _file(f"{TOP}/a.txt", b"hi")]), name="innocent.txt")
    root = extract_archive_safely(path, tmp_path / "dest", 1 << 20, 1000)
    assert (root / "a.txt").read_bytes() == b"hi"


# -- positive controls -------------------------------------------------------

def test_github_style_wrapper_directory_is_the_returned_root(tmp_path):
    """codeload wraps every repo in ``owner-name-<sha>/``. The compiled argv
    is ``python /work/inputs/code/train.py`` with no wrapper in it, so the
    content root — not the destination — is what the caller must be given."""
    data = _tar([
        _dir(TOP), _file(f"{TOP}/train.py", b"print(1)"),
        _dir(f"{TOP}/pkg"), _file(f"{TOP}/pkg/mod.py", b"x = 1"),
    ])
    root = _extract(tmp_path, data)
    assert root.name == TOP
    assert (root / "train.py").read_bytes() == b"print(1)"
    assert (root / "pkg" / "mod.py").is_file()


def test_multiple_top_level_entries_return_the_destination_itself(tmp_path):
    data = _tar([_file("a.txt", b"a"), _file("b.txt", b"b")])
    root = _extract(tmp_path, data)
    assert root == (tmp_path / "dest").resolve()
    assert (root / "a.txt").is_file() and (root / "b.txt").is_file()


def test_executable_bit_is_not_carried_across(tmp_path):
    """Members are rewritten as plain files. An archive cannot hand a
    volunteer's machine a setuid or world-writable file."""
    info, body = _file(f"{TOP}/script.sh", b"#!/bin/sh\n")
    info.mode = 0o4777
    root = _extract(tmp_path, _tar([_dir(TOP), (info, body)]))
    mode = (root / "script.sh").stat().st_mode
    assert not mode & 0o4000
    assert not mode & 0o111


def test_a_plain_repo_extracts_fully(tmp_path):
    """The whole point of the guard is that legitimate archives still work."""
    files = {f"{TOP}/f{i}.py": f"v = {i}".encode() for i in range(20)}
    data = _tar([_dir(TOP)] + [_file(n, c) for n, c in files.items()])
    root = _extract(tmp_path, data)
    for name, content in files.items():
        assert (root / os.path.relpath(name, TOP)).read_bytes() == content
