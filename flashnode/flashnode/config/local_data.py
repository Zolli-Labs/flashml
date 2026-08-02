"""Local datasets: data the host owner lends to tasks without uploading it.

    FLASHNODE_LOCAL_DATA="patients=/srv/data/patients-2026,labs=/srv/labs"

The owner names a directory on their machine and gives it a LABEL. The label
is what the agent advertises to the coordinator (`NodeRegistration.
local_datasets`, so the placement gate can route a job that needs `patients`
to a machine that has it); the PATH never leaves this machine. That asymmetry
is the feature — a path leaks the owner's directory layout, their username,
and frequently the dataset's identity, and none of that is needed to schedule
work.

Two rules make the rest of the system safe to write:

- **Fail closed on garbage** (same judgement as `HostPolicy` above): a
  malformed value raises rather than yielding the entries that happened to
  parse. Half-applying the owner's intent is the worst outcome — they believe
  two directories are exposed, one is, and the disagreement surfaces hours
  later as a job that never places.
- **A label is a name, not a path fragment.** It is restricted to
  ``[A-Za-z0-9._-]`` here, once, so that no consumer downstream has to defend
  itself: it is a map key, and a single container-side directory segment. If
  someone later joins it to a host path, the charset already forbids the
  escape.
"""

from __future__ import annotations

import os
import re

__all__ = ["LocalDataError", "LABEL_RE", "check_label", "parse_local_data",
           "load_local_data", "LOCAL_DATA_ENV"]

LOCAL_DATA_ENV = "FLASHNODE_LOCAL_DATA"

#: The whole alphabet a dataset label may use. Deliberately narrower than
#: anything a filesystem accepts: '/', '..' and whitespace are the characters
#: that turn a name into a traversal, and they are simply not expressible.
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class LocalDataError(ValueError):
    """The owner's `FLASHNODE_LOCAL_DATA` cannot be honoured as written."""


def check_label(label: str) -> str:
    """Return `label` if it is a legal dataset label; raise otherwise.

    Shared by the parser (owner-supplied labels) and the mount builder
    (payload-supplied labels) so both ends of the wire agree on what a label
    is — one definition, not two that can drift.
    """
    if not LABEL_RE.match(label) or label in (".", ".."):
        raise LocalDataError(
            f"illegal local dataset label {label!r}: labels are names, not "
            "paths — use only [A-Za-z0-9._-]"
        )
    return label


def parse_local_data(raw: str | None) -> dict[str, str]:
    """Parse ``"label=/path,other=/path2"`` into ``{label: path}``.

    Unset, empty, or all-separators means the owner did not opt in, and that
    is the normal case: an empty map, no mounts, nothing advertised.
    """
    result: dict[str, str] = {}
    for entry in (raw or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        label, sep, path = entry.partition("=")
        label, path = label.strip(), path.strip()
        if not sep or not label or not path:
            raise LocalDataError(
                f"malformed {LOCAL_DATA_ENV} entry {entry!r}: expected "
                "label=/absolute/path"
            )
        check_label(label)
        if not path.startswith("/") and ":" not in path:
            # A relative path would resolve against whatever directory the
            # agent happens to have been started in — never what the owner
            # meant, and a moving target between a shell run and a systemd
            # unit. (The ':' escape hatch is a Windows drive letter, which
            # the next check then judges on its own terms.)
            raise LocalDataError(
                f"local dataset {label!r} path {path!r} is not absolute — "
                "refusing to resolve it against the agent's working directory"
            )
        if ":" in path and not re.match(r"^[A-Za-z]:[\\/]", path):
            # `docker -v` splits its argument on ':'. A source containing one
            # silently re-reads as src:dst:opts — i.e. as a mount the owner
            # did not write. A Windows drive letter is the one form that is
            # both legitimate and rewritable (hardening._bind_mount_source).
            raise LocalDataError(
                f"local dataset {label!r} path {path!r} contains ':', which "
                "cannot be used as a bind-mount source"
            )
        if label in result:
            raise LocalDataError(
                f"local dataset label {label!r} is mapped twice — refusing to "
                "guess which directory the owner meant"
            )
        result[label] = path
    return result


def load_local_data(env: dict[str, str] | None = None) -> dict[str, str]:
    """The host owner's dataset map, from the environment."""
    return parse_local_data((env if env is not None else os.environ).get(LOCAL_DATA_ENV))
