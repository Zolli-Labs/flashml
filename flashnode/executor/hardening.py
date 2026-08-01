"""The container security contract, in one place.

Every sandboxed runner builds its `docker run` flags here. Keeping this in a
single function is deliberate: two runners maintaining their own flag lists
drift, and the drift is invisible — the runner that quietly lost
`--cap-drop=ALL` still passes all its behavioural tests.

Changing this function changes the guarantee for ALL runners.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

CONTAINER_WORKDIR = "/work"

# Docker container names must match [a-zA-Z0-9][a-zA-Z0-9_.-]*. We prefix
# with "flashnode-" (always alnum-first) so the sanitized task_id segment
# only has to avoid illegal characters, not worry about its own leading
# character.
_NAME_ILLEGAL = re.compile(r"[^A-Za-z0-9_.-]")


def container_name(task_id: object) -> str:
    """A Docker-legal, collision-resistant name for this attempt's container.

    Shared by every sandboxed runner (docker_runner, argv_runner) so a
    timed-out `docker run` can be killed by the SAME name it was launched
    with — the whole reason this lives in hardening.py rather than being
    reimplemented per runner (AGENTS.md: a single security-contract seam,
    not two copies that can drift).

    task_id comes from an untrusted payload — sanitize it into the name
    rather than interpolating it raw. A random suffix (not task_id alone)
    guarantees uniqueness even when two concurrent attempts share a task_id
    (e.g. a retried attempt racing a slow-to-expire prior one).
    """
    safe = _NAME_ILLEGAL.sub("-", str(task_id or ""))
    suffix = uuid.uuid4().hex[:8]
    return f"flashnode-{safe}-{suffix}" if safe else f"flashnode-{suffix}"


def _user_flag() -> list[str]:
    """The `--user` flag, or nothing, depending on what the platform can
    actually prove about identity.

    - POSIX (`os.getuid`/`os.getgid` exist): pass the invoking user's real
      uid:gid, exactly as before. Files written into the bind-mounted
      workdir come back owned by the caller, not root.
    - Windows: `os.getuid`/`os.getgid` don't exist, and Docker Desktop does
      not map a host identity into the Linux container the way a native
      Linux daemon does — there is no host uid to pass. Omitting `--user`
      here is safe ONLY because every curated image
      (flashml-cloud/images/*/Dockerfile) ends in a fixed non-root `USER`;
      see that directory's README.md and flashml-cloud's
      docs/superpowers/plans/2026-08-01-windows-hosts.md ("The trap at the
      centre of this plan"). If any curated image ever regresses to
      running as root, this omission silently runs strangers' code as
      container root on every Windows host.
    - Anything else: refuse. Security fields fail closed
      (flashruntime/CLAUDE.md rule 3) — an unrecognised platform must not
      silently produce a container running as an unknown, possibly root,
      identity.
    """
    if hasattr(os, "getuid"):
        return ["--user", f"{os.getuid()}:{os.getgid()}"]
    if sys.platform == "win32":
        return []
    raise RuntimeError(
        f"cannot determine a safe --user for platform {sys.platform!r}: "
        "no os.getuid/os.getgid and not win32 — refusing to run "
        "unprivileged-in-name-only"
    )


def harden_args(
    workdir: Path,
    *,
    cpus: float,
    memory_gb: float,
    pids_limit: int = 512,
) -> list[str]:
    """Docker flags common to every sandboxed task."""
    return [
        # the job never reaches the volunteer's LAN or the internet; the
        # agent is the courier for inputs, outputs, and checkpoints
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        *_user_flag(),
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={pids_limit}",
        "--cpus", str(cpus),
        # equal values: with a larger memory-swap the memory cap is
        # bypassable by swapping
        "--memory", f"{memory_gb}g",
        "--memory-swap", f"{memory_gb}g",
        "--ulimit", "nofile=1024:1024",
        "-v", f"{workdir}:{CONTAINER_WORKDIR}",
        "-w", CONTAINER_WORKDIR,
    ]
