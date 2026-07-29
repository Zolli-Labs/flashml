"""The container security contract, in one place.

Every sandboxed runner builds its `docker run` flags here. Keeping this in a
single function is deliberate: two runners maintaining their own flag lists
drift, and the drift is invisible — the runner that quietly lost
`--cap-drop=ALL` still passes all its behavioural tests.

Changing this function changes the guarantee for ALL runners.
"""

from __future__ import annotations

import os
from pathlib import Path

CONTAINER_WORKDIR = "/work"


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
        "--user", f"{os.getuid()}:{os.getgid()}",
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
