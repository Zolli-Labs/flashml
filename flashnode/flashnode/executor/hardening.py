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
from pathlib import Path, PureWindowsPath

from flashnode.config.local_data import LocalDataError, check_label, load_local_data
from flashnode.executor.runner import TaskExecutionError

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


def _bind_mount_source(workdir: Path) -> str:
    """Render `workdir` as the source half of `-v src:dst`, in a form
    Docker's flag parser won't misinterpret.

    `-v` splits its argument on ':'. A POSIX path never contains one, so it
    passes through byte-identical (this must not change — see the plan's
    Global Constraints). A Windows path (`C:\\Users\\phong\\...`) contains
    both backslashes AND a drive-letter colon, which is ambiguous with the
    src:dst separator itself — that raw form is what breaks the split.

    Detecting "is this a Windows path" by `isinstance(..., PureWindowsPath)`
    rather than `sys.platform == "win32"` is deliberate: a real `Path`
    passed in while running on Windows already IS a `WindowsPath`, a
    `PureWindowsPath` subclass, so the same branch handles the live case
    and lets tests exercise it on any host with a synthetic
    `PureWindowsPath` — no `sys.platform` gate to bit-rot in CI.

    Rewrites `C:\\Users\\phong\\work` to `/c/Users/phong/work`: lowercase
    drive letter, forward slashes, exactly one ':' left in the whole
    argument (the src:dst separator). This is the form Docker Desktop's
    Windows CLI accepts for bind-mount sources.
    """
    if isinstance(workdir, PureWindowsPath):
        drive = workdir.drive  # e.g. "C:" — empty for a driveless path
        if not drive:
            raise ValueError(
                f"Windows workdir {workdir!r} has no drive letter — cannot "
                "build a Docker Desktop bind-mount source from it"
            )
        letter = drive.rstrip(":").lower()
        rest = "/".join(workdir.parts[1:])
        return f"/{letter}/{rest}" if rest else f"/{letter}"
    return str(workdir)


def local_data_mounts(
    local_inputs: object,
    available: dict[str, str] | None = None,
) -> list[str]:
    """`-v` flags binding the host owner's local datasets READ-ONLY under
    `/work/inputs/<label>`, for the labels this payload asked for.

    This is the half of the local-data feature that never uploads anything:
    the bytes stay on the owner's disk and the task reads them in place. Three
    properties are load-bearing.

    - **`:ro`, always.** The owner lends the data; they do not hand a
      stranger's code write access to it. There is no flag to turn this off.
    - **Only what was requested.** A payload that names `patients` gets
      `patients`. A host that also lends `labs` does not silently expose it to
      a job that never asked — least privilege, per task.
    - **Refuse rather than run half-fed** (AGENTS.md rule 3, fail closed). A
      label this host has not mapped is a `TaskExecutionError` naming the
      label — the *task* fails and requeues elsewhere; the agent lives. It
      should not be reachable at all, because the coordinator's placement gate
      reads the same advertisement this host published, so arriving here means
      gate and host disagree — precisely the moment to stop, not to improvise
      an empty directory the workload would read as "no patients".

    `local_inputs` comes from an untrusted payload, so it is type-checked and
    charset-checked here rather than trusted to be the list the compiler
    produced. A label is never joined to a host path: it is a map key, and one
    container-side directory segment.

    `available` defaults to the host owner's `FLASHNODE_LOCAL_DATA` — read
    only when a task actually asks for something, so a host whose value is
    malformed fails the tasks that need it rather than every task on the
    machine.
    """
    if local_inputs is None:
        return []
    if not isinstance(local_inputs, list) or not all(
        isinstance(name, str) for name in local_inputs
    ):
        raise TaskExecutionError(
            "payload 'local_inputs' must be a list of local dataset labels"
        )
    if not local_inputs:
        return []
    if available is None:
        try:
            available = load_local_data()
        except LocalDataError as exc:
            # Barely reachable — `discover()` parses the same value at startup
            # and the agent refuses to register on a malformed one. If it ever
            # is reached, a misconfigured host fails a task; it does not die.
            raise TaskExecutionError(f"host local-data config is unusable: {exc}") from None
    args: list[str] = []
    for label in local_inputs:
        try:
            check_label(label)
        except LocalDataError as exc:
            raise TaskExecutionError(str(exc)) from None
        if label not in available:
            raise TaskExecutionError(
                f"task requires local dataset {label!r}, which this host does "
                "not provide — refusing to run"
            )
        source = _bind_mount_source(Path(available[label]))
        args += ["-v", f"{source}:{CONTAINER_WORKDIR}/inputs/{label}:ro"]
    return args


def gpu_flags(gpus: object) -> list[str]:
    """`--gpus N`, or nothing at all.

    Every other flag in this module NARROWS what a container may touch.
    This one widens it — it hands a device on someone else's machine to a
    stranger's code — so it is absent unless the payload asked, and the ask
    has to be a plain positive integer.

    Anything else emits nothing rather than raising. The coordinator's
    placement gate already refused a task whose `gpus` was not a positive
    int, so a bad value arriving here means gate and payload disagree; this
    is the second lock, not the validation. Two values are worth naming:

    - `True`. `bool` is an `int` subclass, so a naive check reads it as
      "1 GPU" — and `--gpus True` is not something docker accepts anyway.
    - `"all"`. This is *valid* docker syntax meaning every device on the
      host, which is exactly why an untrusted string must never reach the
      flag. A task asks for a count; it does not get to name a device.
    """
    if isinstance(gpus, bool) or not isinstance(gpus, int) or gpus <= 0:
        return []
    return ["--gpus", str(gpus)]


def harden_args(
    workdir: Path,
    *,
    cpus: float,
    memory_gb: float,
    pids_limit: int = 512,
    local_inputs: object = None,
    local_data: dict[str, str] | None = None,
    gpus: object = None,
) -> list[str]:
    """Docker flags common to every sandboxed task.

    `local_inputs` is the payload's list of local dataset labels (None or []
    for the overwhelmingly common task that needs none, and then the flags are
    byte-for-byte what they were before the feature existed). `local_data` is
    the host owner's label→path map; it defaults to their environment, so a
    runner never has to know where the map comes from.

    `gpus` is the payload's device count, straight from the wire and
    therefore typed `object` — None for every job that exists today, and
    then, as with `local_inputs`, the flag list is byte-for-byte what it was
    before this argument existed. `doctor.check_hardened_run` passes neither.
    """
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
        # With the other resource caps, and absent entirely for the job that
        # asked for no device — which is every job that exists today.
        *gpu_flags(gpus),
        "--ulimit", "nofile=1024:1024",
        "-v", f"{_bind_mount_source(workdir)}:{CONTAINER_WORKDIR}",
        # After the workdir mount, never before: these land *inside* it, at
        # /work/inputs/<label>, and the outer mount has to exist first.
        *local_data_mounts(local_inputs, local_data),
        "-w", CONTAINER_WORKDIR,
    ]
