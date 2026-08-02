"""Host health checks for the sandboxed execution tiers.

WHY THIS EXISTS. Two machines stopped the 2026-08-02 §10 run-through, and
neither was a distributed-systems problem: `docker-credential-desktop`
missing on macOS, and a Docker engine answering `_ping` with 500 on Windows.
The startup gate at the time was `shutil.which("docker")`, which BOTH
machines pass — the binary was on PATH in both cases.

What happened instead is worse than a crash. `docker_runner` turns a
non-zero `docker run` into TaskExecutionError; `loop.py` catches it, calls
fail() on the lease, and keeps claiming. A host with broken Docker therefore
claims a task, fails it, claims the next one, and never tells its owner. The
volunteer sees a healthy-looking agent; the submitter sees their job failing
with a Docker error tail from a stranger's laptop.

Every check takes its side effects as a parameter. That is not test
decoration: a diagnostic you can only exercise on a broken machine is one
nobody can keep correct.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from flashnode.executor.hardening import (
    CONTAINER_WORKDIR,
    _bind_mount_source,
    harden_args,
)

__all__ = [
    "PROBE_IMAGE",
    "CheckResult",
    "check_cli_on_path",
    "check_engine",
    "check_hardened_run",
    "check_local_datasets",
    "check_pull",
    "check_workdir_mount",
    "default_workdir",
    "doctor_main",
    "exit_code",
    "format_results",
    "run_checks",
    "run_command",
]

#: The image every container-level check runs. python-slim, never
#: pytorch-cpu: registry auth, TLS and the credential helper are properties
#: of the REGISTRY, so the smallest curated image proves the same thing, and
#: making a volunteer download gigabytes to learn their helper is missing is
#: a hostile diagnostic. Kept in step with flashml-cloud's published tags.
PROBE_IMAGE = "ghcr.io/zolli-labs/flashml-python-slim:2026.08.1"

CommandRunner = Callable[..., subprocess.CompletedProcess]


@dataclass(frozen=True)
class CheckResult:
    """One check's verdict.

    `fix` is the point of the whole module: a volunteer will not derive
    "your credential helper is missing" from `task exited 1`.
    """

    name: str
    status: str  # "ok" | "fail" | "skip"
    detail: str = ""
    fix: str = ""


def run_command(argv: Sequence[str], *, timeout: float = 300.0) -> subprocess.CompletedProcess:
    """The real side effect. Generous timeouts because a cold image pull is
    slow on a home connection, and a doctor that times out on a healthy host
    is worse than no doctor."""
    return subprocess.run(list(argv), capture_output=True, timeout=timeout, check=False)


def _text(raw: bytes | str | None) -> str:
    if raw is None:
        return ""
    if isinstance(raw, str):
        return raw.strip()
    return raw.decode(errors="replace").strip()


def check_cli_on_path(which: Callable[[str], str | None] | None = None) -> CheckResult:
    # Resolved at CALL time, never as a default argument. `def f(which=
    # shutil.which)` captures the original function object at import, so
    # `monkeypatch.setattr("shutil.which", ...)` — the idiom the rest of this
    # suite uses, and documents in test_agent.py — silently fails to reach
    # it. That is not merely a testing inconvenience: it let the `work` gate
    # ignore a patched `which`, run a REAL `docker run` inside unit tests,
    # and pass or fail with the machine's Docker state.
    which = which or shutil.which
    name = "docker CLI on PATH"
    found = which("docker")
    if found:
        return CheckResult(name, "ok", detail=found)
    return CheckResult(
        name, "fail",
        detail="`docker` was not found on PATH",
        fix="Install Docker Desktop (macOS/Windows) or your distribution's "
            "docker package, then re-run `flashnode doctor`.",
    )


def check_engine(run: CommandRunner) -> CheckResult:
    """The binary existing says nothing about the daemon behind it. This is
    the check the Windows machine needed and did not have."""
    name = "docker engine reachable"
    argv = ["docker", "version", "--format", "{{.Server.Version}}"]
    try:
        proc = run(argv, timeout=30.0)
    except FileNotFoundError as exc:
        # `docker` vanished between check 1 and here. Report, never raise:
        # a diagnostic that crashes has diagnosed nothing.
        return CheckResult(name, "fail", detail=str(exc),
                           fix="Install Docker and re-run `flashnode doctor`.")
    except subprocess.TimeoutExpired:
        return CheckResult(
            name, "fail", detail="`docker version` did not answer within 30s",
            fix="The Docker daemon is hung. Restart Docker Desktop (or "
                "`systemctl restart docker`) and re-run `flashnode doctor`.",
        )
    if proc.returncode == 0 and _text(proc.stdout):
        return CheckResult(name, "ok", detail=f"server {_text(proc.stdout)}")
    return CheckResult(
        name, "fail", detail=_text(proc.stderr) or _text(proc.stdout),
        fix="The docker CLI is installed but no daemon answered. Start "
            "Docker Desktop and wait for it to report Running, or start the "
            "docker service, then re-run `flashnode doctor`.",
    )


def check_pull(run: CommandRunner, image: str = PROBE_IMAGE) -> CheckResult:
    """Pull the smallest curated image.

    This is the check the Mac needed. The pull is the only step that touches
    a registry and therefore the only one that consults a credential helper
    — tasks themselves run `--network none`. It also re-catches the GHCR
    visibility regression, where the images went private and every job died
    at execution while signup, install and enrolment all looked fine.
    """
    name = "pull a curated image"
    try:
        proc = run(["docker", "pull", image], timeout=600.0)
    except subprocess.TimeoutExpired:
        return CheckResult(
            name, "fail", detail=f"`docker pull {image}` did not finish in 10 minutes",
            fix="Check this machine's internet connection, then re-run "
                "`flashnode doctor`.",
        )
    except OSError as exc:
        return CheckResult(name, "fail", detail=str(exc),
                           fix="Install Docker and re-run `flashnode doctor`.")
    if proc.returncode == 0:
        return CheckResult(name, "ok", detail=image)
    err = _text(proc.stderr) or _text(proc.stdout)
    if "credential" in err or "docker-credential" in err:
        fix = ("Your ~/.docker/config.json names a credential helper that is "
               "not installed. Either start Docker Desktop, or remove the "
               '"credsStore" line — these images are public and need no login.')
    elif "denied" in err or "unauthorized" in err:
        fix = ("The registry refused an anonymous pull. These images are "
               "meant to be public; report this — it is our bug, not yours.")
    else:
        fix = ("Check this machine's internet connection and that "
               "ghcr.io is reachable, then re-run `flashnode doctor`.")
    return CheckResult(name, "fail", detail=f"{image}\n{err}", fix=fix)


PROBE_FILENAME = "flashnode-doctor-probe.txt"
PROBE_CONTENT = "flashnode-doctor"


def default_workdir() -> Path:
    """Where the agent will actually stage task inputs.

    Mirrors ExecutorLoop's own default (`workdir_base=None` → the system
    temp dir) deliberately. On macOS that is /var/folders/…, which colima
    cannot see — so the doctor fails on precisely the machines where the
    agent would, which is the entire point.
    """
    return Path(os.environ.get("FLASHNODE_WORKDIR") or tempfile.gettempdir())


def _mount_failure_fix(err: str) -> str:
    if "No such image" in err or "not found" in err.lower():
        return ("The probe image is not cached on this machine. Run "
                "`flashnode doctor` once — it pulls it.")
    return ("The container could not see this directory. On macOS, "
            "colima and Docker Desktop share only $HOME: set "
            "FLASHNODE_WORKDIR to a path under your home directory "
            "(e.g. export FLASHNODE_WORKDIR=$HOME/.flashnode/work) and "
            "re-run `flashnode doctor`.")


def check_workdir_mount(
    run: CommandRunner, workdir: Path, image: str = PROBE_IMAGE
) -> CheckResult:
    """Can a container see the directory the agent stages inputs in?

    Minimum viable flags, on purpose. Check 5 runs the same probe with the
    full hardening set, so a failure HERE is a mount problem and a failure
    THERE is a flag problem — localised without pattern-matching stderr.

    The container READS a file the host wrote rather than writing one: the
    curated images end in a non-root USER and this flag set has no --user,
    so a write would hit permission denied on a healthy Linux host.
    """
    name = "workdir bind-mounts"
    probe = Path(workdir) / PROBE_FILENAME
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(PROBE_CONTENT)
    except OSError as exc:
        return CheckResult(
            name, "fail", detail=f"{workdir}: {exc}",
            fix="The agent cannot write to its own work directory. Set "
                "FLASHNODE_WORKDIR to a writable path and re-run "
                "`flashnode doctor`.",
        )
    argv = [
        "docker", "run", "--rm", "--pull=never",
        "-v", f"{_bind_mount_source(Path(workdir))}:{CONTAINER_WORKDIR}",
        "-w", CONTAINER_WORKDIR,
        image,
        "python", "-c",
        f"print(open('{CONTAINER_WORKDIR}/{PROBE_FILENAME}').read(), end='')",
    ]
    try:
        proc = run(argv, timeout=120.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(name, "fail", detail=str(exc),
                           fix=_mount_failure_fix(str(exc)))
    finally:
        probe.unlink(missing_ok=True)
    if proc.returncode == 0 and _text(proc.stdout) == PROBE_CONTENT:
        return CheckResult(name, "ok", detail=str(workdir))
    err = _text(proc.stderr) or _text(proc.stdout) or "the container read nothing back"
    return CheckResult(name, "fail", detail=f"{workdir}\n{err}",
                       fix=_mount_failure_fix(err))


def check_hardened_run(
    run: CommandRunner, workdir: Path, image: str = PROBE_IMAGE
) -> CheckResult:
    """The same probe as check 4, with the REAL sandbox flags.

    Check 4 uses the minimum that can work; this uses everything a task
    gets. So 4 passing and 5 failing localises the fault to a hardening
    flag, with no stderr pattern-matching — and on Windows this is the first
    thing in the system that has ever EXECUTED the platform-conditional
    --user path from Plan 6, which until now was only argv-verified.
    """
    name = "a hardened container runs"
    probe = Path(workdir) / PROBE_FILENAME
    try:
        probe.parent.mkdir(parents=True, exist_ok=True)
        probe.write_text(PROBE_CONTENT)
    except OSError as exc:
        return CheckResult(name, "fail", detail=f"{workdir}: {exc}",
                           fix="Set FLASHNODE_WORKDIR to a writable path.")
    try:
        flags = harden_args(Path(workdir), cpus=1.0, memory_gb=1.0)
    except (RuntimeError, ValueError) as exc:
        probe.unlink(missing_ok=True)
        return CheckResult(
            name, "fail", detail=str(exc),
            fix="This platform is not one the sandbox can secure. Report "
                "it — running your machine unprivileged-in-name-only is not "
                "something we will do.",
        )
    argv = [
        "docker", "run", "--rm", "--pull=never", *flags, image,
        "python", "-c",
        f"print(open('{CONTAINER_WORKDIR}/{PROBE_FILENAME}').read(), end='')",
    ]
    try:
        proc = run(argv, timeout=120.0)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult(name, "fail", detail=str(exc),
                           fix=_mount_failure_fix(str(exc)))
    finally:
        probe.unlink(missing_ok=True)
    if proc.returncode == 0 and _text(proc.stdout) == PROBE_CONTENT:
        return CheckResult(name, "ok", detail="sandbox flags accepted")
    err = _text(proc.stderr) or _text(proc.stdout) or "the container read nothing back"
    if "No such image" in err:
        return CheckResult(name, "fail", detail=err, fix=_mount_failure_fix(err))
    return CheckResult(
        name, "fail", detail=err,
        fix="The workdir mounts (check above passed) but your Docker "
            "rejected one of the sandbox flags. Please report this output — "
            "we will not loosen the sandbox to work around it.",
    )


def check_local_datasets(raw: str | None = None) -> CheckResult:
    """Every label this host advertises must resolve to a readable directory.

    parse_local_data checks the label charset, that the path is absolute,
    that it carries no ':' that would re-read as a second mount, and that no
    label is mapped twice. It never stats the path. So a typo advertises a
    dataset this host cannot serve — and because the coordinator's
    fail-closed placement gate trusts that advertisement, this host becomes
    the ONLY one eligible for the job, and every retry comes back here.
    """
    from flashnode.config.local_data import (
        LOCAL_DATA_ENV,
        LocalDataError,
        parse_local_data,
    )

    name = "local datasets readable"
    value = os.environ.get(LOCAL_DATA_ENV) if raw is None else raw
    try:
        mapping = parse_local_data(value)
    except LocalDataError as exc:
        return CheckResult(
            name, "fail", detail=str(exc),
            fix=f"{LOCAL_DATA_ENV} must look like "
                "label=/absolute/path,other=/absolute/path2. Fix it and "
                "re-run `flashnode doctor`.",
        )
    if not mapping:
        return CheckResult(name, "ok", detail="none configured")
    problems = []
    for label, path in sorted(mapping.items()):
        p = Path(path)
        if not p.exists():
            problems.append(f"{label}: {path} does not exist")
        elif not p.is_dir():
            problems.append(f"{label}: {path} is not a directory")
        elif not os.access(p, os.R_OK | os.X_OK):
            problems.append(f"{label}: {path} is not readable")
    if problems:
        return CheckResult(
            name, "fail", detail="\n".join(problems),
            fix=f"Point {LOCAL_DATA_ENV} at directories that exist and are "
                "readable, or remove the labels you cannot serve — the "
                "coordinator sends local-data jobs ONLY to hosts advertising "
                "them, so a bad label strands the job here.",
        )
    return CheckResult(name, "ok", detail=", ".join(sorted(mapping)))


def run_checks(
    *,
    pull: bool,
    run: CommandRunner | None = None,
    which: Callable[[str], str | None] | None = None,
    workdir: Path | None = None,
    raw_local_data: str | None = None,
) -> list[CheckResult]:
    """Run every check, in order, skipping what a prior failure makes
    meaningless.

    `pull=False` is the `flashnode work` path: an agent is a long-running
    daemon on someone else's machine, and a transient registry blip must not
    stop one whose images are already cached (spec §4.1).
    """
    # Same call-time resolution as check_cli_on_path, for the same reason.
    run = run or run_command
    which = which or shutil.which
    base = workdir if workdir is not None else default_workdir()
    # Every container-level check, in order, each paired with the callable
    # that runs it. A failure skips the rest of THIS list; the local-dataset
    # check is independent of Docker and always runs, after the loop.
    staged: list[tuple[str, Callable[[], CheckResult]]] = [
        ("docker CLI on PATH", lambda: check_cli_on_path(which=which)),
        ("docker engine reachable", lambda: check_engine(run)),
    ]
    if pull:
        staged.append(("pull a curated image", lambda: check_pull(run)))
    staged += [
        ("workdir bind-mounts", lambda: check_workdir_mount(run, base)),
        ("a hardened container runs", lambda: check_hardened_run(run, base)),
    ]

    results: list[CheckResult] = []
    stopped = False
    for label, check in staged:
        if stopped:
            results.append(CheckResult(label, "skip", detail="needs the check above"))
            continue
        results.append(check())
        if results[-1].status != "ok":
            stopped = True
    results.append(check_local_datasets(raw=raw_local_data))
    return results


def format_results(results: Sequence[CheckResult]) -> str:
    lines = []
    for r in results:
        tag = {"ok": "[ok]  ", "fail": "[FAIL]", "skip": "[skip]"}[r.status]
        head = r.detail.splitlines()[0] if r.detail else ""
        lines.append(f"  {tag} {r.name:<30} {head}".rstrip())
        for extra in r.detail.splitlines()[1:]:
            lines.append(f"         {extra}")
        if r.fix:
            lines.append(f"         fix: {r.fix}")
    failed = sum(1 for r in results if r.status == "fail")
    skipped = sum(1 for r in results if r.status == "skip")
    if failed or skipped:
        parts = []
        if failed:
            parts.append(f"{failed} check{'s' if failed != 1 else ''} failed")
        if skipped:
            parts.append(f"{skipped} skipped")
        lines.append(
            ", ".join(parts) + ". Fix the above, then re-run `flashnode doctor`."
        )
    else:
        lines.append("All checks passed. Start contributing with "
                     "`flashnode work --runner docker`.")
    return "\n".join(lines)


def exit_code(results: Sequence[CheckResult]) -> int:
    """Skipped counts as not-passed. A host whose checks did not run has not
    been certified, and calling it healthy is the exact failure this command
    removes."""
    return 0 if all(r.status == "ok" for r in results) else 1


def doctor_main(argv: list[str]) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="flashnode doctor",
        description="Check this machine can actually run FlashML tasks.",
    )
    parser.parse_args(argv)
    results = run_checks(pull=True)
    print(format_results(results))
    return exit_code(results)
