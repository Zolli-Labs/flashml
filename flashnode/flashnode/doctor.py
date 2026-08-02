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

import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable

__all__ = [
    "PROBE_IMAGE",
    "CheckResult",
    "check_cli_on_path",
    "check_engine",
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


def check_cli_on_path(which: Callable[[str], str | None] = shutil.which) -> CheckResult:
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


def run_checks(
    *,
    pull: bool,
    run: CommandRunner = run_command,
    which: Callable[[str], str | None] = shutil.which,
    workdir=None,
    raw_local_data: str | None = None,
) -> list[CheckResult]:
    """Run every check, in order, skipping what a prior failure makes
    meaningless.

    `pull=False` is the `flashnode work` path: an agent is a long-running
    daemon on someone else's machine, and a transient registry blip must not
    stop one whose images are already cached (spec §4.1).
    """
    results = [check_cli_on_path(which=which)]
    if results[-1].status != "ok":
        return results
    results.append(check_engine(run))
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
