"""Unit tests for the host doctor. No Docker daemon required — every check
takes its subprocess/which call as a parameter, so failures are driven by
recorded `docker` output rather than by a broken machine.

The stderr strings below are the REAL ones from the 2026-08-02 §10 attempt
(flashml-cloud PROGRESS.md). They are the reason this command exists; keep
them verbatim.
"""

from __future__ import annotations

import subprocess

from flashnode.doctor import (
    CheckResult,
    check_cli_on_path,
    check_engine,
    exit_code,
    format_results,
)

# The Windows failure that stopped the 2026-08-02 run-through.
ENGINE_PING_500 = (
    "error during connect: Get "
    '"http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/_ping": '
    "The system cannot find the file specified."
)


def _proc(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["docker"], returncode=returncode,
        stdout=stdout.encode(), stderr=stderr.encode(),
    )


def _runner(proc):
    def run(argv, **kwargs):
        return proc
    return run


def test_cli_on_path_passes_when_docker_is_found():
    result = check_cli_on_path(which=lambda _: "/usr/local/bin/docker")
    assert result.status == "ok"
    assert "/usr/local/bin/docker" in result.detail


def test_cli_on_path_fails_when_docker_is_absent():
    result = check_cli_on_path(which=lambda _: None)
    assert result.status == "fail"
    assert "install" in result.fix.lower()


def test_engine_passes_and_reports_the_server_version():
    result = check_engine(_runner(_proc(0, stdout="27.4.0\n")))
    assert result.status == "ok"
    assert "27.4.0" in result.detail


def test_engine_fails_on_a_ping_500_and_says_to_start_docker():
    """The exact Windows failure. `shutil.which` cannot see this: the binary
    is on PATH and the daemon behind it is dead."""
    result = check_engine(_runner(_proc(1, stderr=ENGINE_PING_500)))
    assert result.status == "fail"
    assert "_ping" in result.detail
    assert "start" in result.fix.lower() and "docker" in result.fix.lower()


def test_engine_fails_rather_than_raises_when_docker_is_missing_mid_run():
    def run(argv, **kwargs):
        raise FileNotFoundError("docker")
    result = check_engine(run)
    assert result.status == "fail"


def test_exit_code_is_zero_only_when_everything_passed():
    assert exit_code([CheckResult("a", "ok"), CheckResult("b", "ok")]) == 0


def test_a_skipped_check_exits_nonzero():
    """A host with unrun checks has not been certified. Reporting it healthy
    is the failure mode this command exists to remove (spec §2)."""
    assert exit_code([CheckResult("a", "ok"), CheckResult("b", "skip")]) == 1


def test_a_failed_check_exits_nonzero():
    assert exit_code([CheckResult("a", "fail")]) == 1


def test_format_shows_the_fix_for_a_failure_and_a_trailing_count():
    text = format_results([
        CheckResult("docker CLI on PATH", "ok", detail="/usr/local/bin/docker"),
        CheckResult("docker engine reachable", "fail",
                    detail=ENGINE_PING_500, fix="Start Docker Desktop."),
        CheckResult("pull a curated image", "skip", detail="needs the engine"),
    ])
    assert "[ok]" in text and "[FAIL]" in text and "[skip]" in text
    assert "Start Docker Desktop." in text
    assert "1 check failed, 1 skipped" in text


def test_format_says_nothing_alarming_when_all_pass():
    text = format_results([CheckResult("docker CLI on PATH", "ok")])
    assert "failed" not in text
