"""Unit tests for the host doctor. No Docker daemon required — every check
takes its subprocess/which call as a parameter, so failures are driven by
recorded `docker` output rather than by a broken machine.

The stderr strings below are the REAL ones from the 2026-08-02 §10 attempt
(flashml-cloud PROGRESS.md). They are the reason this command exists; keep
them verbatim.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from flashnode.doctor import (
    CheckResult,
    check_cli_on_path,
    check_engine,
    check_hardened_run,
    check_pull,
    check_workdir_mount,
    default_workdir,
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


# The macOS failure that stopped the 2026-08-02 run-through. The engine was
# healthy; a credential helper named in ~/.docker/config.json was not
# installed, and Docker consults it when authenticating a registry pull.
CREDS_HELPER_MISSING = (
    'error getting credentials - err: exec: "docker-credential-desktop": '
    "executable file not found in $PATH, out: ``"
)


def test_pull_passes_and_names_the_image():
    result = check_pull(_runner(_proc(0, stdout="Status: Image is up to date")))
    assert result.status == "ok"
    assert "flashml-python-slim" in result.detail


def test_pull_fails_on_a_missing_credential_helper_and_says_it_needs_no_login():
    result = check_pull(_runner(_proc(1, stderr=CREDS_HELPER_MISSING)))
    assert result.status == "fail"
    assert "docker-credential-desktop" in result.detail
    assert "credsStore" in result.fix
    assert "no login" in result.fix.lower()


def test_pull_fails_on_denied_and_points_at_image_visibility():
    """The GHCR-private outage: every job died at execution after signup,
    install and enrolment all appeared to work."""
    result = check_pull(_runner(_proc(1, stderr="denied: denied")))
    assert result.status == "fail"
    assert "public" in result.fix.lower()


def test_pull_uses_the_small_image_never_pytorch():
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = list(argv)
        return _proc(0)

    check_pull(run)
    assert seen["argv"][:2] == ["docker", "pull"]
    assert "python-slim" in seen["argv"][2]
    assert "pytorch" not in seen["argv"][2]


def test_pull_reports_a_timeout_rather_than_raising():
    def run(argv, **kwargs):
        raise subprocess.TimeoutExpired(cmd="docker", timeout=300)
    assert check_pull(run).status == "fail"


def test_workdir_mount_passes_when_the_container_reads_the_probe_back(tmp_path):
    result = check_workdir_mount(_runner(_proc(0, stdout="flashnode-doctor")), tmp_path)
    assert result.status == "ok"


def test_workdir_mount_writes_a_probe_file_the_container_can_read(tmp_path):
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = list(argv)
        # The file must exist on the host BEFORE the container runs.
        assert (tmp_path / "flashnode-doctor-probe.txt").is_file()
        return _proc(0, stdout="flashnode-doctor")

    check_workdir_mount(run, tmp_path)
    assert "-v" in seen["argv"]


def test_workdir_mount_fails_when_the_mount_is_empty_and_names_flashnode_workdir(tmp_path):
    """The colima gotcha: Docker Desktop and colima share only $HOME on
    macOS, so a workdir under /var/folders mounts as an EMPTY directory and
    every task silently sees no inputs."""
    result = check_workdir_mount(
        _runner(_proc(1, stderr="FileNotFoundError: /work/flashnode-doctor-probe.txt")),
        tmp_path,
    )
    assert result.status == "fail"
    assert "FLASHNODE_WORKDIR" in result.fix
    assert "$HOME" in result.fix


def test_workdir_mount_fails_when_the_container_reads_the_wrong_content(tmp_path):
    result = check_workdir_mount(_runner(_proc(0, stdout="something else")), tmp_path)
    assert result.status == "fail"


def test_workdir_mount_never_pulls(tmp_path):
    """Startup must not depend on a registry (spec 4.1)."""
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = list(argv)
        return _proc(0, stdout="flashnode-doctor")

    check_workdir_mount(run, tmp_path)
    assert "--pull=never" in seen["argv"]


def test_workdir_mount_says_run_the_doctor_when_the_image_is_not_cached(tmp_path):
    result = check_workdir_mount(
        _runner(_proc(125, stderr="Error: No such image: "
                                  "ghcr.io/zolli-labs/flashml-python-slim:2026.08.1")),
        tmp_path,
    )
    assert result.status == "fail"
    assert "flashnode doctor" in result.fix


def test_workdir_mount_cleans_up_its_probe_file(tmp_path):
    check_workdir_mount(_runner(_proc(0, stdout="flashnode-doctor")), tmp_path)
    assert not (tmp_path / "flashnode-doctor-probe.txt").exists()


def test_default_workdir_prefers_flashnode_workdir(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHNODE_WORKDIR", str(tmp_path))
    assert default_workdir() == tmp_path


def test_default_workdir_falls_back_to_the_system_temp_dir(monkeypatch):
    """Deliberately the same default ExecutorLoop uses (workdir_base=None),
    so the doctor fails on exactly the machines the agent would."""
    monkeypatch.delenv("FLASHNODE_WORKDIR", raising=False)
    assert default_workdir() == Path(tempfile.gettempdir())


def test_hardened_run_passes_when_the_probe_reads_back(tmp_path):
    result = check_hardened_run(_runner(_proc(0, stdout="flashnode-doctor")), tmp_path)
    assert result.status == "ok"


def test_hardened_run_carries_the_real_sandbox_flags(tmp_path):
    """Not a re-implementation of harden_args — the real one, so a change
    there is exercised here rather than drifting silently."""
    seen = {}

    def run(argv, **kwargs):
        seen["argv"] = list(argv)
        return _proc(0, stdout="flashnode-doctor")

    check_hardened_run(run, tmp_path)
    argv = seen["argv"]
    assert "--network" in argv and "none" in argv
    assert "--read-only" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--pull=never" in argv


def test_hardened_run_fails_on_a_rejected_flag_and_blames_the_flags(tmp_path):
    result = check_hardened_run(
        _runner(_proc(125, stderr="docker: Error response from daemon: "
                                  "invalid argument for --pids-limit")),
        tmp_path,
    )
    assert result.status == "fail"
    assert "--pids-limit" in result.detail
    assert "report" in result.fix.lower()


def test_hardened_run_fails_when_the_user_flag_cannot_be_built(tmp_path, monkeypatch):
    """_user_flag raises on a platform with no getuid and not win32 — a
    refusal to run unprivileged-in-name-only. The doctor must report that,
    not crash."""
    import flashnode.doctor as doctor_mod

    def boom(*a, **k):
        raise RuntimeError("cannot determine a safe --user for platform 'sunos5'")

    monkeypatch.setattr(doctor_mod, "harden_args", boom)
    result = check_hardened_run(_runner(_proc(0)), tmp_path)
    assert result.status == "fail"
    assert "safe --user" in result.detail


def test_hardened_run_cleans_up_its_probe_file(tmp_path):
    check_hardened_run(_runner(_proc(0, stdout="flashnode-doctor")), tmp_path)
    assert not (tmp_path / "flashnode-doctor-probe.txt").exists()
