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
    NON_BLOCKING_STATUSES,
    CheckResult,
    check_cli_on_path,
    check_engine,
    check_gpus,
    check_hardened_run,
    check_local_datasets,
    check_pull,
    check_workdir_mount,
    default_workdir,
    exit_code,
    format_results,
    run_checks,
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


def test_local_datasets_passes_when_none_are_configured():
    result = check_local_datasets(raw="")
    assert result.status == "ok"
    assert "none" in result.detail.lower()


def test_local_datasets_passes_for_a_readable_directory(tmp_path):
    data = tmp_path / "patients"
    data.mkdir()
    result = check_local_datasets(raw=f"patients={data}")
    assert result.status == "ok"
    assert "patients" in result.detail


def test_local_datasets_fails_on_a_path_that_does_not_exist_and_names_the_label(tmp_path):
    """The typo case. It parses, it advertises, the placement gate believes
    it, and every attempt routes back to this host."""
    result = check_local_datasets(raw=f"patients={tmp_path / 'typo'}")
    assert result.status == "fail"
    assert "patients" in result.detail


def test_local_datasets_fails_when_the_path_is_a_file_not_a_directory(tmp_path):
    f = tmp_path / "patients.csv"
    f.write_text("x")
    result = check_local_datasets(raw=f"patients={f}")
    assert result.status == "fail"
    assert "directory" in result.detail.lower()


def test_local_datasets_fails_on_an_unreadable_directory(tmp_path):
    import os

    import pytest as _pytest

    if os.getuid() == 0:
        _pytest.skip("root reads everything")
    data = tmp_path / "locked"
    data.mkdir(mode=0o000)
    try:
        result = check_local_datasets(raw=f"locked={data}")
        assert result.status == "fail"
    finally:
        data.chmod(0o755)


def test_local_datasets_reports_a_malformed_value_rather_than_raising():
    result = check_local_datasets(raw="patients")  # no '=' at all
    assert result.status == "fail"
    assert "FLASHNODE_LOCAL_DATA" in result.fix


def test_local_datasets_lists_every_bad_label_not_just_the_first(tmp_path):
    result = check_local_datasets(raw=f"a={tmp_path/'x'},b={tmp_path/'y'}")
    assert "a" in result.detail and "b" in result.detail


def test_run_checks_skips_container_checks_when_the_engine_is_down_but_still_checks_datasets():
    results = run_checks(
        pull=True,
        run=_runner(_proc(1, stderr=ENGINE_PING_500)),
        which=lambda _: "/usr/local/bin/docker",
        raw_local_data="",
    )
    by_name = {r.name: r.status for r in results}
    assert by_name["docker engine reachable"] == "fail"
    assert by_name["pull a curated image"] == "skip"
    assert by_name["a hardened container runs"] == "skip"
    assert by_name["local datasets readable"] == "ok"
    assert exit_code(results) == 1


def test_run_checks_without_pull_never_calls_docker_pull(tmp_path):
    """The `flashnode work` path: a registry blip must not stop an agent
    whose images are cached (spec 4.1)."""
    calls = []

    def run(argv, **kwargs):
        calls.append(list(argv))
        return _proc(0, stdout="flashnode-doctor")

    run_checks(pull=False, run=run, which=lambda _: "/usr/local/bin/docker",
               workdir=tmp_path, raw_local_data="")
    assert not any(c[:2] == ["docker", "pull"] for c in calls)


def test_patching_shutil_which_is_observed_by_the_doctor(monkeypatch):
    """Regression. `def check_cli_on_path(which=shutil.which)` binds the
    ORIGINAL function at import, so monkeypatching the module attribute —
    the idiom the rest of this suite uses — silently failed to reach it.

    That was not a testing inconvenience. It let the `flashnode work` gate
    ignore a patched `which`, execute a REAL `docker run` inside unit tests,
    and pass or fail with whatever state the machine's Docker happened to be
    in. Resolve side effects at call time, not in a default argument.
    """
    monkeypatch.setattr("shutil.which", lambda name: None)
    assert check_cli_on_path().status == "fail"

    results = run_checks(pull=True, raw_local_data="")
    assert results[0].status == "fail"
    assert exit_code(results) == 1


def test_run_checks_defaults_do_not_capture_run_command_at_import(monkeypatch):
    """Same bug class for the command runner: patching the module attribute
    must be observed, or the gate shells out for real under test."""
    calls = []
    monkeypatch.setattr("shutil.which", lambda name: "/usr/local/bin/docker")
    monkeypatch.setattr(
        "flashnode.doctor.run_command",
        lambda argv, **kw: calls.append(list(argv)) or _proc(1, stderr=ENGINE_PING_500),
    )
    results = run_checks(pull=True, raw_local_data="")
    assert calls, "run_checks did not use the patched run_command"
    assert results[1].status == "fail"


# -- check 7: GPUs, and the first NON-GATING check in this module -------------
#
# Every check before this one blocks: `exit_code` and the `flashnode work`
# gate both treat anything that is not "ok" as a refusal, which is right for
# six checks about whether this machine can run a task at all. It is wrong
# for this one. Most volunteers have no GPU and must keep taking CPU work —
# a gating GPU check would lock the entire existing fleet out of the network
# on upgrade. So "info" exists: reported, never blocking.


def test_the_gpu_check_reports_what_the_probe_found():
    from flashruntime.protocol.v1alpha1 import GpuInfo

    result = check_gpus(probe=lambda: [
        GpuInfo(index=0, name="NVIDIA GeForce RTX 4090", memory_total_mb=24564),
        GpuInfo(index=1, name="NVIDIA GeForce RTX 4090", memory_total_mb=24564),
    ])
    assert result.status == "ok"
    assert "2" in result.detail
    assert "RTX 4090" in result.detail


def test_the_gpu_check_says_plainly_that_it_found_nothing():
    result = check_gpus(probe=lambda: [])
    assert result.status == "info"
    assert "no GPU" in result.detail
    # The point of the check, per spec §8: a host who BELIEVES they
    # contributed a GPU finds out here that they did not.
    assert result.fix


def test_no_gpu_is_not_a_failure():
    assert check_gpus(probe=lambda: []).status in NON_BLOCKING_STATUSES
    assert exit_code([CheckResult("a", "ok"), check_gpus(probe=lambda: [])]) == 0


def test_the_gpu_check_never_raises():
    def boom():
        raise RuntimeError("nvidia-smi went sideways")

    result = check_gpus(probe=boom)
    assert result.status in NON_BLOCKING_STATUSES


def test_format_shows_an_info_line_and_calls_nothing_failed():
    text = format_results([
        CheckResult("docker CLI on PATH", "ok", detail="/usr/local/bin/docker"),
        CheckResult("GPU devices", "info", detail="no GPU detected",
                    fix="Install the NVIDIA driver."),
    ])
    assert "[info]" in text
    assert "no GPU detected" in text
    assert "failed" not in text


def test_run_checks_includes_the_gpu_check_and_a_gpu_less_host_still_passes(tmp_path):
    results = run_checks(
        pull=False,
        run=_runner(_proc(0, stdout="flashnode-doctor")),
        which=lambda _: "/usr/local/bin/docker",
        workdir=tmp_path,
        raw_local_data="",
        gpu_probe=lambda: [],
    )
    by_name = {r.name: r.status for r in results}
    assert "GPU devices" in by_name
    assert by_name["GPU devices"] == "info"
    assert exit_code(results) == 0, format_results(results)


def test_the_gpu_check_runs_even_when_docker_is_down():
    """It has nothing to do with the engine, like the local-dataset check —
    a host debugging Docker should still learn whether their GPU is seen."""
    results = run_checks(
        pull=True,
        run=_runner(_proc(1, stderr=ENGINE_PING_500)),
        which=lambda _: "/usr/local/bin/docker",
        raw_local_data="",
        gpu_probe=lambda: [],
    )
    by_name = {r.name: r.status for r in results}
    assert by_name["GPU devices"] == "info"      # reported, not "skip"
    assert exit_code(results) == 1               # the engine still fails it


def test_run_checks_does_not_shell_out_to_nvidia_smi_by_default(monkeypatch):
    """Same rule as `run` and `which`: resolved at call time, so the suite
    never touches the machine's real driver state."""
    calls = []
    monkeypatch.setattr("flashnode.doctor.probe_gpus",
                        lambda: calls.append(1) or [])
    monkeypatch.setattr("shutil.which", lambda name: None)
    run_checks(pull=True, raw_local_data="")
    assert calls, "run_checks did not use the patched probe_gpus"


def test_an_unknown_status_still_blocks():
    """`exit_code` widened from "everything ok" to "nothing blocking". It
    must stay fail-closed: a status this module does not know is not a pass."""
    assert exit_code([CheckResult("a", "ok"), CheckResult("b", "mystery")]) == 1
