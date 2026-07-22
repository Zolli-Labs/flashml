"""LocalProcessLauncher runs real subprocesses — these tests execute
`python -c` children and assert the Launcher contract."""

from __future__ import annotations

import sys

import pytest


def _launcher(tmp_path):
    from flashruntime.launchers.local import LocalProcessLauncher

    return LocalProcessLauncher(output_root=tmp_path / "runs")


def _spec(code: str, workdir: str):
    from flashruntime.strategies import LaunchSpec

    return LaunchSpec(argv=[sys.executable, "-c", code], workdir_hint=workdir)


def test_success_sets_contract_env_and_succeeds(tmp_path):
    from flashruntime.launchers import LaunchState

    code = (
        "import os, pathlib;"
        "out = pathlib.Path(os.environ['FLASHML_OUTPUT_DIR']);"
        "(out / 'proof.txt').write_text(os.environ['FLASHML_JOB_ID'] + ':' + os.environ['FLASHML_ATTEMPT_ID']);"
        "print('hello from child')"
    )
    handle = _launcher(tmp_path).launch(_spec(code, str(tmp_path)), "j1", "a1")
    assert handle.wait(timeout_seconds=30) is LaunchState.SUCCEEDED
    assert (handle.output_dir / "proof.txt").read_text() == "j1:a1"
    assert "hello from child" in handle.logs()
    assert handle.execution_id  # pid string


def test_ckpt_dir_is_job_scoped_not_attempt_scoped(tmp_path):
    code = "import os; print(os.environ['FLASHML_CKPT_DIR'])"
    launcher = _launcher(tmp_path)
    h1 = launcher.launch(_spec(code, str(tmp_path)), "j1", "a1")
    h2 = launcher.launch(_spec(code, str(tmp_path)), "j1", "a2")
    h1.wait(30), h2.wait(30)
    assert h1.logs().strip() == h2.logs().strip()  # same ckpt tree across attempts


def test_nonzero_exit_reports_failed_and_caches(tmp_path):
    from flashruntime.launchers import LaunchState

    handle = _launcher(tmp_path).launch(_spec("raise SystemExit(3)", str(tmp_path)), "j1", "a1")
    assert handle.wait(30) is LaunchState.FAILED
    assert handle.poll() is LaunchState.FAILED  # terminal state cached


def test_preflight_failures_raise_launch_error(tmp_path):
    from flashruntime.launchers import LaunchError
    from flashruntime.strategies import LaunchSpec

    with pytest.raises(LaunchError, match="workdir"):
        _launcher(tmp_path).launch(
            LaunchSpec(argv=["true"], workdir_hint=str(tmp_path / "nope")), "j", "a"
        )
    with pytest.raises(LaunchError, match="argv"):
        _launcher(tmp_path).launch(LaunchSpec(argv=[], workdir_hint=str(tmp_path)), "j", "a")
