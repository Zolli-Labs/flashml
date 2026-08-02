"""The doctor against a real Docker daemon.

Opt-in: pytest -m integration. Auto-skips without a daemon, matching
tests/integration/test_argv_runner_docker.py.

Uses the `docker_workdir` fixture (tests/conftest.py), NOT pytest's
tmp_path: on macOS colima and Docker Desktop share only $HOME, so tmp_path
bind-mounts as an empty directory — which is the very condition check 4
exists to detect, and would make this test fail for the right reason on a
perfectly good machine.
"""

import shutil
import subprocess

import pytest

from flashnode.doctor import exit_code, format_results, run_checks

pytestmark = pytest.mark.integration


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


@pytest.mark.skipif(not _docker_available(), reason="needs a real docker daemon")
def test_every_check_passes_on_a_healthy_host(docker_workdir):
    results = run_checks(pull=True, workdir=docker_workdir, raw_local_data="")
    assert len(results) == 6, [r.name for r in results]
    assert exit_code(results) == 0, "\n" + format_results(results)


@pytest.mark.skipif(not _docker_available(), reason="needs a real docker daemon")
def test_a_bad_local_dataset_label_is_caught_against_a_real_filesystem(docker_workdir):
    results = run_checks(
        pull=True, workdir=docker_workdir,
        raw_local_data=f"patients={docker_workdir / 'not-here'}",
    )
    bad = [r for r in results if r.name == "local datasets readable"]
    assert bad and bad[0].status == "fail"
    assert exit_code(results) == 1
