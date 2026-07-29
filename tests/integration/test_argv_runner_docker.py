"""Real-docker proof that the sandbox flags are enforced, not just passed.

Opt-in: pytest -m integration. Auto-skips without a docker daemon.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.runner import TaskExecutionError

pytestmark = pytest.mark.integration

IMAGE = "python:3.11-alpine"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _docker_available(), reason="needs a docker daemon")]


def _runner():
    return ArgvDockerRunner(allowed_images=frozenset({IMAGE}), timeout_seconds=120.0)


def test_argv_task_runs_and_produces_metrics(tmp_path):
    payload = {"image": IMAGE, "task_id": "task-000",
               "argv": ["python", "-c",
                        "open('/work/out/metrics.json','w').write('{\"acc\": 1.0}')"]}
    outdir = _runner().run(payload, tmp_path, {})
    assert (outdir / "metrics.json").read_text() == '{"acc": 1.0}'


def test_network_is_really_off(tmp_path):
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "import socket; socket.create_connection(('1.1.1.1', 53), 5)"]}
    with pytest.raises(TaskExecutionError):
        _runner().run(payload, tmp_path, {})


def test_rootfs_is_really_read_only(tmp_path):
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c", "open('/etc/passwd','a').write('x')"]}
    with pytest.raises(TaskExecutionError):
        _runner().run(payload, tmp_path, {})


def test_inputs_are_visible_at_work_inputs(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "data.txt").write_text("hello")
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "d=open('/work/inputs/data.txt').read();"
                        "open('/work/out/metrics.json','w').write('{\"n\": %d}' % len(d))"]}
    outdir = _runner().run(payload, tmp_path, {"data": tmp_path / "inputs" / "data.txt"})
    assert '"n": 5' in (outdir / "metrics.json").read_text()
