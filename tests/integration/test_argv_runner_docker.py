"""Real-docker proof that the sandbox flags are enforced, not just passed.

Opt-in: pytest -m integration. Auto-skips without a docker daemon.

Uses the `docker_workdir` fixture (tests/conftest.py), NOT pytest's
`tmp_path`. On macOS, colima/Docker Desktop only share $HOME by default, so
`tmp_path` (under /var/folders) either bind-mounts as an empty directory or
fails the mount outright — a task using it can raise TaskExecutionError for
a completely unrelated reason (missing file, failed mount) while the
assertion `pytest.raises(TaskExecutionError)` still passes. That would make
the network/read-only tests below green without ever exercising the
sandbox flag they exist to prove. See tests/conftest.py and AGENTS.md for
the same constraint documented for the real agent's FLASHNODE_WORKDIR.
"""
import re
import shutil
import subprocess

import pytest

from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.runner import TaskExecutionError

IMAGE = "python:3.11-alpine"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _docker_available(), reason="needs a docker daemon")]


def _runner():
    return ArgvDockerRunner(allowed_images=frozenset({IMAGE}), timeout_seconds=120.0)


def test_argv_task_runs_and_produces_metrics(docker_workdir):
    payload = {"image": IMAGE, "task_id": "task-000",
               "argv": ["python", "-c",
                        "open('/work/out/metrics.json','w').write('{\"acc\": 1.0}')"]}
    outdir = _runner().run(payload, docker_workdir, {})
    assert (outdir / "metrics.json").read_text() == '{"acc": 1.0}'


def test_network_is_really_off(docker_workdir):
    # PROBE_OK is printed before the forbidden call so a container that
    # never gets that far (bad image, broken bind mount, python missing)
    # is distinguishable from one whose *network* attempt was denied — the
    # thing this test actually exists to prove.
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "import socket, sys\n"
                        "print('PROBE_OK', file=sys.stderr); sys.stderr.flush()\n"
                        "socket.create_connection(('1.1.1.1', 53), 5)\n"]}
    with pytest.raises(TaskExecutionError) as exc_info:
        _runner().run(payload, docker_workdir, {})
    message = str(exc_info.value)
    assert "PROBE_OK" in message, "container never got to the network call — not a sandbox failure"
    # --network none removes every route but loopback, so connect() fails
    # immediately with ENETUNREACH ("Network is unreachable") or ENETDOWN-
    # adjacent EHOSTUNREACH ("No route to host"). Deliberately NOT matching
    # "connection refused" or "timed out": those are signatures of a
    # REACHABLE network (something answered, or nothing did but the route
    # existed) — on a host with restricted egress this test would pass even
    # with --network none silently removed, which defeats the one thing
    # this test exists to prove (Minor #8).
    assert re.search(r"network is unreachable|no route to host",
                      message, re.IGNORECASE), f"not a network-denial signature: {message!r}"


def test_rootfs_is_really_read_only(docker_workdir):
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "import sys\n"
                        "print('PROBE_OK', file=sys.stderr); sys.stderr.flush()\n"
                        "open('/etc/passwd','a').write('x')\n"]}
    with pytest.raises(TaskExecutionError) as exc_info:
        _runner().run(payload, docker_workdir, {})
    message = str(exc_info.value)
    assert "PROBE_OK" in message, "container never got to the write — not a sandbox failure"
    # --read-only surfaces as EROFS ("Read-only file system") from the
    # kernel, distinct from a permission or missing-path error.
    assert "read-only file system" in message.lower(), f"not a read-only-fs signature: {message!r}"


def test_inputs_are_visible_at_work_inputs(docker_workdir):
    (docker_workdir / "inputs").mkdir()
    (docker_workdir / "inputs" / "data.txt").write_text("hello")
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "d=open('/work/inputs/data.txt').read();"
                        "open('/work/out/metrics.json','w').write('{\"n\": %d}' % len(d))"]}
    outdir = _runner().run(payload, docker_workdir, {"data": docker_workdir / "inputs" / "data.txt"})
    assert '"n": 5' in (outdir / "metrics.json").read_text()
