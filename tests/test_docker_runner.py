"""DockerRunner (Tier 2) tests — written before the implementation (TDD).

The runner must satisfy the same `run(payload, workdir, inputs) → outdir`
interface as SubprocessRunner, but execute inside a container with the
security contract from AGENTS.md: allowlisted image, `--network none`,
cpu/memory limits, non-root-equivalent user mapping, read-only rootfs,
wall-clock timeout, no docker socket exposure to the workload.

Unit tests capture the constructed `docker run` argv via a stubbed
subprocess; a final smoke test uses real Docker and auto-skips without it.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from flashnode.executor.docker_runner import DockerRunner
from flashnode.executor.runner import TaskExecutionError

IMAGE = "ghcr.io/example/trial:v1"


def _payload(**over):
    payload = {
        "module": "flashml_workloads.sklearn_trial",
        "task_id": "t1",
        "params": {"C": 1.0},
        "image": IMAGE,
    }
    payload.update(over)
    return payload


@pytest.fixture()
def capture(monkeypatch):
    """Stub subprocess.run inside the docker_runner module; record argv and
    fabricate a successful container run that wrote metrics.json."""
    calls: list[dict] = []

    def fake_run(argv, **kw):
        calls.append({"argv": argv, "kw": kw})
        # emulate the containerized task writing its output
        for i, part in enumerate(argv):
            if part == "-v":
                host = argv[i + 1].split(":", 1)[0]
                out = Path(host) / "out"
                out.mkdir(parents=True, exist_ok=True)
                (out / "metrics.json").write_text("{}")
        return subprocess.CompletedProcess(argv, 0, stdout=b"", stderr=b"")

    monkeypatch.setattr("flashnode.executor.docker_runner.subprocess.run", fake_run)
    return calls


def _runner(**over):
    kw = dict(
        allowed_images=frozenset({IMAGE}),
        allowed_modules=frozenset({"flashml_workloads.sklearn_trial"}),
        cpus=2.0,
        memory_gb=2.0,
    )
    kw.update(over)
    return DockerRunner(**kw)


def test_refuses_unlisted_image_before_any_subprocess(capture, tmp_path):
    with pytest.raises(TaskExecutionError, match="image .* not allowlisted"):
        _runner().run(_payload(image="evil/image:latest"), tmp_path, {})
    assert capture == []  # fail closed: docker never invoked


def test_refuses_missing_image_and_unlisted_module(capture, tmp_path):
    with pytest.raises(TaskExecutionError, match="no image"):
        _runner().run(_payload(image=None), tmp_path, {})
    with pytest.raises(TaskExecutionError, match="module .* not allowlisted"):
        _runner().run(_payload(module="os.system"), tmp_path, {})
    assert capture == []


def test_docker_command_carries_the_security_contract(capture, tmp_path):
    _runner().run(_payload(), tmp_path, {})
    argv = capture[0]["argv"]
    assert argv[:2] == ["docker", "run"]
    joined = " ".join(argv)
    assert "--rm" in argv
    assert "--network none" in joined
    assert "--cpus 2.0" in joined
    assert "--memory 2.0g" in joined
    assert "--read-only" in argv
    assert f"{tmp_path}:/work" in joined  # workdir bind-mounted at /work
    # image comes before the in-container command
    assert argv.index(IMAGE) < argv.index("python")
    assert joined.endswith(
        f"{IMAGE} python -m flashml_workloads.sklearn_trial --spec /work/spec.json --out /work/out"
    )


def test_spec_json_uses_container_paths_for_inputs(capture, tmp_path):
    host_input = tmp_path / "inputs" / "data.csv"
    host_input.parent.mkdir(parents=True)
    host_input.write_text("1,2\n")
    _runner().run(_payload(), tmp_path, {"dataset": host_input})
    spec = json.loads((tmp_path / "spec.json").read_text())
    assert spec["inputs"]["dataset"] == "/work/inputs/data.csv"  # not the host path


def test_nonzero_container_exit_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "flashnode.executor.docker_runner.subprocess.run",
        lambda argv, **kw: subprocess.CompletedProcess(argv, 137, stdout=b"", stderr=b"oom-killed"),
    )
    with pytest.raises(TaskExecutionError, match="exited 137"):
        _runner().run(_payload(), tmp_path, {})


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return subprocess.run(["docker", "info"], capture_output=True, timeout=10).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


# docker_workdir fixture: see tests/conftest.py — it is shared with
# tests/integration/ rather than redefined here.


@pytest.mark.skipif(not _docker_available(), reason="no reachable Docker daemon")
def test_real_docker_runs_a_task_end_to_end(docker_workdir):
    """Build a micro image containing a conforming task module and run it
    through the real path. Slow-ish; auto-skips without Docker."""
    tmp_path = docker_workdir
    image = "flashnode-test-task:local"
    ctx = tmp_path / "ctx"
    pkg = ctx / "testpkg"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "hello.py").write_text(
        "import argparse, json, pathlib\n"
        "p = argparse.ArgumentParser(); p.add_argument('--spec'); p.add_argument('--out')\n"
        "a = p.parse_args()\n"
        "spec = json.loads(pathlib.Path(a.spec).read_text())\n"
        "out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'metrics.json').write_text(json.dumps({'hello': spec['params']['name']}))\n"
    )
    (ctx / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY testpkg /app/testpkg\nENV PYTHONPATH=/app\n"
    )
    subprocess.run(["docker", "build", "-q", "-t", image, str(ctx)], check=True, timeout=300)

    runner = DockerRunner(
        allowed_images=frozenset({image}),
        allowed_modules=frozenset({"testpkg.hello"}),
    )
    workdir = tmp_path / "work"
    workdir.mkdir()
    outdir = runner.run(
        {"module": "testpkg.hello", "task_id": "t1", "params": {"name": "flashml"}, "image": image},
        workdir,
        {},
    )
    assert json.loads((outdir / "metrics.json").read_text()) == {"hello": "flashml"}
