"""Executor tests with a stubbed coordinator transport — no HTTP, no server.

`CoordinatorClient._request` is the single transport seam; the stub speaks
the coordinator's wire behavior (204 on empty queue, 410 on dead lease,
raw bytes for artifacts) so the loop, runner contract, and failure
reporting are tested exactly as they run against the real service.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from flashnode.executor import (
    CoordinatorClient,
    ExecutorLoop,
    LeaseLost,
    SubprocessRunner,
    TaskExecutionError,
)


class StubCoordinator:
    """In-memory coordinator double wired into CoordinatorClient._request."""

    def __init__(self):
        self.leases: list[dict] = []
        self.artifacts: dict[str, bytes] = {}
        self.completed: list[tuple[str, str]] = []
        self.failed: list[tuple[str, str]] = []
        self.dead_leases: set[str] = set()
        self.node_heartbeats = 0
        self.registered = True  # set False to simulate a restarted coordinator
        self.register_calls = 0

    def make_lease(self, task_id="t1", payload=None):
        lease = {
            "lease_id": f"ls-{task_id}",
            "task_id": task_id,
            "job_id": "j1",
            "node_id": "n1",
            "attempt_number": 1,
            "deadline": (datetime.now(timezone.utc) + timedelta(seconds=60)).isoformat(),
            "payload": payload or {},
        }
        self.leases.append(lease)
        return lease

    def request(self, method, path, body=None, content_type=None):
        if path == "/v1alpha1/leases/claim":
            if not self.registered:
                return 403, b'{"detail": "unregistered node"}'
            if not self.leases:
                return 204, b""
            return 200, json.dumps(self.leases.pop(0)).encode()
        if path.endswith("/heartbeat") and "/attempts/" in path:
            lease_id = path.split("/")[3]
            if lease_id in self.dead_leases:
                return 410, b'{"detail": "gone"}'
            return 200, b"{}"
        if path.endswith("/heartbeat"):
            if not self.registered:
                return 404, b'{"detail": "unknown node"}'
            self.node_heartbeats += 1
            return 200, b"{}"
        if path.endswith("/complete"):
            self.completed.append((path.split("/")[3], json.loads(body)["output_sha256"]))
            return 200, b'{"accepted": true}'
        if path.endswith("/fail"):
            self.failed.append((path.split("/")[3], json.loads(body)["reason"]))
            return 200, b"{}"
        if path.startswith("/v1alpha1/artifacts/"):
            key = path.removeprefix("/v1alpha1/artifacts/")
            if method == "PUT":
                self.artifacts[key] = body
                return 200, b"{}"
            if key in self.artifacts:
                return 200, self.artifacts[key]
            return 404, b'{"detail": "missing"}'
        if path == "/v1alpha1/nodes/register":
            self.registered = True
            self.register_calls += 1
            return 200, b'{"status": "registered"}'
        raise AssertionError(f"unexpected request {method} {path}")


@pytest.fixture()
def stub(monkeypatch):
    stub = StubCoordinator()
    client = CoordinatorClient("http://stub")
    monkeypatch.setattr(
        client, "_request",
        lambda m, p, body=None, content_type="", headers=None: stub.request(m, p, body),
    )
    return stub, client


ENV_SPY_MODULE = """
import argparse, json, os, pathlib
p = argparse.ArgumentParser(); p.add_argument("--spec"); p.add_argument("--out")
a = p.parse_args()
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
(out / "metrics.json").write_text(json.dumps(sorted(os.environ.keys())))
"""


def test_subprocess_runner_scrubs_agent_secrets_from_task_env(tmp_path, monkeypatch):
    """Task code must never see the agent's environment (join codes,
    coordinator URLs) — only a minimal whitelist survives into the child."""
    pkg = tmp_path / "spy_workloads"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "envspy.py").write_text(ENV_SPY_MODULE)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    monkeypatch.setenv("FLASHNODE_JOIN_CODE", "SUPER-SECRET")
    monkeypatch.setenv("FLASHNODE_COORDINATOR_URL", "http://internal:8100")

    runner = SubprocessRunner(allowed_modules=frozenset({"spy_workloads.envspy"}))
    outdir = runner.run({"module": "spy_workloads.envspy", "task_id": "t1"}, tmp_path, {})
    child_env = json.loads((outdir / "metrics.json").read_text())
    assert "FLASHNODE_JOIN_CODE" not in child_env
    assert "FLASHNODE_COORDINATOR_URL" not in child_env
    assert "PYTHONPATH" in child_env  # the whitelist still works


def test_loop_reregisters_after_coordinator_restart(stub, fake_module):
    """A restarted coordinator forgets the node registry; the agent must
    notice the 404 heartbeat and re-register instead of starving forever."""
    from flashruntime.protocol.v1alpha1 import NodeCapabilities, NodeRegistration

    coordinator, client = stub
    coordinator.registered = False  # simulate post-restart amnesia
    coordinator.make_lease("t1", payload={"module": fake_module, "task_id": "t1", "params": {}})
    reg = NodeRegistration(node_id="n1", kubernetes_node="", hostname="n1",
                           capabilities=NodeCapabilities(), environment="local")
    loop = ExecutorLoop(
        client, "n1",
        runner=SubprocessRunner(allowed_modules=frozenset({fake_module})),
        registration=reg,
        node_heartbeat_seconds=0.0,  # heartbeat every cycle
    )
    accepted = loop.run(max_tasks=1, idle_exit=True)
    assert accepted == 1
    assert coordinator.register_calls == 1  # re-registered exactly once


# A tiny allowlisted task module for runner tests, created on the fly.
TASK_MODULE = """
import argparse, json, pathlib
p = argparse.ArgumentParser(); p.add_argument("--spec"); p.add_argument("--out")
a = p.parse_args()
spec = json.loads(pathlib.Path(a.spec).read_text())
out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
data = pathlib.Path(spec["inputs"]["dataset"]).read_text() if spec.get("inputs") else ""
(out / "metrics.json").write_text(json.dumps(
    {"task_id": spec["task_id"], "n_chars": len(data), "params": spec["params"]}))
"""


@pytest.fixture()
def fake_module(tmp_path, monkeypatch):
    pkg = tmp_path / "fake_workloads"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("")
    (pkg / "echo_trial.py").write_text(TASK_MODULE)
    # PYTHONPATH, not syspath_prepend: the runner executes the module in a
    # *subprocess*, which only sees the environment.
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))
    return "fake_workloads.echo_trial"


def test_subprocess_runner_refuses_argv_payloads(tmp_path):
    """Tier 1 is not a security boundary. An argv payload reaching it would
    be arbitrary code execution on the host with no isolation at all."""
    with pytest.raises(TaskExecutionError, match="argv"):
        SubprocessRunner().run({"argv": ["python", "evil.py"]}, tmp_path, {})


def test_runner_refuses_unlisted_module(tmp_path):
    runner = SubprocessRunner()
    with pytest.raises(TaskExecutionError, match="not allowlisted"):
        runner.run({"module": "os.system"}, tmp_path, {})


def test_runner_executes_allowlisted_module(tmp_path, fake_module):
    runner = SubprocessRunner(allowed_modules=frozenset({fake_module}))
    dataset = tmp_path / "data.csv"
    dataset.write_text("a,b,c\n")
    outdir = runner.run(
        {"module": fake_module, "task_id": "t1", "params": {"x": 1}},
        tmp_path,
        {"dataset": dataset},
    )
    metrics = json.loads((outdir / "metrics.json").read_text())
    assert metrics == {"task_id": "t1", "n_chars": 6, "params": {"x": 1}}


def test_loop_executes_downloads_uploads_commits(stub, fake_module):
    coordinator, client = stub
    coordinator.artifacts["jobs/j1/input/data.csv"] = b"1,2,3\n4,5,6\n"
    coordinator.make_lease(
        "t1",
        payload={
            "module": fake_module,
            "task_id": "t1",
            "params": {},
            "inputs": {"dataset": "artifact://jobs/j1/input/data.csv"},
            "output_prefix": "jobs/j1/t1/",
        },
    )
    loop = ExecutorLoop(
        client, "n1", runner=SubprocessRunner(allowed_modules=frozenset({fake_module}))
    )
    accepted = loop.run(max_tasks=1, idle_exit=True)
    assert accepted == 1
    assert coordinator.completed and coordinator.completed[0][0] == "ls-t1"
    uploaded = json.loads(coordinator.artifacts["jobs/j1/t1/metrics.json"])
    assert uploaded["n_chars"] == len(b"1,2,3\n4,5,6\n")


def test_loop_reports_failure_and_moves_on(stub, fake_module):
    coordinator, client = stub
    coordinator.make_lease("bad", payload={"module": "not.allowlisted"})
    coordinator.make_lease(
        "good", payload={"module": fake_module, "task_id": "good", "params": {}}
    )
    loop = ExecutorLoop(
        client, "n1", runner=SubprocessRunner(allowed_modules=frozenset({fake_module}))
    )
    accepted = loop.run(max_tasks=1, idle_exit=True)
    assert accepted == 1
    assert coordinator.failed[0][0] == "ls-bad"
    assert "not allowlisted" in coordinator.failed[0][1]


def test_client_raises_lease_lost_on_410(stub):
    coordinator, client = stub
    coordinator.dead_leases.add("ls-dead")
    with pytest.raises(LeaseLost):
        client.attempt_heartbeat("ls-dead")


def test_client_claim_returns_none_on_empty_queue(stub):
    _, client = stub
    assert client.claim("n1") is None


def test_client_sends_join_code_header_on_register(monkeypatch):
    """A coordinator with FLASHML_JOIN_CODE set refuses registration without
    the X-FlashML-Join-Code header — the client must send it when given one."""
    from flashruntime.protocol.v1alpha1 import NodeCapabilities, NodeRegistration

    seen: dict = {}

    def capture(method, path, body=None, content_type="application/json", headers=None):
        seen.update({"path": path, "headers": headers or {}})
        return 200, b'{"status": "registered"}'

    client = CoordinatorClient("http://stub", join_code="LOCAL-2026")
    monkeypatch.setattr(client, "_request", capture)
    client.register(
        NodeRegistration(node_id="n1", kubernetes_node="", hostname="n1",
                         capabilities=NodeCapabilities(), environment="local")
    )
    assert seen["path"] == "/v1alpha1/nodes/register"
    assert seen["headers"].get("X-FlashML-Join-Code") == "LOCAL-2026"


def test_work_cli_refuses_docker_runner_without_image_allowlist(monkeypatch, capsys):
    """`flashnode work --runner docker` with no FLASHNODE_ALLOWED_IMAGES
    must exit with an error before touching the network — an empty image
    allowlist would make every task fail closed anyway."""
    from flashnode.agent import cli

    monkeypatch.delenv("FLASHNODE_ALLOWED_IMAGES", raising=False)
    rc = cli.main(["work", "--runner", "docker", "--coordinator", "http://localhost:1"])
    assert rc == 2
    assert "FLASHNODE_ALLOWED_IMAGES" in capsys.readouterr().err


@pytest.mark.parametrize("runner", ["docker", "argv"])
def test_work_cli_refuses_sandboxed_runner_without_docker_binary(monkeypatch, capsys, runner):
    """F6: subprocess.run(["docker", ...]) raises FileNotFoundError when
    docker isn't installed; execute_one only catches TaskExecutionError/
    LeaseLost, so that traceback would kill the whole agent on its first
    task. Refuse to start at all, same failure mode as the allowlist check."""
    from flashnode.agent import cli

    monkeypatch.setenv("FLASHNODE_ALLOWED_IMAGES", "img:1")
    monkeypatch.setattr("shutil.which", lambda name: None)
    rc = cli.main(["work", "--runner", runner, "--coordinator", "http://localhost:1"])
    assert rc == 2
    assert "docker" in capsys.readouterr().err.lower()


def test_loop_uploads_nested_output_and_still_commits_metrics_hash(stub):
    """F7: ArgvDockerRunner size-caps with rglob (recursive) but the loop
    used to upload with iterdir (top level only) — a job writing
    out/checkpoints/model.pt passed the cap, uploaded nothing for it, and
    still committed successfully. Nested files must upload under their
    relative path, and metrics.json's hash — the commit key — must still be
    exactly what's sent to complete()."""
    coordinator, client = stub

    class NestedOutputRunner:
        def run(self, payload, workdir, inputs):
            outdir = workdir / "out"
            (outdir / "checkpoints").mkdir(parents=True)
            (outdir / "checkpoints" / "model.pt").write_bytes(b"weights")
            (outdir / "metrics.json").write_text('{"acc": 1.0}')
            return outdir

    coordinator.make_lease(
        "t1",
        payload={"task_id": "t1", "params": {}, "output_prefix": "jobs/j1/t1/"},
    )
    loop = ExecutorLoop(client, "n1", runner=NestedOutputRunner())
    accepted = loop.run(max_tasks=1, idle_exit=True)

    assert accepted == 1
    # nested file uploaded under its relative path
    assert coordinator.artifacts["jobs/j1/t1/checkpoints/model.pt"] == b"weights"
    # top-level metrics.json also uploaded
    assert coordinator.artifacts["jobs/j1/t1/metrics.json"] == b'{"acc": 1.0}'
    # the hash committed is metrics.json's hash, not some other file's
    import hashlib

    expected_sha = hashlib.sha256(b'{"acc": 1.0}').hexdigest()
    assert coordinator.completed[0] == ("ls-t1", expected_sha)


def test_loop_honors_workdir_base(stub, fake_module, tmp_path):
    """Docker VMs on macOS only share $HOME — task workdirs must be
    placeable somewhere the VM can see (FLASHNODE_WORKDIR)."""
    coordinator, client = stub
    coordinator.make_lease("t1", payload={"module": fake_module, "task_id": "t1", "params": {}})
    base = tmp_path / "visible-to-vm"
    base.mkdir()
    seen: list[str] = []

    class RecordingRunner(SubprocessRunner):
        def run(self, payload, workdir, inputs):
            seen.append(str(workdir))
            return super().run(payload, workdir, inputs)

    loop = ExecutorLoop(
        client, "n1",
        runner=RecordingRunner(allowed_modules=frozenset({fake_module})),
        workdir_base=base,
    )
    assert loop.run(max_tasks=1, idle_exit=True) == 1
    assert seen and seen[0].startswith(str(base))
