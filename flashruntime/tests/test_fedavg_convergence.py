"""Federated averaging against a REAL coordinator over real HTTP.

The unit tests pin the arithmetic against a fake; this pins the whole loop:
real job expansion, real leases, real artifact storage, real commit-time
sha256 validation. The "agent" here is a few lines of urllib rather than
flashnode — flashruntime must not depend on flashnode (CLAUDE.md rule #1),
and the point being proven is the ROUND loop, not the sandbox.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from flashml_workloads import fedavg_worker
from flashml_workloads.fedavg_driver import HttpCoordinator, run_fedavg


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def coordinator(tmp_path: Path):
    """A real coordinator subprocess — same shape as e2e/conftest.py."""
    port = _free_port()
    env = {
        **os.environ,
        "FLASHML_ENABLE_KUBERAY": "0",
        "FLASHML_SERVICE_AUTOINIT": "1",
        "FLASHML_LEDGER_PATH": str(tmp_path / "ledger.db"),
        "FLASHML_LOCAL_ARTIFACTS_DIR": str(tmp_path / "artifacts"),
    }
    proc = subprocess.Popen(
        # `-u` because uvicorn buffers through a pipe and the process then
        # looks hung (HANDOFF.md §3). cwd is the repo root, never the
        # workspace root — from there `flashruntime/` resolves as a namespace
        # package and imports fail strangely.
        [sys.executable, "-u", "-m", "uvicorn", "flashruntime.service.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env, cwd=str(Path(__file__).parent.parent),
    )
    base = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(f"{base}/healthz", timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("coordinator did not become healthy in 20s")
                time.sleep(0.2)
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def _req(method: str, url: str, body=None, raw: bytes | None = None):
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type",
                       "application/octet-stream" if raw is not None else "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        payload = r.read()
        return r.status, (json.loads(payload) if payload else None)


def register_node(base_url: str, node_id: str) -> None:
    _req("POST", f"{base_url}/v1alpha1/nodes/register", {
        "schema_version": "v1alpha1",
        "node_id": node_id,
        "kubernetes_node": node_id,
        "hostname": node_id,
        "capabilities": {"cpu_cores": 2, "memory_bytes": 2 * 1024**3,
                         "gpus": [], "os": "linux", "architecture": "x86_64"},
        "sandbox_capable": False,
        "module_capable": True,
    })


def run_round_worker(base_url: str, node_id: str, workdir: Path) -> bool:
    """Claim one task, execute it, upload, commit. False if nothing to claim."""
    status, lease = _req("POST", f"{base_url}/v1alpha1/leases/claim",
                         {"node_id": node_id})
    if status == 204 or not lease:
        return False

    payload = lease["payload"]
    spec = {"params": payload["params"], "inputs": {}}

    for name, uri in (payload.get("inputs") or {}).items():
        key = uri.removeprefix("artifact://")
        _, blob = _req("GET", f"{base_url}/v1alpha1/artifacts/{key}")
        local = workdir / f"{name}.json"
        local.write_text(json.dumps(blob))
        spec["inputs"][name] = str(local)

    outdir = workdir / payload["task_id"]
    outdir.mkdir(parents=True, exist_ok=True)
    fedavg_worker.run_worker(spec, outdir)

    commit_sha = ""
    for f in sorted(outdir.iterdir()):
        raw = f.read_bytes()
        _req("PUT", f"{base_url}/v1alpha1/artifacts/{payload['output_prefix']}{f.name}",
             raw=raw)
        if f.name == "metrics.json":
            commit_sha = hashlib.sha256(raw).hexdigest()

    _req("POST", f"{base_url}/v1alpha1/attempts/{lease['lease_id']}/complete",
         {"output_sha256": commit_sha})
    return True


def drain(base_url: str, node_ids: list[str], workdir: Path, budget: int = 40) -> None:
    """Round-robin the nodes until the queue is empty."""
    for _ in range(budget):
        if not any(run_round_worker(base_url, n, workdir) for n in node_ids):
            return


WORKER_PARAMS = {"local_steps": 20, "lr": 0.1, "batch_size": 16, "seed": 0,
                 "in_dim": 8, "hidden": 16, "out_dim": 2, "dataset_size": 256}


def _initial_weights():
    model = fedavg_worker.build_model(
        WORKER_PARAMS["seed"], WORKER_PARAMS["in_dim"],
        WORKER_PARAMS["hidden"], WORKER_PARAMS["out_dim"])
    return fedavg_worker.state_to_blob(model)


def test_rounds_reduce_loss_across_two_nodes(coordinator, tmp_path):
    """The premise: two independent workers jointly improve one model."""
    import threading

    nodes = ["node-a", "node-b"]
    for n in nodes:
        register_node(coordinator, n)

    stop = threading.Event()

    def agent_loop():
        while not stop.is_set():
            if not run_round_worker(coordinator, "node-a", tmp_path / "a"):
                if not run_round_worker(coordinator, "node-b", tmp_path / "b"):
                    time.sleep(0.1)

    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    t = threading.Thread(target=agent_loop, daemon=True)
    t.start()
    try:
        result = run_fedavg(
            HttpCoordinator(coordinator), rounds=4, num_shards=2,
            min_participants=2, worker_params=WORKER_PARAMS,
            initial_weights=_initial_weights(),
            round_timeout_s=120.0, poll_seconds=0.25,
        )
    finally:
        stop.set()
        t.join(timeout=10)

    losses = [h["mean_loss"] for h in result["history"]]
    assert len(losses) == 4
    assert all(h["participants"] == 2 for h in result["history"])
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"


def test_round_completes_on_quorum_when_a_node_never_reports(coordinator, tmp_path):
    """The volunteer case: 3 shards dispatched, but the agent pool is capped
    to exactly MAX_CLAIMS=2 successful claims and then stops claiming — the
    third shard is never bound to any node and sits PENDING for the whole
    test. There is no shard-to-node binding here (either node can claim
    either shard), so what actually proves "one shard abandoned" is the
    claim cap, not the node count: without it, both node identities could
    sequentially serve all 3 shards before the driver's poll notices
    quorum, and `participants` would be 3.

    The round must aggregate on quorum (2 committed) rather than stall
    waiting for the abandoned third shard until the deadline.
    """
    nodes = ["node-a", "node-b"]
    for n in nodes:
        register_node(coordinator, n)
    (tmp_path / "w").mkdir()

    import threading

    stop = threading.Event()
    claims_done = 0
    claims_lock = threading.Lock()
    MAX_CLAIMS = 2  # < num_shards=3: shard #3 is deliberately left unclaimed

    def agent_loop():
        nonlocal claims_done
        while not stop.is_set():
            with claims_lock:
                if claims_done >= MAX_CLAIMS:
                    break
            if drain_once(coordinator, nodes, tmp_path / "w"):
                with claims_lock:
                    claims_done += 1
            else:
                time.sleep(0.1)

    def drain_once(base, ns, wd) -> bool:
        return any(run_round_worker(base, n, wd) for n in ns)

    t = threading.Thread(target=agent_loop, daemon=True)
    t.start()
    try:
        result = run_fedavg(
            HttpCoordinator(coordinator), rounds=1, num_shards=3,
            min_participants=2, worker_params=WORKER_PARAMS,
            initial_weights=_initial_weights(),
            round_timeout_s=120.0, poll_seconds=0.25,
        )
    finally:
        stop.set()
        t.join(timeout=10)

    # Exact, not >=: a regression that let the third (abandoned) shard get
    # served too — e.g. quorum aggregation silently waiting for it, or the
    # claim cap failing to hold it back — must fail this test, not pass it
    # by accident under `>=`.
    assert result["history"][0]["participants"] == 2
