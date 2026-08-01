#!/usr/bin/env python
"""Human-watchable federated-averaging convergence demo.

Runs the exact same loop as `tests/test_fedavg_convergence.py`, but as a
foreground script that prints one line per round instead of asserting in
silence. It brings up a real coordinator subprocess, registers two
"volunteer" nodes, and drives `run_fedavg` while a background thread plays
those nodes: claim a lease, train locally, upload the delta, commit.

Exit code is the check: a demo that prints numbers nobody verifies is not
evidence, so this script exits non-zero if the final round's loss is not
below the first round's.

Usage:
    PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -u scripts/fedavg_local_demo.py
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

ROUNDS = 4
NUM_SHARDS = 2
MIN_PARTICIPANTS = 2
WORKER_PARAMS = {
    "local_steps": 20, "lr": 0.1, "batch_size": 16, "seed": 0,
    "in_dim": 8, "hidden": 16, "out_dim": 2, "dataset_size": 256,
}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


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
    from flashml_workloads import fedavg_worker

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


def main() -> int:
    import torch  # noqa: F401 -- fail loud & early if torch is missing

    from flashml_workloads import fedavg_worker
    from flashml_workloads.fedavg_driver import HttpCoordinator, run_fedavg

    tmp = Path(tempfile.mkdtemp(prefix="fedavg-demo-"))
    port = _free_port()
    env = {
        **os.environ,
        "FLASHML_ENABLE_KUBERAY": "0",
        "FLASHML_SERVICE_AUTOINIT": "1",
        "FLASHML_LEDGER_PATH": str(tmp / "ledger.db"),
        "FLASHML_LOCAL_ARTIFACTS_DIR": str(tmp / "artifacts"),
    }
    print(f"starting coordinator on 127.0.0.1:{port} (workdir {tmp}) ...")
    proc = subprocess.Popen(
        [sys.executable, "-u", "-m", "uvicorn", "flashruntime.service.app:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        env=env, cwd=str(REPO_ROOT),
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    base_url = f"http://127.0.0.1:{port}"

    try:
        deadline = time.monotonic() + 20
        while True:
            try:
                with urllib.request.urlopen(f"{base_url}/healthz", timeout=1):
                    break
            except OSError:
                if time.monotonic() > deadline:
                    raise RuntimeError("coordinator did not become healthy in 20s")
                time.sleep(0.2)

        nodes = ["node-a", "node-b"]
        for n in nodes:
            register_node(base_url, n)
        (tmp / "a").mkdir()
        (tmp / "b").mkdir()
        print(f"registered {len(nodes)} volunteer nodes: {', '.join(nodes)}")

        stop = threading.Event()

        def agent_loop() -> None:
            while not stop.is_set():
                if not run_round_worker(base_url, "node-a", tmp / "a"):
                    if not run_round_worker(base_url, "node-b", tmp / "b"):
                        time.sleep(0.1)

        t = threading.Thread(target=agent_loop, daemon=True)
        t.start()

        def _initial_weights():
            model = fedavg_worker.build_model(
                WORKER_PARAMS["seed"], WORKER_PARAMS["in_dim"],
                WORKER_PARAMS["hidden"], WORKER_PARAMS["out_dim"])
            return fedavg_worker.state_to_blob(model)

        def on_round(result: dict) -> None:
            print(f"round {result['round']}  "
                  f"participants {result['participants']}/{NUM_SHARDS}  "
                  f"mean_loss {result['mean_loss']:.4f}")

        try:
            result = run_fedavg(
                HttpCoordinator(base_url), rounds=ROUNDS, num_shards=NUM_SHARDS,
                min_participants=MIN_PARTICIPANTS, worker_params=WORKER_PARAMS,
                initial_weights=_initial_weights(),
                round_timeout_s=120.0, poll_seconds=0.25,
                on_round=on_round,
            )
        finally:
            stop.set()
            t.join(timeout=10)
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()

    losses = [h["mean_loss"] for h in result["history"]]
    first, last = losses[0], losses[-1]
    print(f"converged: {first:.4f} -> {last:.4f} over {len(losses)} rounds")

    if not (last < first):
        print(f"FAIL: final loss {last:.4f} is not below first-round loss {first:.4f}",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
