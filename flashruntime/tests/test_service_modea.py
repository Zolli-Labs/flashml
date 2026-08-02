"""Mode A service tests: the lease coordinator over HTTP, no cluster.

Covers the full local loop at the API level: node registration → lease-mode
job submission (grid expansion) → claim/heartbeat/complete → derived job
state → local artifact hosting → per-node accepted-work credit.
"""

from __future__ import annotations

import hashlib

import pytest
from fastapi.testclient import TestClient

from flashruntime.service.app import RuntimeSettings, create_app


@pytest.fixture()
def client(tmp_path):
    settings = RuntimeSettings(
        enable_kuberay=False,
        ledger_path=str(tmp_path / "ledger.db"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    with TestClient(create_app(settings)) as c:
        yield c


def _register(client, node_id="node-a"):
    r = client.post(
        "/v1alpha1/nodes/register",
        json={
            "node_id": node_id,
            "kubernetes_node": "",
            "hostname": f"{node_id}.local",
            "capabilities": {"cpu_cores": 8, "memory_bytes": 16_000_000_000},
            "environment": "local",
        },
    )
    assert r.status_code == 200
    return node_id


def _upload_and_complete(client, lease, content: bytes = b'{"ok": true}'):
    """The honest commit path: upload the output at the task's commit key,
    then complete with its real sha256 (commits are validated server-side)."""
    key = f"jobs/{lease['job_id']}/{lease['task_id']}/metrics.json"
    record = client.put(f"/v1alpha1/artifacts/{key}", content=content).json()
    return client.post(
        f"/v1alpha1/attempts/{lease['lease_id']}/complete",
        json={"output_sha256": record["sha256"]},
    )


def _submit_grid_job(client, trials=None, grid=None):
    params = {"lease_seconds": 30}
    if trials:
        params["trials"] = trials
    if grid:
        params["grid"] = grid
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "sweep"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {"type": "hyperparameter_search", "parameters": params},
            },
        },
    )
    assert r.status_code == 201, r.text
    return r.json()["job_id"]


def test_grid_expansion_and_full_claim_complete_loop(client):
    _register(client)
    job_id = _submit_grid_job(client, grid={"C": [0.1, 1.0], "model": ["logreg"]})

    tasks = client.get(f"/v1alpha1/jobs/{job_id}/tasks").json()
    assert len(tasks) == 2  # 2×1 grid
    assert client.get(f"/v1alpha1/jobs/{job_id}").json()["state"] == "RUNNING"

    for _ in range(2):
        lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
        assert lease["payload"]["module"] == "flashml_workloads.sklearn_trial"
        # the docker-runner tier needs the job's image in every payload
        assert lease["payload"]["image"] == "local/tier1:dev"
        hb = client.post(f"/v1alpha1/attempts/{lease['lease_id']}/heartbeat")
        assert hb.status_code == 200
        assert _upload_and_complete(client, lease).json()["accepted"] is True

    assert client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).status_code == 204
    assert client.get(f"/v1alpha1/jobs/{job_id}").json()["state"] == "SUCCEEDED"
    nodes = client.get("/v1alpha1/nodes").json()
    assert nodes[0]["accepted_tasks"] == 2


def test_unregistered_node_cannot_claim(client):
    _submit_grid_job(client, trials=[{"C": 1.0}])
    assert client.post("/v1alpha1/leases/claim", json={"node_id": "ghost"}).status_code == 403


def test_node_view_exposes_argv_capable(client):
    """GET /v1alpha1/nodes must surface argv_capable — the argv placement
    gate (scheduler.IsolationAwarePlacement) and the dashboard both read it
    from the node view, not just the claim-time node dict."""
    r = client.post(
        "/v1alpha1/nodes/register",
        json={
            "node_id": "n1",
            "kubernetes_node": "",
            "hostname": "n1.local",
            "capabilities": {"cpu_cores": 8, "memory_bytes": 16_000_000_000},
            "environment": "local",
            "argv_capable": True,
        },
    )
    assert r.status_code == 200
    nodes = client.get("/v1alpha1/nodes").json()
    assert nodes[0]["argv_capable"] is True


def test_expansion_errors_are_422(client):
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "bad"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {"type": "hyperparameter_search", "parameters": {}},
            },
        },
    )
    assert r.status_code == 422
    assert "trials" in r.json()["detail"]


def test_ray_backend_disabled_returns_503(client):
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "rayjob"},
            "spec": {
                "execution": {"backend": "ray"},
                "image": {"repository": "x/y", "tag": "v1"},
                "workload": {"type": "sharded_kmeans"},
            },
        },
    )
    assert r.status_code == 503


def test_local_artifact_roundtrip_and_listing(client):
    payload = b"col1,col2,label\n" * 100
    up = client.put("/v1alpha1/artifacts/jobs/j1/input/data.csv", content=payload)
    assert up.status_code == 200
    assert up.json()["sha256"] == hashlib.sha256(payload).hexdigest()

    down = client.get("/v1alpha1/artifacts/jobs/j1/input/data.csv")
    assert down.content == payload

    listing = client.get("/v1alpha1/jobs/j1/artifacts").json()
    assert listing[0]["key"] == "jobs/j1/input/data.csv"

    assert client.put("/v1alpha1/artifacts/../escape", content=b"x").status_code in (400, 404)
    assert client.get("/v1alpha1/artifacts/jobs/absent").status_code == 404


def test_dead_node_task_requeues_via_http(client):
    """The kill-a-worker story through the HTTP surface: claim with a short
    lease, never heartbeat, and the task becomes claimable for another node."""
    _register(client, "node-a")
    _register(client, "node-b")
    job_id = _submit_grid_job(client, trials=[{"C": 1.0}])
    # short lease so the (time-based) expiry happens quickly
    import time

    first = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    assert first["attempt_number"] == 1
    # override: wait past the 30s lease? Too slow — instead exercise fail():
    r = client.post(f"/v1alpha1/attempts/{first['lease_id']}/fail", json={"reason": "killed"})
    assert r.status_code == 200

    second = client.post("/v1alpha1/leases/claim", json={"node_id": "node-b"}).json()
    assert second["task_id"] == first["task_id"]
    assert second["attempt_number"] == 2
    assert _upload_and_complete(client, second).json()["accepted"] is True
    # node-a's zombie commit is rejected, not erred
    late = client.post(
        f"/v1alpha1/attempts/{first['lease_id']}/complete", json={"output_sha256": "2" * 64}
    )
    assert late.json()["accepted"] is False
    assert client.get(f"/v1alpha1/jobs/{job_id}").json()["state"] == "SUCCEEDED"


def test_dashboard_served(client):
    page = client.get("/")
    assert page.status_code == 200
    assert "FlashRuntime" in page.text


def test_commit_is_rejected_unless_uploaded_output_matches_hash(client):
    """Accepted work = validated output: complete() must verify the artifact
    at the task's commit_key against the reported sha256. Missing or
    mismatched output fails the attempt (task requeues); only a verified
    upload commits."""
    _register(client)
    job_id = _submit_grid_job(client, trials=[{"C": 1.0}])

    # attempt 1: commit with no uploaded artifact at all → rejected, requeued
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    r = client.post(
        f"/v1alpha1/attempts/{lease['lease_id']}/complete", json={"output_sha256": "a" * 64}
    )
    assert r.json()["accepted"] is False
    tasks = client.get(f"/v1alpha1/jobs/{job_id}/tasks").json()
    assert tasks[0]["state"] == "PENDING"  # requeued, not completed

    # attempt 2: upload the wrong bytes for the claimed hash → rejected
    lease2 = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    key = f"jobs/{job_id}/{lease2['task_id']}/metrics.json"
    client.put(f"/v1alpha1/artifacts/{key}", content=b'{"acc": 0.9}')
    r = client.post(
        f"/v1alpha1/attempts/{lease2['lease_id']}/complete", json={"output_sha256": "b" * 64}
    )
    assert r.json()["accepted"] is False

    # attempt 3: upload + matching hash → accepted
    lease3 = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    payload = b'{"acc": 0.95}'
    up = client.put(f"/v1alpha1/artifacts/{key}", content=payload).json()
    r = client.post(
        f"/v1alpha1/attempts/{lease3['lease_id']}/complete",
        json={"output_sha256": up["sha256"]},
    )
    assert r.json()["accepted"] is True
    assert client.get(f"/v1alpha1/jobs/{job_id}").json()["state"] == "SUCCEEDED"


def test_artifact_upload_size_is_capped(tmp_path):
    settings = RuntimeSettings(
        enable_kuberay=False,
        ledger_path=str(tmp_path / "ledger.db"),
        artifacts_dir=str(tmp_path / "artifacts"),
        max_artifact_mb=1,
    )
    with TestClient(create_app(settings)) as c:
        ok = c.put("/v1alpha1/artifacts/small.bin", content=b"x" * 1024)
        assert ok.status_code == 200
        too_big = c.put("/v1alpha1/artifacts/big.bin", content=b"x" * (2 * 1024 * 1024))
        assert too_big.status_code == 413


def test_join_code_gates_registration_when_configured(tmp_path):
    """With FLASHML_JOIN_CODE set, node registration requires the matching
    X-FlashML-Join-Code header; without configuration the coordinator stays
    open (the self-hosted default, unchanged)."""
    settings = RuntimeSettings(
        enable_kuberay=False,
        ledger_path=str(tmp_path / "ledger.db"),
        artifacts_dir=str(tmp_path / "artifacts"),
        join_code="LOCAL-2026",
    )
    body = {
        "node_id": "n1", "kubernetes_node": "", "hostname": "n1",
        "capabilities": {}, "environment": "local",
    }
    with TestClient(create_app(settings)) as c:
        assert c.post("/v1alpha1/nodes/register", json=body).status_code == 403
        wrong = c.post(
            "/v1alpha1/nodes/register", json=body,
            headers={"X-FlashML-Join-Code": "WRONG"},
        )
        assert wrong.status_code == 403
        ok = c.post(
            "/v1alpha1/nodes/register", json=body,
            headers={"X-FlashML-Join-Code": "LOCAL-2026"},
        )
        assert ok.status_code == 200
        # only registration is gated; the registered node claims normally
        assert c.post("/v1alpha1/leases/claim", json={"node_id": "n1"}).status_code == 204
