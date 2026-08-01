"""A node may write only under the tasks it currently holds."""

import pytest
from fastapi.testclient import TestClient

from flashruntime.service.app import create_app, RuntimeSettings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHML_NODE_TOKENS", "node-a:tok-a,node-b:tok-b")
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_SERVICE_AUTOINIT", "1")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "ledger.db"))
    return TestClient(create_app())


def _register(client, node_id):
    client.post("/v1alpha1/nodes/register", json={
        "schema_version": "v1alpha1", "node_id": node_id,
        "kubernetes_node": node_id, "hostname": node_id,
        "capabilities": {"cpu_cores": 1, "memory_bytes": 1 << 30,
                         "gpus": [], "os": "linux", "architecture": "x86_64"},
    })


def _submit_one_task_job(client):
    r = client.post("/v1alpha1/jobs", json={
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": "scope"},
        "spec": {"execution": {"backend": "leases"},
                 "image": {"repository": "local/tier1", "tag": "dev"},
                 "workload": {"type": "hyperparameter_search",
                              "parameters": {"trials": [{"C": 1.0}]}}},
    })
    return r.json()["job_id"]


def test_write_without_a_token_is_401(client):
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json", content=b"{}")
    assert r.status_code == 401


def test_write_with_a_bad_token_is_401(client):
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_write_outside_any_live_lease_is_403(client):
    _register(client, "node-a")
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_the_lease_holder_may_write_under_its_own_task(client):
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 200


def test_another_node_may_not_write_under_that_task(client):
    """The core exploit: one volunteer overwriting another's committed result."""
    _register(client, "node-a")
    _register(client, "node-b")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-b"})
    assert r.status_code == 403


def test_the_holder_may_not_write_outside_its_task_prefix(client):
    """Guards the federated-averaging model key, which lives at
    jobs/{job}/round-NNN/weights.json — outside any task prefix."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"})
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/round-000/weights.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_a_sibling_prefix_does_not_satisfy_the_check(client):
    """`jobs/j/trial-000extra/` must not pass because it starts with
    `jobs/j/trial-000`. The prefix must end at a separator."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}extra/metrics.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_checkpoint_part_outside_a_live_lease_is_403(client):
    # Body shape matches RegisterPartRequest (attempt_id + nested part) —
    # the brief's flat payload doesn't match the pre-existing checkpoint
    # schema and would 422 before authorization ever runs.
    _register(client, "node-a")
    r = client.post(
        "/v1alpha1/jobs/other-job/tasks/trial-000/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": "jobs/other-job/trial-000/ckpt/step-10.json",
                       "sha256": "0" * 64, "size_bytes": 2}},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert r.status_code == 403


def test_checkpoint_part_without_a_token_is_401(client):
    r = client.post(
        "/v1alpha1/jobs/other-job/tasks/trial-000/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": "k", "sha256": "0" * 64, "size_bytes": 2}},
    )
    assert r.status_code == 401


def test_the_lease_holder_may_register_a_checkpoint_part(client):
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    key = f"jobs/{job_id}/{task_id}/ckpt/step-10.json"
    client.put(f"/v1alpha1/artifacts/{key}", content=b"{}",
               headers={"Authorization": "Bearer tok-a"})
    r = client.post(
        f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": key, "sha256": "0" * 64, "size_bytes": 2}},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert r.status_code in (200, 201)


def test_reads_are_not_scoped(client):
    """Drivers read other tasks' outputs and agents download shared inputs."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
               content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert client.get(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json").status_code == 200
