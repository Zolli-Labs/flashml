"""Checkpoint HTTP surface — written before the implementation (TDD).

Wires the (already-tested) CheckpointCatalog into the coordinator, scoped
per (job, task) so Mode A tasks each have their own checkpoint lineage.
The parts-first/manifest-last rule crosses the wire intact: registering
parts never creates a checkpoint; only a commit whose expected parts all
verify does.
"""

from __future__ import annotations

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


def _part(key, sha="a", size=100):
    return {"key": key, "sha256": sha * 64, "size_bytes": size}


def _register_and_commit(client, step, *, job="j1", task="t1", attempt="at1", parts=("w.json",)):
    for key in parts:
        r = client.post(
            f"/v1alpha1/jobs/{job}/tasks/{task}/checkpoints/parts",
            json={"attempt_id": attempt, "step": step, "part": _part(key)},
        )
        assert r.status_code == 200
    return client.post(
        f"/v1alpha1/jobs/{job}/tasks/{task}/checkpoints/commit",
        json={
            "attempt_id": attempt,
            "step": step,
            "expected_parts": [_part(k) for k in parts],
            "storage_prefix": f"artifact://jobs/{job}/{task}/ckpt/{step}/",
        },
    )


def test_commit_after_matching_parts_yields_hash_verified_manifest(client):
    r = _register_and_commit(client, step=100)
    assert r.status_code == 200, r.text
    manifest = r.json()
    assert manifest["step"] == 100
    assert manifest["validation"] == "hash_verified"


def test_commit_with_missing_part_is_409_and_no_manifest_exists(client):
    r = client.post(
        "/v1alpha1/jobs/j1/tasks/t1/checkpoints/commit",
        json={
            "attempt_id": "at1",
            "step": 100,
            "expected_parts": [_part("never-uploaded.json")],
            "storage_prefix": "artifact://jobs/j1/t1/ckpt/100/",
        },
    )
    assert r.status_code == 409
    assert client.get("/v1alpha1/jobs/j1/tasks/t1/checkpoints/latest").status_code == 404


def test_latest_returns_newest_step_scoped_per_task(client):
    _register_and_commit(client, step=100)
    _register_and_commit(client, step=200)
    _register_and_commit(client, step=999, task="t2")  # different task, own lineage
    latest = client.get("/v1alpha1/jobs/j1/tasks/t1/checkpoints/latest").json()
    assert latest["step"] == 200
    assert latest["parts"][0]["key"] == "w.json"


def test_lost_work_reports_steps_since_latest_valid(client):
    _register_and_commit(client, step=400)
    r = client.get("/v1alpha1/jobs/j1/tasks/t1/checkpoints/lost-work?failed_at_step=470")
    assert r.json() == {"lost_steps": 70, "latest_step": 400}
    none = client.get("/v1alpha1/jobs/j1/tasks/tX/checkpoints/lost-work?failed_at_step=470")
    assert none.json() == {"lost_steps": None, "latest_step": None}


def test_hyperparameter_search_expansion_carries_checkpoint_config(client):
    """A checkpointed training task rides the existing lease path: the
    trainer module is allowlisted and the job's checkpoint config reaches
    every task payload (which is what turns the executor's relay on)."""
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "train"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {
                    "type": "hyperparameter_search",
                    "parameters": {
                        "module": "flashml_workloads.sgd_trainer",
                        "trials": [{"steps": 100, "checkpoint_every": 20}],
                        "checkpoint": {"enabled": True},
                    },
                },
            },
        },
    )
    assert r.status_code == 201, r.text
    client.post(
        "/v1alpha1/nodes/register",
        json={"node_id": "n1", "kubernetes_node": "", "hostname": "n1",
              "capabilities": {}, "environment": "local"},
    )
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "n1"}).json()
    assert lease["payload"]["module"] == "flashml_workloads.sgd_trainer"
    assert lease["payload"]["checkpoint"] == {"enabled": True}
