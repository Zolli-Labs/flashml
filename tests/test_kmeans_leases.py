"""Leased sharded K-means — tests written before the implementation (TDD).

The Mode A map/reduce pattern: broadcast centroids → each shard task
computes partial sums/counts (map) → the driver sums partials and divides
(reduce) → next iteration is a new job. One iteration = one lease job of N
independent shard tasks, so a dead worker costs one shard retry, never the
round.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from flashml_workloads.kmeans_driver import init_centroids, reduce_partials
from flashml_workloads.kmeans_shard import run_shard
from flashruntime.service.app import RuntimeSettings, create_app

# ---------------------------------------------------------------------------
# map: per-shard partial sums
# ---------------------------------------------------------------------------


def test_shard_partial_sums_match_hand_computation(tmp_path):
    """Points (0,0),(1,1) belong to centroid A=(0,0); (9,9) to B=(10,10).
    sums_A=(1,1) count_A=2; sums_B=(9,9) count_B=1;
    inertia = 0 + 2 + 2 = 4.0 (squared distances)."""
    shard = tmp_path / "shard.csv"
    shard.write_text("0,0\n1,1\n9,9\n")
    metrics = run_shard(
        {
            "task_id": "it00-shard-000",
            "params": {"centroids": [[0.0, 0.0], [10.0, 10.0]]},
            "inputs": {"shard": str(shard)},
        }
    )
    assert metrics["counts"] == [2, 1]
    assert metrics["sums"] == [[1.0, 1.0], [9.0, 9.0]]
    assert metrics["inertia"] == pytest.approx(0.0 + 2.0 + 2.0)
    assert metrics["n"] == 3


# ---------------------------------------------------------------------------
# reduce: combine partials into new centroids
# ---------------------------------------------------------------------------


def test_reduce_partials_computes_weighted_means():
    partials = [
        {"sums": [[1.0, 1.0], [9.0, 9.0]], "counts": [2, 1], "inertia": 4.0, "n": 3},
        {"sums": [[3.0, 3.0], [0.0, 0.0]], "counts": [2, 0], "inertia": 2.0, "n": 2},
    ]
    centroids, inertia = reduce_partials(partials, prev_centroids=[[0.0, 0.0], [10.0, 10.0]])
    assert centroids == [[1.0, 1.0], [9.0, 9.0]]  # (1+3)/4, (9+0)/1
    assert inertia == pytest.approx(6.0)


def test_reduce_keeps_previous_centroid_for_empty_cluster():
    partials = [{"sums": [[4.0], [0.0]], "counts": [2, 0], "inertia": 1.0, "n": 2}]
    centroids, _ = reduce_partials(partials, prev_centroids=[[0.0], [7.5]])
    assert centroids == [[2.0], [7.5]]  # empty cluster: centroid unchanged


def test_init_centroids_is_deterministic():
    rows = [[float(i), float(i)] for i in range(10)]
    assert init_centroids(rows, k=3) == init_centroids(rows, k=3)
    assert len(init_centroids(rows, k=3)) == 3


# ---------------------------------------------------------------------------
# expansion: one iteration = one lease job of shard tasks
# ---------------------------------------------------------------------------


@pytest.fixture()
def client(tmp_path):
    settings = RuntimeSettings(
        enable_kuberay=False,
        ledger_path=str(tmp_path / "ledger.db"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    with TestClient(create_app(settings)) as c:
        yield c


def test_sharded_kmeans_job_expands_one_task_per_shard(client):
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "kmeans-it2"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {
                    "type": "sharded_kmeans",
                    "parameters": {
                        "shards": ["artifact://data/s0.csv", "artifact://data/s1.csv"],
                        "centroids": [[0.0, 0.0], [1.0, 1.0]],
                        "iteration": 2,
                    },
                },
            },
        },
    )
    assert r.status_code == 201, r.text
    job_id = r.json()["job_id"]
    tasks = client.get(f"/v1alpha1/jobs/{job_id}/tasks").json()
    assert [t["task_id"] for t in tasks] == ["it02-shard-000", "it02-shard-001"]

    client.post(
        "/v1alpha1/nodes/register",
        json={"node_id": "n1", "kubernetes_node": "", "hostname": "n1",
              "capabilities": {}, "environment": "local"},
    )
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "n1"}).json()
    assert lease["payload"]["module"] == "flashml_workloads.kmeans_shard"
    assert lease["payload"]["params"]["centroids"] == [[0.0, 0.0], [1.0, 1.0]]
    assert lease["payload"]["inputs"]["shard"] == "artifact://data/s0.csv"


def test_sharded_kmeans_requires_shards_and_centroids(client):
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "bad-kmeans"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {"type": "sharded_kmeans", "parameters": {}},
            },
        },
    )
    assert r.status_code == 422
    assert "shards" in r.json()["detail"]
