"""Durable lease state — written before the implementation (TDD).

"Nodes are disposable; state is not" must include the coordinator itself:
tasks, leases, attempt counts, and lease history survive a process restart
via SqliteLeaseStore, and a lease issued before the restart remains
honorable after it.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient

from flashruntime.leases import LeaseManager
from flashruntime.leases.sqlite_store import SqliteLeaseStore
from flashruntime.protocol.v1alpha1 import TaskSpec, TaskState
from flashruntime.service.app import RuntimeSettings, create_app

T0 = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)


def _task(task_id="t1", job_id="j1"):
    return TaskSpec(task_id=task_id, job_id=job_id, commit_key=f"{job_id}/{task_id}", lease_seconds=60)


def test_store_rehydrates_full_task_state_across_instances(tmp_path):
    db = tmp_path / "leases.db"
    mgr = LeaseManager(store=SqliteLeaseStore(db))
    mgr.add_task(_task(), now=T0)
    lease = mgr.claim("node-a", now=T0)

    # a brand-new store + manager on the same file sees everything
    mgr2 = LeaseManager(store=SqliteLeaseStore(db))
    [record] = mgr2.records("j1")
    assert record.state == TaskState.LEASED
    assert record.attempts_used == 1
    assert record.active_lease.lease_id == lease.lease_id
    assert lease.lease_id in record.lease_history

    # the pre-restart lease is still honorable: heartbeat renews, commit wins
    renewed = mgr2.heartbeat(lease.lease_id, now=T0 + timedelta(seconds=30))
    assert renewed.deadline == T0 + timedelta(seconds=90)
    assert mgr2.complete(lease.lease_id, output_sha256="a" * 64, now=T0 + timedelta(seconds=40))

    # and the commit is durable too
    mgr3 = LeaseManager(store=SqliteLeaseStore(db))
    assert mgr3.records("j1")[0].state == TaskState.COMPLETED


def test_terminal_and_requeue_transitions_are_durable(tmp_path):
    db = tmp_path / "leases.db"
    mgr = LeaseManager(store=SqliteLeaseStore(db))
    mgr.add_task(_task(task_id="t1"), now=T0)
    lease = mgr.claim("node-a", now=T0)
    mgr.fail(lease.lease_id, "oom", now=T0)  # requeue

    mgr2 = LeaseManager(store=SqliteLeaseStore(db))
    assert mgr2.records("j1")[0].state == TaskState.PENDING
    assert mgr2.records("j1")[0].attempts_used == 1


def test_inflight_work_survives_coordinator_restart(tmp_path):
    """The end-to-end statement: submit → claim → coordinator restarts →
    the same worker re-registers, heartbeats its old lease, uploads, and
    commits — the job succeeds with attempt_number still 1."""
    settings = RuntimeSettings(
        enable_kuberay=False,
        ledger_path=str(tmp_path / "ledger.db"),
        artifacts_dir=str(tmp_path / "artifacts"),
    )
    node = {"node_id": "n1", "kubernetes_node": "", "hostname": "n1",
            "capabilities": {}, "environment": "local"}
    job = {
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": "restart-proof"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": "local/tier1", "tag": "dev"},
            "workload": {"type": "hyperparameter_search",
                         "parameters": {"trials": [{"C": 1.0}], "lease_seconds": 120}},
        },
    }

    with TestClient(create_app(settings)) as c1:
        c1.post("/v1alpha1/nodes/register", json=node)
        job_id = c1.post("/v1alpha1/jobs", json=job).json()["job_id"]
        lease = c1.post("/v1alpha1/leases/claim", json={"node_id": "n1"}).json()
        assert lease["attempt_number"] == 1

    # ---- coordinator restarts: a completely new app on the same state ----
    with TestClient(create_app(settings)) as c2:
        # node registry is volatile by design — the agent re-registers
        c2.post("/v1alpha1/nodes/register", json=node)
        # the pre-restart lease is still live and renewable
        assert c2.post(f"/v1alpha1/attempts/{lease['lease_id']}/heartbeat").status_code == 200
        key = f"jobs/{job_id}/{lease['task_id']}/metrics.json"
        record = c2.put(f"/v1alpha1/artifacts/{key}", content=b'{"acc": 1.0}').json()
        done = c2.post(
            f"/v1alpha1/attempts/{lease['lease_id']}/complete",
            json={"output_sha256": record["sha256"]},
        )
        assert done.json()["accepted"] is True
        assert c2.get(f"/v1alpha1/jobs/{job_id}").json()["state"] == "SUCCEEDED"
        tasks = c2.get(f"/v1alpha1/jobs/{job_id}/tasks").json()
        assert tasks[0]["attempts"] == 1  # no spurious retry from the restart