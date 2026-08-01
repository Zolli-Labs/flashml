"""Which (job, task) pairs may a node write to right now?

Time is controlled by passing `now=` to the manager — LeaseManager takes no
clock; every time-sensitive method accepts a `now: datetime | None`.
"""

from datetime import datetime, timedelta, timezone

import pytest

from flashruntime.leases.manager import LeaseManager
from flashruntime.leases.store import InMemoryLeaseStore
from flashruntime.protocol.v1alpha1 import TaskSpec

T0 = datetime(2026, 7, 31, 12, 0, 0, tzinfo=timezone.utc)


def _task(job_id: str, task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id, job_id=job_id,
        commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
        max_attempts=4, lease_seconds=30.0, payload={},
    )


@pytest.fixture()
def manager():
    return LeaseManager(store=InMemoryLeaseStore())


def test_no_leases_means_no_scope(manager):
    assert manager.live_leases_for_node("node-a", now=T0) == set()


def test_a_claimed_task_is_in_scope_for_its_holder(manager):
    manager.add_task(_task("job-1", "task-000"), now=T0)
    manager.claim("node-a", now=T0)
    assert manager.live_leases_for_node("node-a", now=T0) == {("job-1", "task-000")}


def test_another_node_gets_no_scope_from_it(manager):
    manager.add_task(_task("job-1", "task-000"), now=T0)
    manager.claim("node-a", now=T0)
    assert manager.live_leases_for_node("node-b", now=T0) == set()


def test_multiple_live_leases_all_appear(manager):
    manager.add_task(_task("job-1", "task-000"), now=T0)
    manager.add_task(_task("job-2", "task-000"), now=T0)
    manager.claim("node-a", now=T0)
    manager.claim("node-a", now=T0)
    assert manager.live_leases_for_node("node-a", now=T0) == {
        ("job-1", "task-000"), ("job-2", "task-000"),
    }


def test_an_expired_lease_leaves_scope(manager):
    """A straggler whose lease lapsed must not be able to overwrite the
    result of whoever reclaimed its task."""
    manager.add_task(_task("job-1", "task-000"), now=T0)
    manager.claim("node-a", now=T0)
    assert manager.live_leases_for_node("node-a", now=T0) == {("job-1", "task-000")}
    later = T0 + timedelta(seconds=31)
    assert manager.live_leases_for_node("node-a", now=later) == set()


def test_a_completed_task_leaves_scope(manager):
    """A task whose result was already accepted no longer appears in the
    node's write scope — otherwise a second upload could replace a
    committed artifact.

    Note: `complete()` clears `record.active_lease` and moves the record off
    LEASED, so this record is simply absent from `_store.leased()` and the
    scan never reaches it. This does NOT exercise `_is_live`'s record-state
    check specifically — it is a regression guard that `live_leases_for_node`
    sources scope from `record.active_lease` (the live source of truth),
    not from `lease_history` or `_store.all()`, either of which would still
    surface this lease after completion."""
    manager.add_task(_task("job-1", "task-000"), now=T0)
    lease = manager.claim("node-a", now=T0)
    manager.complete(lease.lease_id, output_sha256="0" * 64, now=T0)
    assert manager.live_leases_for_node("node-a", now=T0) == set()
