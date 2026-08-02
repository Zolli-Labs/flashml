"""Lease manager tests — the Stage-0 story as executable spec.

The core scenario every FlashML demo is built on: a worker claims a task,
dies, its lease expires, another worker completes the task, and the dead
worker's late commit is rejected. Time is injected, so the tests control the
clock exactly — no sleeps, no flakes.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from flashruntime.leases import LeaseError, LeaseManager
from flashruntime.leases.store import InMemoryLeaseStore, TaskRecord
from flashruntime.protocol.v1alpha1 import EventType, TaskSpec, TaskState

T0 = datetime(2026, 7, 19, 12, 0, 0, tzinfo=timezone.utc)


def _mgr(events=None):
    return LeaseManager(on_event=events.append if events is not None else None)


def _task(task_id="t1", job_id="j1", lease_seconds=60.0, max_attempts=3):
    return TaskSpec(
        task_id=task_id,
        job_id=job_id,
        commit_key=f"{job_id}/{task_id}/output",
        lease_seconds=lease_seconds,
        max_attempts=max_attempts,
    )


def _task_spec(job_id: str, task_id: str) -> TaskSpec:
    """Minimal spec for the (job_id, task_id) collision tests below."""
    return TaskSpec(task_id=task_id, job_id=job_id, commit_key=f"{job_id}/{task_id}/m.json")


def test_claim_heartbeat_complete_happy_path():
    events = []
    mgr = _mgr(events)
    mgr.add_task(_task(), now=T0)

    lease = mgr.claim("node-a", now=T0)
    assert lease is not None and lease.attempt_number == 1
    assert lease.deadline == T0 + timedelta(seconds=60)

    renewed = mgr.heartbeat(lease.lease_id, now=T0 + timedelta(seconds=30))
    assert renewed.deadline == T0 + timedelta(seconds=90)

    assert mgr.complete(lease.lease_id, output_sha256="a" * 64, now=T0 + timedelta(seconds=40))
    assert mgr.job_state("j1") == {TaskState.COMPLETED.value: 1}
    assert [e.type for e in events] == [
        EventType.TASK_CREATED,
        EventType.LEASE_CLAIMED,
        EventType.LEASE_RENEWED,
        EventType.TASK_COMMIT_ACCEPTED,
    ]


def test_dead_worker_lease_expires_and_task_requeues_to_another_node():
    """The kill-a-worker demo, in library form."""
    events = []
    mgr = _mgr(events)
    mgr.add_task(_task(), now=T0)

    dead = mgr.claim("node-a", now=T0)
    # node-a goes silent; 61 s later node-b asks for work: the sweep runs
    # inside claim, expires node-a's lease, and hands the task to node-b.
    t1 = T0 + timedelta(seconds=61)
    second = mgr.claim("node-b", now=t1)
    assert second is not None and second.task_id == dead.task_id
    assert second.attempt_number == 2
    assert EventType.LEASE_EXPIRED in [e.type for e in events]
    assert EventType.TASK_REQUEUED in [e.type for e in events]

    # node-b finishes; then node-a wakes up and tries its late commit.
    assert mgr.complete(second.lease_id, output_sha256="b" * 64, now=t1 + timedelta(seconds=10))
    assert not mgr.complete(dead.lease_id, output_sha256="a" * 64, now=t1 + timedelta(seconds=20))
    rejected = [e for e in events if e.type == EventType.TASK_COMMIT_REJECTED]
    assert rejected and "late/duplicate" in rejected[0].message
    # exactly-once: still exactly one COMPLETED task
    assert mgr.job_state("j1") == {TaskState.COMPLETED.value: 1}


def test_commit_is_idempotent_first_wins():
    mgr = _mgr()
    mgr.add_task(_task(), now=T0)
    lease = mgr.claim("node-a", now=T0)
    assert mgr.complete(lease.lease_id, output_sha256="a" * 64, now=T0)
    assert not mgr.complete(lease.lease_id, output_sha256="a" * 64, now=T0)  # duplicate


def test_heartbeat_after_expiry_is_refused():
    """A worker whose heartbeat is refused must stop — its lease is gone."""
    mgr = _mgr()
    mgr.add_task(_task(lease_seconds=10), now=T0)
    lease = mgr.claim("node-a", now=T0)
    mgr.sweep(now=T0 + timedelta(seconds=11))
    with pytest.raises(LeaseError):
        mgr.heartbeat(lease.lease_id, now=T0 + timedelta(seconds=12))


def test_worker_reported_failure_requeues_then_exhausts():
    events = []
    mgr = _mgr(events)
    mgr.add_task(_task(max_attempts=2), now=T0)

    l1 = mgr.claim("node-a", now=T0)
    mgr.fail(l1.lease_id, "OOM", now=T0 + timedelta(seconds=5))
    assert mgr.job_state("j1") == {TaskState.PENDING.value: 1}

    l2 = mgr.claim("node-b", now=T0 + timedelta(seconds=10))
    mgr.fail(l2.lease_id, "OOM again", now=T0 + timedelta(seconds=15))
    assert mgr.job_state("j1") == {TaskState.FAILED.value: 1}
    assert EventType.TASK_EXHAUSTED in [e.type for e in events]
    assert mgr.claim("node-c", now=T0 + timedelta(seconds=20)) is None


def test_twelve_trial_sweep_with_one_dead_worker_completes_exactly_once():
    """The hackathon acceptance criterion (master report §15.5): 12 trials,
    one worker dies mid-run, every task accepted exactly once."""
    mgr = _mgr()
    for i in range(12):
        mgr.add_task(_task(task_id=f"trial-{i:02d}", lease_seconds=30), now=T0)

    clock = T0
    workers = ["node-a", "node-b", "node-c"]
    dead = set()
    accepted = 0
    while accepted < 12:
        clock += timedelta(seconds=10)
        for w in workers:
            if w in dead:
                continue
            lease = mgr.claim(w, now=clock)
            if lease is None:
                continue
            # node-b dies while holding trial-05 and never comes back
            if w == "node-b" and lease.payload is not None and lease.task_id == "trial-05" and lease.attempt_number == 1:
                dead.add(w)
                continue  # no heartbeat, no commit — the sweep will handle it
            clock += timedelta(seconds=1)
            if mgr.complete(lease.lease_id, output_sha256="c" * 64, now=clock):
                accepted += 1
        clock += timedelta(seconds=31)  # let any orphaned lease expire

    states = mgr.job_state("j1")
    assert states == {TaskState.COMPLETED.value: 12}


def test_cancel_is_terminal_and_idempotent():
    mgr = _mgr()
    mgr.add_task(_task(), now=T0)
    mgr.cancel_task("j1", "t1")
    mgr.cancel_task("j1", "t1")  # idempotent
    assert mgr.claim("node-a", now=T0) is None
    assert mgr.job_state("j1") == {TaskState.CANCELLED.value: 1}


def test_two_jobs_may_share_a_task_id():
    store = InMemoryLeaseStore()
    store.add(TaskRecord(_task_spec("job-a", "task-000")))
    store.add(TaskRecord(_task_spec("job-b", "task-000")))  # must NOT raise
    assert store.get("job-a", "task-000").spec.job_id == "job-a"
    assert store.get("job-b", "task-000").spec.job_id == "job-b"


def test_duplicate_within_one_job_still_rejected():
    store = InMemoryLeaseStore()
    store.add(TaskRecord(_task_spec("job-a", "task-000")))
    with pytest.raises(ValueError):
        store.add(TaskRecord(_task_spec("job-a", "task-000")))


def test_get_is_job_scoped():
    store = InMemoryLeaseStore()
    store.add(TaskRecord(_task_spec("job-a", "task-000")))
    assert store.get("job-b", "task-000") is None


def test_claim_with_policy_picks_the_right_job():
    """The policy path re-finds its chosen spec in `pending`. Matching on
    task_id alone crosses job boundaries when ids collide."""
    class PickJobB:
        def choose(self, specs, node):
            return next(s for s in specs if s.job_id == "job-b")

    mgr = LeaseManager()
    mgr.add_task(_task_spec("job-a", "task-000"))
    mgr.add_task(_task_spec("job-b", "task-000"))
    lease = mgr.claim("node-1", policy=PickJobB(), node={"node_id": "node-1"})
    assert lease.job_id == "job-b"


def test_cancel_task_is_job_scoped():
    mgr = LeaseManager()
    mgr.add_task(_task_spec("job-a", "task-000"))
    mgr.add_task(_task_spec("job-b", "task-000"))
    mgr.cancel_task("job-a", "task-000")
    assert mgr.records("job-a")[0].state == TaskState.CANCELLED
    assert mgr.records("job-b")[0].state == TaskState.PENDING
