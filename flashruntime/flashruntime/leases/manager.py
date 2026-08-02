"""The Mode A lease state machine — FlashRuntime's core reliability pattern.

    Work is never pushed to a machine. A machine *claims* a time-limited
    lease on a task, proves liveness with *heartbeats*, and only the first
    attempt to *commit* a valid result wins.

A dead worker needs no special handling: its lease passes the deadline, the
sweep expires it, and the task returns to PENDING for someone else. A worker
that wakes up late and tries to commit is rejected by the idempotency rule.

The manager is a pure state machine: no I/O, no threads, no clock of its
own. Time is injected (`now` parameter) so tests control it exactly;
durability is behind `LeaseStore`; every transition is emitted as a typed
protocol `Event` through a callback — the event ledger, dashboards, and
metrics all hang off that single stream.

State transitions (task):

    PENDING ──claim──▶ LEASED ──complete (first valid)──▶ COMPLETED
       ▲                 │
       └──sweep expiry / fail (attempts left)──┘
    LEASED ──fail/expiry with no attempts left──▶ FAILED
    any non-terminal ──cancel──▶ CANCELLED
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Callable

from flashruntime.leases.store import InMemoryLeaseStore, LeaseStore, TaskRecord
from flashruntime.protocol.v1alpha1 import (
    Event,
    EventType,
    Lease,
    TaskSpec,
    TaskState,
)

EventSink = Callable[[Event], None]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class LeaseError(Exception):
    """A request that the state machine must refuse (wrong holder, terminal
    task, unknown id). Callers translate this to their transport's error."""


class LeaseManager:
    """Coordinates tasks, leases, heartbeats, expiry, and idempotent commit.

    Embeddable anywhere: the FlashRuntime service exposes it over HTTP, but
    a script can drive it in-process (see tests/test_leases.py for the full
    kill-a-worker story in twenty lines).
    """

    def __init__(
        self,
        store: LeaseStore | None = None,
        on_event: EventSink | None = None,
    ) -> None:
        self._store = store if store is not None else InMemoryLeaseStore()
        self._on_event = on_event

    # -- job side -----------------------------------------------------------

    def add_task(self, spec: TaskSpec, now: datetime | None = None) -> None:
        """Register one unit of claimable work."""
        self._store.add(TaskRecord(spec))
        self._emit(EventType.TASK_CREATED, spec.job_id, spec.task_id, now=now)

    def cancel_task(self, job_id: str, task_id: str, now: datetime | None = None) -> None:
        record = self._require(job_id, task_id)
        if record.state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED):
            return  # terminal states are final; cancel is idempotent
        record.state = TaskState.CANCELLED
        record.active_lease = None
        self._store.save(record)

    # -- worker side --------------------------------------------------------

    def claim(
        self,
        node_id: str,
        job_id: str | None = None,
        now: datetime | None = None,
        policy: object | None = None,
        node: dict | None = None,
    ) -> Lease | None:
        """Claim the next PENDING task for `node_id`, or None when nothing is
        claimable. Expired leases are swept first so a claim never starves
        behind a dead worker.

        `policy`/`node` are the scheduler seam (flashruntime/scheduler):
        the store yields queue-ordered candidates, the policy filters and
        picks. Duck-typed (`choose(pending_specs, node) -> TaskSpec|None`)
        so this package gains no scheduler import. Without a policy,
        behavior is bit-identical to the original FIFO claim."""
        now = now or _utcnow()
        self.sweep(now=now)
        if policy is None:
            record = self._store.next_pending(job_id)
        else:
            pending = [r for r in self._store.all(job_id) if r.state == TaskState.PENDING]
            chosen = policy.choose([r.spec for r in pending], node or {"node_id": node_id})
            record = None
            if chosen is not None:
                # default None: a policy that returns a foreign/unknown spec
                # falls through to the `record is None` path below, not StopIteration.
                # Match on both fields: task_id alone is positional within a
                # job (task-000), so two jobs can collide on it.
                record = next(
                    (
                        r
                        for r in pending
                        if r.spec.task_id == chosen.task_id
                        and r.spec.job_id == chosen.job_id
                    ),
                    None,
                )
        if record is None:
            return None
        record.attempts_used += 1
        lease = Lease(
            lease_id=f"ls-{uuid.uuid4().hex[:12]}",
            task_id=record.spec.task_id,
            job_id=record.spec.job_id,
            node_id=node_id,
            attempt_number=record.attempts_used,
            deadline=now + timedelta(seconds=record.spec.lease_seconds),
            payload=record.spec.payload,
        )
        record.state = TaskState.LEASED
        record.active_lease = lease
        record.lease_history[lease.lease_id] = lease
        self._store.save(record)
        self._emit(
            EventType.LEASE_CLAIMED,
            lease.job_id,
            lease.task_id,
            node_id=node_id,
            detail=f"attempt {lease.attempt_number}, deadline {lease.deadline.isoformat()}",
            now=now,
        )
        return lease

    def heartbeat(self, lease_id: str, now: datetime | None = None) -> Lease:
        """Renew a live lease (extends the deadline by the task's lease
        window). Refused for unknown/expired/superseded leases — a worker
        whose heartbeat is refused must stop working on the task."""
        now = now or _utcnow()
        record, lease = self._require_live_lease(lease_id, now)
        renewed = lease.model_copy(
            update={"deadline": now + timedelta(seconds=record.spec.lease_seconds)}
        )
        record.active_lease = renewed
        self._store.save(record)
        self._emit(EventType.LEASE_RENEWED, lease.job_id, lease.task_id, node_id=lease.node_id, now=now)
        return renewed

    def complete(
        self,
        lease_id: str,
        output_sha256: str,
        now: datetime | None = None,
    ) -> bool:
        """Commit the result of a leased attempt.

        Returns True when the commit is *accepted*. The idempotency rule:
        exactly one accepted output per task, ever. A late commit from an
        expired or superseded lease returns False (rejected) — and is
        recorded, because rejected duplicates are recovery evidence, not
        noise. Raises LeaseError only for leases that never existed.
        """
        now = now or _utcnow()
        record, lease = self._find_lease(lease_id)
        if record.state == TaskState.COMPLETED or not self._is_live(record, lease, now):
            self._emit(
                EventType.TASK_COMMIT_REJECTED,
                lease.job_id,
                lease.task_id,
                node_id=lease.node_id,
                detail=f"late/duplicate commit from attempt {lease.attempt_number} rejected",
                now=now,
            )
            return False
        record.state = TaskState.COMPLETED
        record.accepted_attempt_id = lease.lease_id
        record.active_lease = None
        self._store.save(record)
        self._emit(
            EventType.TASK_COMMIT_ACCEPTED,
            lease.job_id,
            lease.task_id,
            node_id=lease.node_id,
            detail=f"attempt {lease.attempt_number} accepted, sha256={output_sha256[:12]}…",
            now=now,
        )
        return True

    def fail(self, lease_id: str, reason: str, now: datetime | None = None) -> None:
        """A worker reports its attempt failed. The task requeues while
        attempts remain, otherwise it is FAILED for good."""
        now = now or _utcnow()
        record, lease = self._require_live_lease(lease_id, now)
        self._emit(
            EventType.TASK_ATTEMPT_FAILED,
            lease.job_id,
            lease.task_id,
            node_id=lease.node_id,
            detail=reason,
            now=now,
        )
        self._release(record, now, requeue_event=EventType.TASK_REQUEUED)

    # -- coordinator side ---------------------------------------------------

    def sweep(self, now: datetime | None = None) -> int:
        """Expire every lease past its deadline; requeue (or exhaust) the
        tasks. Returns how many leases were expired. Call on a timer — or
        rely on `claim`, which sweeps first."""
        now = now or _utcnow()
        expired = 0
        for record in self._store.leased():
            lease = record.active_lease
            if lease is None or lease.deadline > now:
                continue
            expired += 1
            self._emit(
                EventType.LEASE_EXPIRED,
                lease.job_id,
                lease.task_id,
                node_id=lease.node_id,
                detail=f"attempt {lease.attempt_number} heartbeat deadline passed",
                now=now,
            )
            self._release(record, now, requeue_event=EventType.TASK_REQUEUED)
        return expired

    def job_state(self, job_id: str) -> dict[str, int]:
        """Task counts by state — the number the dashboard shows."""
        counts: dict[str, int] = {}
        for record in self._store.all(job_id):
            counts[record.state.value] = counts.get(record.state.value, 0) + 1
        return counts

    def records(self, job_id: str | None = None) -> list[TaskRecord]:
        """Read-only view of task records (for status endpoints/dashboards)."""
        return self._store.all(job_id)

    def lease_info(self, lease_id: str) -> Lease | None:
        """Look up any lease ever issued (live or superseded), or None."""
        try:
            _, lease = self._find_lease(lease_id)
        except LeaseError:
            return None
        return lease

    def live_leases_for_node(
        self, node_id: str, now: datetime | None = None
    ) -> set[tuple[str, str]]:
        """(job_id, task_id) this node may currently write to.

        Delegates liveness to `_is_live` rather than re-testing `deadline`
        itself: that predicate also requires the lease to still be the active
        one and the record to still be LEASED, so a task whose result was
        already accepted stops being writable. Two copies of this rule would
        drift, and the drift would be a silent authorization hole.
        """
        now = now if now is not None else _utcnow()
        scope: set[tuple[str, str]] = set()
        for record in self._store.leased():
            lease = record.active_lease
            if lease is None or lease.node_id != node_id:
                continue
            # At this call site, `_is_live`'s `current.lease_id ==
            # lease.lease_id` check is always true (we pass in
            # `record.active_lease` itself) and its `record.state ==
            # LEASED` check is already guaranteed by `_store.leased()`
            # above — so today this reduces to the deadline comparison.
            # Keep the delegation anyway: `leased()`'s filtering is a
            # store-level contract that could change, and `_is_live` is
            # the single canonical definition of "this lease is still
            # good". Two copies of an authorization rule drift apart,
            # and the drift would be a silent hole.
            if not self._is_live(record, lease, now):
                continue
            scope.add((lease.job_id, lease.task_id))
        return scope

    # -- internals ----------------------------------------------------------

    def _release(self, record: TaskRecord, now: datetime, requeue_event: EventType) -> None:
        record.active_lease = None
        if record.attempts_used >= record.spec.max_attempts:
            record.state = TaskState.FAILED
            self._emit(
                EventType.TASK_EXHAUSTED,
                record.spec.job_id,
                record.spec.task_id,
                detail=f"all {record.spec.max_attempts} attempts used",
                now=now,
            )
        else:
            record.state = TaskState.PENDING
            self._emit(requeue_event, record.spec.job_id, record.spec.task_id, now=now)
        self._store.save(record)

    def _is_live(self, record: TaskRecord, lease: Lease, now: datetime) -> bool:
        current = record.active_lease
        return (
            current is not None
            and current.lease_id == lease.lease_id
            and current.deadline > now
            and record.state == TaskState.LEASED
        )

    def _require(self, job_id: str, task_id: str) -> TaskRecord:
        record = self._store.get(job_id, task_id)
        if record is None:
            raise LeaseError(f"unknown task {task_id} in job {job_id}")
        return record

    def _find_lease(self, lease_id: str) -> tuple[TaskRecord, Lease]:
        # Every issued lease stays in its task's history, so a superseded or
        # long-dead lease is still *identifiable* — its commit gets rejected
        # with evidence instead of erroring as unknown.
        for record in self._store.all():
            lease = record.lease_history.get(lease_id)
            if lease is not None:
                return record, lease
        raise LeaseError(f"unknown lease {lease_id}")

    def _require_live_lease(self, lease_id: str, now: datetime) -> tuple[TaskRecord, Lease]:
        record, lease = self._find_lease(lease_id)
        if not self._is_live(record, lease, now):
            raise LeaseError(f"lease {lease_id} is no longer live")
        return record, lease

    def _emit(
        self,
        event_type: EventType,
        job_id: str,
        task_id: str,
        *,
        node_id: str | None = None,
        detail: str = "",
        now: datetime | None = None,
    ) -> None:
        if self._on_event is None:
            return
        self._on_event(
            Event(
                type=event_type,
                job_id=job_id,
                source="flashruntime.leases",
                message=detail or event_type.value,
                data={"task_id": task_id, **({"node_id": node_id} if node_id else {})},
                timestamp=now or _utcnow(),
            )
        )
