"""Storage interface for the lease manager, plus the in-memory reference
implementation.

The `LeaseStore` protocol is the seam between the pure state machine
(`manager.py`) and durability: the in-memory store backs unit tests and
embedded use; the FlashRuntime service wires a SQLite/Postgres store behind
the same five methods. Keeping the interface this small is deliberate — the
*semantics* (who may claim, when a commit wins) live entirely in the
manager, never in a store, so every backend behaves identically.
"""

from __future__ import annotations

from typing import Protocol

from flashruntime.protocol.v1alpha1 import Lease, TaskSpec, TaskState


class TaskRecord:
    """Mutable server-side state of one task (not wire-visible).

    `lease_history` keeps every lease ever issued for the task so a late
    commit from a long-dead worker can be *identified and rejected* rather
    than erroring as unknown — rejected duplicates are recovery evidence.
    """

    __slots__ = ("spec", "state", "attempts_used", "active_lease", "accepted_attempt_id", "lease_history")

    def __init__(self, spec: TaskSpec):
        self.spec = spec
        self.state = TaskState.PENDING
        self.attempts_used = 0
        self.active_lease: Lease | None = None
        self.accepted_attempt_id: str | None = None
        self.lease_history: dict[str, Lease] = {}


class LeaseStore(Protocol):
    """Minimal persistence contract for tasks and their leases.

    The manager mutates TaskRecords in memory and calls `save(record)` after
    every state transition — durable stores persist there; the in-memory
    store's live references make it a no-op."""

    def add(self, record: TaskRecord) -> None: ...

    def save(self, record: TaskRecord) -> None: ...

    def get(self, task_id: str) -> TaskRecord | None: ...

    def next_pending(self, job_id: str | None = None) -> TaskRecord | None:
        """Any PENDING task (optionally scoped to a job), or None. Must be
        deterministic for a given state (e.g. insertion order)."""
        ...

    def leased(self) -> list[TaskRecord]:
        """All tasks currently in LEASED state (for the expiry sweep)."""
        ...

    def all(self, job_id: str | None = None) -> list[TaskRecord]: ...


class InMemoryLeaseStore:
    """Reference implementation: insertion-ordered dict, no persistence."""

    def __init__(self) -> None:
        self._tasks: dict[str, TaskRecord] = {}

    def add(self, record: TaskRecord) -> None:
        if record.spec.task_id in self._tasks:
            raise ValueError(f"task {record.spec.task_id} already exists")
        self._tasks[record.spec.task_id] = record

    def save(self, record: TaskRecord) -> None:
        pass  # live references — mutations are already visible

    def get(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def next_pending(self, job_id: str | None = None) -> TaskRecord | None:
        for record in self._tasks.values():
            if record.state == TaskState.PENDING and (
                job_id is None or record.spec.job_id == job_id
            ):
                return record
        return None

    def leased(self) -> list[TaskRecord]:
        return [r for r in self._tasks.values() if r.state == TaskState.LEASED]

    def all(self, job_id: str | None = None) -> list[TaskRecord]:
        return [
            r
            for r in self._tasks.values()
            if job_id is None or r.spec.job_id == job_id
        ]
