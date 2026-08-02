"""SQLite-backed LeaseStore: the coordinator's lease table survives restarts.

Same semantics as InMemoryLeaseStore — the store keeps live TaskRecord
objects in an insertion-ordered cache (so the manager's in-place mutations
behave identically) and persists every record on `save()`. A new instance
on the same file rehydrates the full state: specs, task states, attempt
counts, the active lease, and the complete lease history — which is what
lets a lease issued *before* a coordinator restart still be renewed,
committed, or rejected *after* it.

Single-writer by design (the service's event loop); `check_same_thread` is
disabled only so test fixtures may construct/inspect across threads.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from flashruntime.leases.store import TaskRecord
from flashruntime.protocol.v1alpha1 import Lease, TaskSpec, TaskState

_SCHEMA = """
CREATE TABLE IF NOT EXISTS lease_tasks (
    task_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts_used INTEGER NOT NULL,
    active_lease_json TEXT,
    accepted_attempt_id TEXT,
    lease_history_json TEXT NOT NULL,
    seq INTEGER,
    PRIMARY KEY (job_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_lease_tasks_job ON lease_tasks (job_id);
"""

_COLUMNS = (
    "task_id, job_id, spec_json, state, attempts_used, active_lease_json,"
    " accepted_attempt_id, lease_history_json, seq"
)


class SqliteLeaseStore:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._migrate()                 # legacy single-PK table → composite
        self._conn.executescript(_SCHEMA)
        self._cache: dict[tuple[str, str], TaskRecord] = {}
        self._seq = 0
        self._load()

    # -- schema migration -----------------------------------------------------

    def _migrate(self) -> None:
        """Upgrade a pre-composite-key database in place.

        SQLite cannot alter a primary key, so the table is rebuilt and the
        rows copied — never dropped. In-flight leases must survive the
        upgrade; that durability is the reason this store exists.
        """
        info = self._conn.execute("PRAGMA table_info(lease_tasks)").fetchall()
        if not info:
            return  # fresh database; _SCHEMA already creates the right table
        pk_cols = sorted(col[1] for col in info if col[5] > 0)
        if pk_cols == ["job_id", "task_id"]:
            return  # already migrated
        self._conn.executescript(
            "BEGIN;"
            "ALTER TABLE lease_tasks RENAME TO lease_tasks_legacy;"
            + _SCHEMA.replace("IF NOT EXISTS lease_tasks", "lease_tasks")
            + f"INSERT INTO lease_tasks ({_COLUMNS})"
            f" SELECT {_COLUMNS} FROM lease_tasks_legacy;"
            "DROP TABLE lease_tasks_legacy;"
            "COMMIT;"
        )
        self._conn.commit()

    # -- rehydration ---------------------------------------------------------

    def _load(self) -> None:
        rows = self._conn.execute(
            "SELECT spec_json, state, attempts_used, active_lease_json,"
            " accepted_attempt_id, lease_history_json, seq"
            " FROM lease_tasks ORDER BY seq"
        ).fetchall()
        for spec_json, state, attempts, lease_json, accepted, history_json, seq in rows:
            record = TaskRecord(TaskSpec.model_validate_json(spec_json))
            record.state = TaskState(state)
            record.attempts_used = attempts
            record.active_lease = Lease.model_validate_json(lease_json) if lease_json else None
            record.accepted_attempt_id = accepted
            record.lease_history = {
                lid: Lease.model_validate(raw)
                for lid, raw in json.loads(history_json).items()
            }
            self._cache[(record.spec.job_id, record.spec.task_id)] = record
            self._seq = max(self._seq, seq or 0)

    # -- LeaseStore protocol -------------------------------------------------

    def add(self, record: TaskRecord) -> None:
        key = (record.spec.job_id, record.spec.task_id)
        if key in self._cache:
            raise ValueError(
                f"task {record.spec.task_id} already exists in job {record.spec.job_id}"
            )
        self._cache[key] = record
        self._seq += 1
        self._persist(record, self._seq)

    def save(self, record: TaskRecord) -> None:
        self._persist(record, None)

    def get(self, job_id: str, task_id: str) -> TaskRecord | None:
        return self._cache.get((job_id, task_id))

    def next_pending(self, job_id: str | None = None) -> TaskRecord | None:
        for record in self._cache.values():
            if record.state == TaskState.PENDING and (
                job_id is None or record.spec.job_id == job_id
            ):
                return record
        return None

    def leased(self) -> list[TaskRecord]:
        return [r for r in self._cache.values() if r.state == TaskState.LEASED]

    def all(self, job_id: str | None = None) -> list[TaskRecord]:
        return [
            r for r in self._cache.values() if job_id is None or r.spec.job_id == job_id
        ]

    # -- persistence ---------------------------------------------------------

    def _persist(self, record: TaskRecord, seq: int | None) -> None:
        history = json.dumps(
            {lid: json.loads(lease.model_dump_json()) for lid, lease in record.lease_history.items()}
        )
        self._conn.execute(
            "INSERT INTO lease_tasks"
            " (task_id, job_id, spec_json, state, attempts_used, active_lease_json,"
            "  accepted_attempt_id, lease_history_json, seq)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?,"
            "   COALESCE(?, (SELECT seq FROM lease_tasks"
            "                 WHERE job_id = ? AND task_id = ?)))"
            " ON CONFLICT(job_id, task_id) DO UPDATE SET"
            "  state=excluded.state, attempts_used=excluded.attempts_used,"
            "  active_lease_json=excluded.active_lease_json,"
            "  accepted_attempt_id=excluded.accepted_attempt_id,"
            "  lease_history_json=excluded.lease_history_json",
            (
                record.spec.task_id,
                record.spec.job_id,
                record.spec.model_dump_json(),
                record.state.value,
                record.attempts_used,
                record.active_lease.model_dump_json() if record.active_lease else None,
                record.accepted_attempt_id,
                history,
                seq,
                record.spec.job_id,
                record.spec.task_id,
            ),
        )
        self._conn.commit()
