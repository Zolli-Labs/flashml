"""Append-only job/event ledger on SQLite.

POC-scale persistence: one file, synchronous sqlite3 behind a lock, called
via asyncio.to_thread. Postgres is a flashml-cloud concern; the runtime's
ledger only needs durability and ordering.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

from flashruntime.protocol.v1alpha1 import (
    ArtifactRecord,
    Event,
    JobRecord,
    JobState,
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    record TEXT NOT NULL,
    state TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id TEXT NOT NULL,
    type TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_job ON events (job_id, seq);
"""


class Ledger:
    def __init__(self, path: str | Path):
        self._path = str(path)
        self._lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    # -- jobs --------------------------------------------------------------

    def upsert_job(self, job: JobRecord) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, record, state, created_at) VALUES (?,?,?,?) "
                "ON CONFLICT(job_id) DO UPDATE SET record=excluded.record, state=excluded.state",
                (
                    job.job_id,
                    job.model_dump_json(),
                    job.state.value,
                    job.created_at.isoformat(),
                ),
            )

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._lock, self._conn() as conn:
            row = conn.execute(
                "SELECT record FROM jobs WHERE job_id=?", (job_id,)
            ).fetchone()
        return JobRecord.model_validate_json(row["record"]) if row else None

    def list_jobs(self) -> list[JobRecord]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT record FROM jobs ORDER BY created_at DESC"
            ).fetchall()
        return [JobRecord.model_validate_json(r["record"]) for r in rows]

    # -- events ------------------------------------------------------------

    def append_event(self, event: Event) -> None:
        with self._lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO events (job_id, type, timestamp, payload) VALUES (?,?,?,?)",
                (
                    event.job_id,
                    event.type.value,
                    event.timestamp.isoformat(),
                    event.model_dump_json(),
                ),
            )

    def events_for(self, job_id: str) -> list[Event]:
        with self._lock, self._conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM events WHERE job_id=? ORDER BY seq", (job_id,)
            ).fetchall()
        return [Event.model_validate_json(r["payload"]) for r in rows]
