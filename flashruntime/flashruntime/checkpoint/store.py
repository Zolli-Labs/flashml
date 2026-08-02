"""Manifest persistence seam (research item R1).

Today `CheckpointCatalog` keeps manifests in a plain dict — a coordinator
restart orphans perfectly good checkpoint *files* because the metadata
proving their validity dies with the process. This module defines the
persistence interface that closes that gap, mirroring the lease pattern
that already works (`leases/store.py` + `leases/sqlite_store.py`):

    the catalog mutates live objects in an insertion-ordered cache and
    calls `save(manifest)` after every transition; a durable store
    persists there and rehydrates the cache at construction.

Migration path (half a day, SPRINT_PLAN Day 3):
1. `CheckpointCatalog.__init__` grows `store: ManifestStore | None = None`
   (default `InMemoryManifestStore()` — zero behavior change).
2. The catalog's `_manifests` dict is replaced by the store's cache, and
   every mutation (`commit`, `mark_restore_verified`, `quarantine`) ends
   with `store.save(manifest)`.
3. `SqliteManifestStore` lands beside `leases.db` (one table: manifest_id
   PK, scope job_id, manifest JSON, seq for order) with the same
   COALESCE-seq upsert as `SqliteLeaseStore._persist` — copy that code,
   it is tested and correct.
4. The conformance test is `tests/test_checkpoint.py` re-run against the
   new store, plus a restart-survival test shaped exactly like
   `tests/test_leases_sqlite.py::test_inflight_work_survives_coordinator_restart`.

Note the pending-parts registry (`CheckpointCatalog._registered`) is
deliberately NOT persisted: parts without a committed manifest are, by the
manifest-last rule, not checkpoints — losing them on restart just means
the in-flight checkpoint re-uploads, which the relay already handles.
"""

from __future__ import annotations

from typing import Protocol

from flashruntime.protocol.v1alpha1 import CheckpointManifest

__all__ = ["ManifestStore", "InMemoryManifestStore"]


class ManifestStore(Protocol):
    """Minimal persistence contract for checkpoint manifests.

    Semantics (identical in spirit to `LeaseStore`):
    - `add` registers a NEW manifest (duplicate manifest_id → ValueError);
    - `save` persists the current state of an already-added manifest
      (validation upgrades/quarantine); no-op for in-memory stores whose
      cache holds live references;
    - `get`/`all` read from the cache — `all(scope)` filters by the
      catalog's scope key (the service uses `f"{job_id}::{task_id}"`) and
      MUST preserve insertion order (selection tie-breaks depend on it).
    """

    def add(self, manifest: CheckpointManifest) -> None: ...

    def save(self, manifest: CheckpointManifest) -> None: ...

    def get(self, manifest_id: str) -> CheckpointManifest | None: ...

    def all(self, scope: str | None = None) -> list[CheckpointManifest]: ...


class InMemoryManifestStore:
    """Reference implementation: live references, insertion-ordered,
    volatile. This is functionally what the catalog does today — extracted
    behind the seam so the SQLite store is a drop-in."""

    def __init__(self) -> None:
        self._manifests: dict[str, CheckpointManifest] = {}

    def add(self, manifest: CheckpointManifest) -> None:
        if manifest.manifest_id in self._manifests:
            raise ValueError(f"manifest {manifest.manifest_id} already exists")
        self._manifests[manifest.manifest_id] = manifest

    def save(self, manifest: CheckpointManifest) -> None:
        pass  # live references — mutations already visible

    def get(self, manifest_id: str) -> CheckpointManifest | None:
        return self._manifests.get(manifest_id)

    def all(self, scope: str | None = None) -> list[CheckpointManifest]:
        return [
            m for m in self._manifests.values() if scope is None or m.job_id == scope
        ]
