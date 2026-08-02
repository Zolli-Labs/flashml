"""The checkpoint catalog: validity by construction, selection by policy.

The core rule (evaluation §H): **parts first, manifest last**. Workers
upload shard files and register each one (key + sha256 + size). Only when a
manifest's *expected* part set exactly matches the *registered* parts —
every key present, every hash equal — does `commit()` store the manifest
with `hash_verified` status. Until that moment the checkpoint does not
exist: a crash mid-upload leaves orphan parts, never a selectable
checkpoint. There is no code path that can mark a partial checkpoint valid,
which is stronger than any check.

Validation ladder: `hash_verified` (cheap, automatic) → `restore_verified`
(a real load succeeded — recovery prefers these) → `invalid` (quarantined;
never selectable again).

Selection: `latest_valid()` returns the newest-by-step manifest that is
(a) verified, (b) compatible with the requested world size (its own, or one
listed in `compatible_world_sizes` — resharding territory), and (c) not
quarantined. Recovery restores from that or reports honest lost work.

Storage note: this catalog is deliberately in-memory + event-emitting; the
FlashRuntime service persists it by replaying the same events into its
ledger. The blob bytes themselves live in the artifact store — the catalog
holds *metadata about* checkpoints, never checkpoint data.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Callable

from flashruntime.protocol.v1alpha1 import (
    CheckpointManifest,
    CheckpointPart,
    CheckpointValidation,
    Event,
    EventType,
)

EventSink = Callable[[Event], None]


class CheckpointError(Exception):
    """Refused catalog operation (mismatched parts, unknown manifest, ...)."""


class CheckpointCatalog:
    def __init__(self, on_event: EventSink | None = None) -> None:
        self._on_event = on_event
        self._manifests: dict[str, CheckpointManifest] = {}
        # (job_id, attempt_id, step) -> {part_key: (sha256, size_bytes)}
        self._registered: dict[tuple[str, str, int], dict[str, tuple[str, int]]] = {}

    # -- upload side --------------------------------------------------------

    def register_part(
        self, job_id: str, attempt_id: str, step: int, part: CheckpointPart
    ) -> None:
        """Record one uploaded shard file (call after the bytes are durably
        in the artifact store, with the hash the store reported)."""
        bucket = self._registered.setdefault((job_id, attempt_id, step), {})
        bucket[part.key] = (part.sha256, part.size_bytes)

    def commit(
        self,
        *,
        job_id: str,
        attempt_id: str,
        step: int,
        expected_parts: list[CheckpointPart],
        storage_prefix: str,
        world_size: int = 1,
        compatible_world_sizes: list[int] | None = None,
        framework: str = "",
        strategy_family: str = "",
        checkpoint_duration_s: float | None = None,
    ) -> CheckpointManifest:
        """Write the manifest — the two-phase-commit point.

        Verifies every expected part against what was registered; any
        missing key or hash mismatch raises (and emits CHECKPOINT_REJECTED),
        leaving no manifest behind.
        """
        registered = self._registered.get((job_id, attempt_id, step), {})
        problems: list[str] = []
        for part in expected_parts:
            got = registered.get(part.key)
            if got is None:
                problems.append(f"missing part {part.key}")
            elif got[0] != part.sha256:
                problems.append(f"hash mismatch for {part.key}")
        if not expected_parts:
            problems.append("manifest with zero parts")
        if problems:
            self._emit(
                EventType.CHECKPOINT_REJECTED,
                job_id,
                detail=f"step {step}: " + "; ".join(problems),
            )
            raise CheckpointError("; ".join(problems))

        manifest = CheckpointManifest(
            manifest_id=f"ck-{uuid.uuid4().hex[:12]}",
            job_id=job_id,
            attempt_id=attempt_id,
            step=step,
            framework=framework,
            strategy_family=strategy_family,
            world_size=world_size,
            compatible_world_sizes=compatible_world_sizes or [world_size],
            storage_prefix=storage_prefix,
            parts=expected_parts,
            validation=CheckpointValidation.HASH_VERIFIED,
        )
        if checkpoint_duration_s is not None:
            manifest.checkpoint_duration_s = checkpoint_duration_s
        self._manifests[manifest.manifest_id] = manifest
        self._emit(
            EventType.CHECKPOINT_MANIFEST_COMMITTED,
            job_id,
            detail=f"step {step}, {len(expected_parts)} part(s), world_size {world_size}",
        )
        return manifest

    # -- validation ladder --------------------------------------------------

    def mark_restore_verified(self, manifest_id: str) -> None:
        """Upgrade after a real, successful load (e.g. the profiling stage's
        save/restore cycle, or a completed recovery)."""
        manifest = self._require(manifest_id)
        manifest.validation = CheckpointValidation.RESTORE_VERIFIED
        self._emit(EventType.CHECKPOINT_RESTORE_VERIFIED, manifest.job_id, detail=f"step {manifest.step}")

    def quarantine(self, manifest_id: str, reason: str) -> None:
        """Mark invalid (e.g. restore failed, producer node untrusted). A
        quarantined manifest is never selectable again."""
        manifest = self._require(manifest_id)
        manifest.validation = CheckpointValidation.INVALID
        self._emit(EventType.CHECKPOINT_REJECTED, manifest.job_id, detail=f"quarantined step {manifest.step}: {reason}")

    # -- selection ----------------------------------------------------------

    def latest_valid(
        self,
        job_id: str,
        world_size: int | None = None,
        require_restore_verified: bool = False,
    ) -> CheckpointManifest | None:
        """Newest verified, topology-compatible manifest — recovery's input.
        Ties on step break toward restore-verified, then newest creation."""
        acceptable = (
            (CheckpointValidation.RESTORE_VERIFIED,)
            if require_restore_verified
            else (CheckpointValidation.HASH_VERIFIED, CheckpointValidation.RESTORE_VERIFIED)
        )
        candidates = [
            m
            for m in self._manifests.values()
            if m.job_id == job_id
            and m.validation in acceptable
            and (world_size is None or world_size in m.compatible_world_sizes)
        ]
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda m: (m.step, m.validation == CheckpointValidation.RESTORE_VERIFIED, m.created),
        )

    def lost_work(self, job_id: str, failed_at_step: int, world_size: int | None = None) -> int | None:
        """Steps lost if recovery happens now — the recovery-economics
        number ("killed at 470, resume from 400 ⇒ 70 lost")."""
        manifest = self.latest_valid(job_id, world_size)
        if manifest is None:
            return None
        return max(0, failed_at_step - manifest.step)

    # -- internals ----------------------------------------------------------

    def _require(self, manifest_id: str) -> CheckpointManifest:
        manifest = self._manifests.get(manifest_id)
        if manifest is None:
            raise CheckpointError(f"unknown manifest {manifest_id}")
        return manifest

    def _emit(self, event_type: EventType, job_id: str, detail: str) -> None:
        if self._on_event is None:
            return
        self._on_event(
            Event(
                type=event_type,
                job_id=job_id,
                source="flashruntime.checkpoint",
                message=detail,
                timestamp=datetime.now(timezone.utc),
            )
        )
