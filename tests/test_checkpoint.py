"""Checkpoint catalog tests: validity by construction, honest selection."""

from __future__ import annotations

import pytest

from flashruntime.checkpoint import CheckpointCatalog, CheckpointError
from flashruntime.protocol.v1alpha1 import CheckpointPart, CheckpointValidation, EventType


def _part(key, sha="a"):
    return CheckpointPart(key=key, sha256=sha * 64, size_bytes=100)


def _committed(catalog, step, *, attempt="at1", world=4, compatible=None, parts=("rank0.pt", "rank1.pt")):
    expected = [_part(k) for k in parts]
    for p in expected:
        catalog.register_part("j1", attempt, step, p)
    return catalog.commit(
        job_id="j1",
        attempt_id=attempt,
        step=step,
        expected_parts=expected,
        storage_prefix=f"artifact://checkpoints/j1/{attempt}/{step}/",
        world_size=world,
        compatible_world_sizes=compatible,
    )


def test_partial_upload_can_never_become_a_checkpoint():
    """The §H rule: one missing part ⇒ commit raises, no manifest exists."""
    events = []
    catalog = CheckpointCatalog(on_event=events.append)
    catalog.register_part("j1", "at1", 100, _part("rank0.pt"))
    # rank1.pt was never uploaded — the crash-mid-upload scenario
    with pytest.raises(CheckpointError, match="missing part rank1.pt"):
        catalog.commit(
            job_id="j1",
            attempt_id="at1",
            step=100,
            expected_parts=[_part("rank0.pt"), _part("rank1.pt")],
            storage_prefix="artifact://checkpoints/j1/at1/100/",
        )
    assert catalog.latest_valid("j1") is None
    assert [e.type for e in events] == [EventType.CHECKPOINT_REJECTED]


def test_hash_mismatch_rejects_commit():
    catalog = CheckpointCatalog()
    catalog.register_part("j1", "at1", 100, _part("rank0.pt", sha="a"))
    with pytest.raises(CheckpointError, match="hash mismatch"):
        catalog.commit(
            job_id="j1",
            attempt_id="at1",
            step=100,
            expected_parts=[_part("rank0.pt", sha="b")],  # expected ≠ uploaded
            storage_prefix="artifact://checkpoints/j1/at1/100/",
        )


def test_verified_commit_and_latest_valid_selection():
    catalog = CheckpointCatalog()
    _committed(catalog, 100)
    m2 = _committed(catalog, 200)
    assert m2.validation is CheckpointValidation.HASH_VERIFIED
    assert catalog.latest_valid("j1").step == 200


def test_topology_compatibility_gates_selection():
    """A world_size=4 checkpoint that only reshards to {2,4} must not be
    offered to a world of 8."""
    catalog = CheckpointCatalog()
    _committed(catalog, 100, world=4, compatible=[2, 4])
    assert catalog.latest_valid("j1", world_size=4) is not None
    assert catalog.latest_valid("j1", world_size=2) is not None
    assert catalog.latest_valid("j1", world_size=8) is None


def test_quarantine_falls_back_to_previous_valid():
    catalog = CheckpointCatalog()
    m1 = _committed(catalog, 100)
    m2 = _committed(catalog, 200)
    catalog.quarantine(m2.manifest_id, "restore failed")
    assert catalog.latest_valid("j1").manifest_id == m1.manifest_id


def test_restore_verified_is_preferred_and_requirable():
    catalog = CheckpointCatalog()
    m1 = _committed(catalog, 100)
    _committed(catalog, 200)  # newer, but only hash-verified
    catalog.mark_restore_verified(m1.manifest_id)
    # default: newest step still wins…
    assert catalog.latest_valid("j1").step == 200
    # …but a caller demanding restore-verified gets the proven one
    assert catalog.latest_valid("j1", require_restore_verified=True).step == 100


def test_lost_work_is_the_recovery_economics_number():
    """Killed at step 470 with a checkpoint at 400 ⇒ 70 steps lost."""
    catalog = CheckpointCatalog()
    _committed(catalog, 400)
    assert catalog.lost_work("j1", failed_at_step=470) == 70
    assert catalog.lost_work("j-unknown", failed_at_step=470) is None
