"""Checkpoint manifests and compatibility-aware selection.

A checkpoint is a manifest (job, attempt, step, world size, part hashes,
validation status), not just a path. Validity is by construction — parts
upload first, the manifest is written last after every hash verifies — so a
partial checkpoint can never be selected. Recovery restores only from
verified, topology-compatible manifests.

    from flashruntime.checkpoint import CheckpointCatalog

    catalog = CheckpointCatalog(on_event=ledger.append)
    catalog.register_part(job, attempt, step, part)      # after upload
    manifest = catalog.commit(job_id=..., expected_parts=[...], ...)
    catalog.latest_valid(job_id, world_size=4)           # recovery's input
    catalog.lost_work(job_id, failed_at_step=470)        # → 70
"""

from flashruntime.checkpoint.catalog import CheckpointCatalog, CheckpointError

__all__ = ["CheckpointCatalog", "CheckpointError"]
