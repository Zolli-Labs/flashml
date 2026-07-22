"""Local checkpoint manifests: parts-first / manifest-last for processes
that have only a filesystem (no coordinator).

`write_manifest` hashes every part file already on disk and writes
manifest.json LAST — a crash mid-checkpoint leaves part files but no
manifest, so the checkpoint does not exist. `latest_valid_manifest`
re-verifies every part hash on read: a corrupted or truncated part
disqualifies its manifest, so recovery can never restore from it.

Consumers: `flashruntime.torch.checkpoint()` and the Hugging Face Trainer
callback. Pure stdlib + pydantic (protocol models) — safe in the core.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from flashruntime.protocol.v1alpha1 import (
    CheckpointManifest,
    CheckpointPart,
    CheckpointValidation,
)

MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    step_dir: Path,
    *,
    job_id: str,
    attempt_id: str,
    step: int,
    world_size: int = 1,
    framework: str = "",
) -> CheckpointManifest:
    """Hash every part file in `step_dir`, then write the manifest LAST."""
    step_dir = Path(step_dir)
    parts = [
        CheckpointPart(key=p.name, sha256=_sha256(p), size_bytes=p.stat().st_size)
        for p in sorted(step_dir.iterdir())
        if p.is_file() and p.name != MANIFEST_NAME
    ]
    if not parts:
        raise ValueError(f"no part files in {step_dir}")
    manifest = CheckpointManifest(
        manifest_id=f"ck-{uuid.uuid4().hex[:12]}",
        job_id=job_id,
        attempt_id=attempt_id,
        step=step,
        framework=framework,
        world_size=world_size,
        compatible_world_sizes=[world_size],
        storage_prefix=str(step_dir),
        parts=parts,
        validation=CheckpointValidation.HASH_VERIFIED,
    )
    (step_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))  # LAST
    return manifest


def latest_valid_manifest(ckpt_root: Path, pattern: str = "step-*") -> CheckpointManifest | None:
    """Newest manifest whose parts all re-verify on disk, or None.

    `pattern` matches the per-step directory names ("step-*" for
    flashruntime.torch, "checkpoint-*" for Hugging Face Trainer output).
    """
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.is_dir():
        return None
    best: CheckpointManifest | None = None
    for mf_path in ckpt_root.glob(f"{pattern}/{MANIFEST_NAME}"):
        try:
            manifest = CheckpointManifest.model_validate_json(mf_path.read_text())
        except ValueError:
            continue  # unreadable manifest: treat as nonexistent
        step_dir = mf_path.parent
        intact = all(
            (step_dir / part.key).is_file() and _sha256(step_dir / part.key) == part.sha256
            for part in manifest.parts
        )
        if intact and (best is None or manifest.step > best.step):
            best = manifest
    return best
