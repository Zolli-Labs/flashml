"""state.collect(run_dir) — assemble the /api/state snapshot from disk.

WHY EXCEPTION-SAFETY IS THE WHOLE POINT: this reads a *live* run's directory
while the SDK is still writing into it. metrics.jsonl grows an unterminated
last line between our read and the writer's flush; a manifest may be
half-written; an attempt's output_dir may not exist yet; a file may be
unreadable. A viewer that raised on any of these would turn "someone glanced
at the run" into "the run's dashboard 500'd" — and worse, invite the reflex
to make the SDK slow down or lock for the reader. So collect() is READ-ONLY
and TOTAL: every disk hazard degrades to a partial snapshot (an empty
section, a skipped record, an `"error"` string) and NEVER an exception. The
run's story is never interrupted by being watched.

Data model: pass the parsed `viewer_v1` run.json through verbatim, then
ENRICH it — each attempt gains its own `metrics` (metrics.jsonl tail) and
`log_tail` (launcher.log tail) read from that attempt's `output_dir`, and a
top-level `checkpoints` list is assembled from the job ckpt roots (each root
is the sibling `ckpt/` dir of an attempt dir).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

_METRICS_TAIL = 500  # last N points per attempt — enough for a loss curve, bounded memory
_LOG_TAIL = 100  # last N lines of launcher.log — a glance at what the process last said


def collect(run_dir: Path) -> dict:
    """Read `<run_dir>/run.json` (the viewer_v1 contract) and return the
    enriched /api/state dict. Never raises: an outer backstop converts any
    unforeseen failure into `{"error": ...}` so a viewer can never crash a
    live run — see the module docstring."""
    try:
        return _collect(Path(run_dir))
    except Exception as exc:  # noqa: BLE001 — total by contract; see module docstring
        return {"error": f"viewer snapshot failed: {exc!r}"}


def _collect(run_dir: Path) -> dict:
    run_json = run_dir / "run.json"
    try:
        raw = run_json.read_text(errors="replace")
    except OSError as exc:
        return {"error": f"run.json not found: {exc}"}
    try:
        doc = json.loads(raw)
    except ValueError as exc:
        # A torn read (writer mid-os.replace is atomic, but a truncated file
        # from any other cause still lands here) yields an error, not a crash.
        return {"error": f"run.json unreadable (torn or invalid JSON): {exc}"}
    contract = doc.get("contract") if isinstance(doc, dict) else None
    if contract != "viewer_v1":
        # Fail closed on an unknown contract: a future/foreign schema is not
        # something this version can honestly render, so say so rather than
        # guess at fields that may have moved.
        return {"error": f"unknown run.json contract: {contract!r}"}

    # Enrich each attempt with its own metrics + log tail (both live in the
    # attempt's output_dir), and collect the distinct job ckpt roots as we go.
    ckpt_roots: dict[str, Path] = {}
    enriched: list[dict] = []
    for attempt in doc.get("attempts") or []:
        row = dict(attempt)
        out = row.get("output_dir")
        attempt_dir = Path(out) if out else None
        row["metrics"] = _metrics_tail(attempt_dir) if attempt_dir else []
        row["log_tail"] = _log_tail(attempt_dir) if attempt_dir else ""
        enriched.append(row)
        if attempt_dir is not None:
            # The job ckpt root is the sibling `ckpt/` dir of the attempt dir
            # (`<run>/<job>/<attempt>` → `<run>/<job>/ckpt`); restarts share a
            # job_id, so many attempts map to one root — dedupe by path.
            root = attempt_dir.parent / "ckpt"
            ckpt_roots[str(root)] = root
    doc["attempts"] = enriched

    manifests: list[dict] = []
    for root in ckpt_roots.values():
        try:
            manifests.extend(_manifests_for_root(root))
        except Exception:  # noqa: BLE001 — one bad root never drops the others
            continue
    manifests.sort(key=lambda m: (m.get("job_id", ""), m.get("step", 0)))
    doc["checkpoints"] = manifests
    return doc


def _metrics_tail(attempt_dir: Path, limit: int = _METRICS_TAIL) -> list[dict]:
    """Last `limit` JSON records from `metrics.jsonl`, keys passed through
    as-is. A half-written final line (the writer appended between our read and
    its newline) fails to parse and is skipped — the earlier points survive."""
    try:
        lines = (attempt_dir / "metrics.jsonl").read_text(errors="replace").splitlines()
    except OSError:
        return []  # no metrics yet / dir gone: empty section, not a crash
    records: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except ValueError:
            continue  # torn/partial line — skip it, keep the rest
        if isinstance(rec, dict):
            records.append(rec)
    return records[-limit:]


def _log_tail(attempt_dir: Path, tail_lines: int = _LOG_TAIL) -> str:
    """Last `tail_lines` lines of `launcher.log`, or "" if unreadable."""
    try:
        lines = (attempt_dir / "launcher.log").read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-tail_lines:])


def _manifests_for_root(ckpt_root: Path) -> list[dict]:
    """List every `step-*/manifest.json` under one job ckpt root, each
    re-verified against disk. Reuses `flashruntime.checkpoint.local`'s
    hashing (`verify_manifest`) and its picker (`latest_valid_manifest`) so
    the "is this checkpoint safe" logic is never duplicated here — the viewer
    only *reads* checkpoints, it must agree exactly with what recovery would
    restore. Each entry: step, re-verified validation, part count, age, and
    whether it is the one recovery would pick (`latest_valid`)."""
    from flashruntime.checkpoint.local import (
        MANIFEST_NAME,
        latest_valid_manifest,
        verify_manifest,
    )
    from flashruntime.protocol.v1alpha1 import CheckpointManifest

    ckpt_root = Path(ckpt_root)
    if not ckpt_root.is_dir():
        return []
    latest = latest_valid_manifest(ckpt_root)  # the manifest recovery would restore
    latest_step = latest.step if latest else None
    out: list[dict] = []
    for mf_path in sorted(ckpt_root.glob(f"step-*/{MANIFEST_NAME}")):
        try:
            manifest = CheckpointManifest.model_validate_json(mf_path.read_text())
        except (OSError, ValueError):
            continue  # torn/invalid manifest — skip, keep scanning (never crash)
        valid = verify_manifest(manifest, mf_path.parent)
        try:
            age_s: float | None = time.time() - manifest.created.timestamp()
        except Exception:  # noqa: BLE001 — a weird created value must not crash the scan
            age_s = None
        out.append(
            {
                "job_id": manifest.job_id,
                "step": manifest.step,
                # Report the RE-VERIFIED state, not the manifest's stored claim:
                # a part corrupted after writing makes a "hash_verified" manifest
                # actually invalid, and the viewer must show that truth.
                "validation": "hash_verified" if valid else "invalid",
                "parts": len(manifest.parts),
                "age_s": age_s,
                "latest_valid": bool(valid and manifest.step == latest_step),
            }
        )
    return out
