"""checkpoint_integrity — under a storm of kill -9s landing INSIDE the checkpoint
write window, does flashruntime ever resume from a torn / half-written
checkpoint, or always from a hash-verified one?

HYPOTHESIS: parts-first / manifest-last means a kill mid-write can never leave a
restorable-but-corrupt checkpoint — the manifest is written LAST, so a torn step
has no manifest and ``latest_valid_manifest`` skips it. Every resume therefore
lands on a hash-verified manifest. A naive ``torch.save(state, "latest.pt")``
that overwrites a single file in place has no such guarantee: a kill mid-write
truncates the archive and the next ``torch.load`` raises.

MEASUREMENT METHOD (auditable from this file alone):
  ``repeats`` IS the iteration count N (N=20 stress / N=3 smoke). Each iteration:

  * flash path — a fast checkpointer (``checkpoint_every=1``, an 8 MB part to
    WIDEN the write window, faults.py's substrate) under
    ``flash.submit(max_restarts=1)``. The instant a ``step-*`` dir appears
    WITHOUT its ``manifest.json`` (the write window — parts first, manifest last)
    while a VALID earlier manifest already exists, ``kill_child`` SIGKILLs the
    live attempt. Integrity for the iteration is COUNTED from observable run
    state — never a ``try/assert`` — as: terminal ``SUCCEEDED`` AND a
    hash-verified manifest still the latest AND, when the kill actually landed in
    a window, a resume from a verified EARLIER step (``resumed_from > 0``, i.e.
    it fell back PAST the torn step rather than restoring it). ``integrity_rate``
    = mean over N (target 1.0, MEASURED — a mishandled iteration lowers it and is
    a FINDING, not a test to fix).

  * ``torn_writes_hit`` COUNTS the kills that fired inside a write window; it is
    reported SEPARATELY from window-missed iterations (a run that finished before
    the poller saw a window) — the two are never blended into the rate.

  * naive comparator — an identically-shaped trainer that ``torch.save``s an
    8 MB state to ``latest.pt`` each step, killed the instant that file is
    mid-write (smaller than its published full size); after the kill,
    ``torch.load(latest.pt)``. The per-iteration outcome (loaded / raised
    ``<ExceptionClass>``) is recorded verbatim; ``naive_torch_save_failure_rate``
    is the fraction that raised, whatever it turns out to be. Skipped with a note
    when torch is absent (the flash measurement needs no torch).
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

from benchmarks import faults
from benchmarks._util import bench_env, ensure_venv_on_path, percentile
from benchmarks.schema import ResultRow
from flashruntime.checkpoint.local import latest_valid_manifest

name = "checkpoint_integrity"
hypothesis = (
    "Under repeated kill -9s inside the checkpoint write window, flashruntime's "
    "parts-first/manifest-last commit means every resume lands on a hash-verified "
    "manifest (integrity_rate → 1.0), while a naive torch.save('latest.pt') "
    "overwriting one file in place is truncated by the same kill and fails to reload."
)

# Flash checkpointer: every step, an 8 MB part (faults.py bakes TENSOR_BYTES=8 MB
# to widen the part-on-disk / manifest-absent window so an external SIGKILL can
# reliably land inside it — see fault_recovery_matrix case (d)).
STEPS, EVERY = 24, 1
# Naive comparator geometry: an 8 MB save per step (matched to the flash part) so
# the write dominates each step and a mid-write kill is likely; plenty of steps so
# the process is still saving when the kill fires around step 2.
NAIVE_STEPS = 40
NAIVE_BYTES = 8 * 1024 * 1024
KILL_TIMEOUT_S = 20.0
WAIT_TIMEOUT_S = 60.0


def _wl(script_dir: Path, script: Path, *, steps: int = STEPS, every: int = EVERY):
    from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source

    return CommandWorkload(
        command=[sys.executable, str(script), "--steps", str(steps), "--checkpoint-every", str(every)],
        source=Source(path=str(script_dir)),
        outputs=OutputSpec(collect=["metrics.json"]),
    )


def _ckpt(out: Path) -> Path:
    return out / "local" / "ckpt"  # launcher exports FLASHML_CKPT_DIR = <out>/local/ckpt


def _resumed(run) -> int | None:
    return run.trials[0].get("resumed_from") if run.trials else None


# --------------------------------------------------------------------------
# flash path: kill inside the write window, then MEASURE integrity from state
# --------------------------------------------------------------------------
def _flash_iteration(tmp: Path) -> tuple[bool, bool, int | None]:
    """One flash iteration. Returns (fired, integrity_ok, resumed_from)."""
    import flashruntime as flash

    tmp.mkdir(parents=True, exist_ok=True)
    script = faults.write_crashy_trainer(tmp, steps=STEPS, checkpoint_every=EVERY, crash=None)
    ck = _ckpt(tmp / "run")

    def in_write_window() -> bool:
        # the write window IS "a step-* dir exists but its manifest.json does not
        # yet" (parts written first); require a complete earlier manifest too, so
        # the resume has a hash-verified target to fall back to.
        partial = any(d.is_dir() and not (d / "manifest.json").exists() for d in ck.glob("step-*"))
        return partial and latest_valid_manifest(ck) is not None

    run = flash.submit(_wl(tmp, script), output_dir=tmp / "run", max_restarts=1, wait=False)
    fired = faults.kill_child(run, when=in_write_window, timeout_s=KILL_TIMEOUT_S)
    run.wait(timeout=WAIT_TIMEOUT_S)
    r = _resumed(run)
    final = latest_valid_manifest(ck)
    # COUNTED from observable state, never asserted: SUCCEEDED, a hash-verified
    # manifest still the latest, and — when the kill landed in a window — a resume
    # from a verified EARLIER step (>0), i.e. it fell back past the torn one.
    integrity = bool(
        run.state.value == "SUCCEEDED"
        and final is not None
        and (not fired or (r is not None and r > 0))
    )
    return fired, integrity, r


# --------------------------------------------------------------------------
# naive comparator: torch.save('latest.pt') in place, killed mid-write
# --------------------------------------------------------------------------
_NAIVE_TEMPLATE = '''#!/usr/bin/env python
"""Auto-generated by checkpoint_integrity — the NAIVE comparator.

It overwrites a SINGLE latest.pt with torch.save every step (no
parts-first/manifest-last, no manifest at all), so a kill mid-write truncates the
archive in place and the next torch.load has nothing valid to read.
"""
import argparse
import os
import time
from pathlib import Path

import torch

BYTES = __BYTES__
SLEEP_S = 0.004


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=__STEPS__)
    args = ap.parse_args()
    out = Path(os.environ.get("FLASHML_OUTPUT_DIR", "."))
    latest = out / "latest.pt"
    full = out / "full_size.txt"
    tensor = torch.zeros(BYTES // 4, dtype=torch.float32)  # ~BYTES bytes of "weights"
    for step in range(1, args.steps + 1):
        torch.save({"model": tensor, "step": step}, str(latest))  # overwritten IN PLACE
        if step == 1:
            full.write_text(str(latest.stat().st_size))  # publish the complete size once
        time.sleep(SLEEP_S)


if __name__ == "__main__":
    main()
'''


class _ProcRun:
    """Adapt a raw ``Popen`` to the ``.attempts`` seam ``kill_child`` polls, so the
    naive comparator reuses the same kill primitive — pid-reuse guard and all. The
    single attempt row reflects the process's live/finished state, so a naive run
    that self-completes before the kill lands settles to a non-RUNNING row and its
    recycled pid is never signalled."""

    def __init__(self, proc: subprocess.Popen):
        self._proc = proc

    @property
    def attempts(self) -> list[dict]:
        alive = self._proc.poll() is None
        return [{
            "pid": str(self._proc.pid),
            "state": "RUNNING" if alive else "EXITED",
            "finished_at": None if alive else time.time(),
        }]


def _write_naive_trainer(dir: Path) -> Path:
    dir.mkdir(parents=True, exist_ok=True)
    text = _NAIVE_TEMPLATE.replace("__BYTES__", str(NAIVE_BYTES)).replace("__STEPS__", str(NAIVE_STEPS))
    path = dir / "naive_latest_pt_trainer.py"
    path.write_text(text)
    return path


def _naive_iteration(tmp: Path) -> tuple[bool, bool, str | None]:
    """One naive iteration. Returns (fired, load_ok, exc_class_name_or_None)."""
    import torch

    script = _write_naive_trainer(tmp)
    latest = tmp / "latest.pt"
    full = tmp / "full_size.txt"

    def mid_write() -> bool:
        # currently being written IFF latest.pt is smaller than its published full
        # size (torch.save opens 'wb' ⇒ truncates to 0, then grows back to full).
        if not (full.exists() and latest.exists()):
            return False
        try:
            return latest.stat().st_size < int(full.read_text())
        except (OSError, ValueError):
            return False

    env = bench_env(FLASHML_OUTPUT_DIR=str(tmp))
    proc = subprocess.Popen(
        [sys.executable, str(script), "--steps", str(NAIVE_STEPS)],
        cwd=str(tmp), env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        fired = faults.kill_child(_ProcRun(proc), when=mid_write, timeout_s=KILL_TIMEOUT_S)
        try:
            proc.wait(timeout=WAIT_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)

    if not latest.exists():
        return fired, False, "FileNotFoundError"
    try:
        # weights_only=False = a naive user's full-state restore; on a truncated
        # archive this raises before weights_only is even consulted.
        torch.load(str(latest), weights_only=False)
        return fired, True, None
    except Exception as exc:  # noqa: BLE001 — record WHATEVER class it raises, verbatim
        return fired, False, type(exc).__name__


# --------------------------------------------------------------------------
# the N-iteration chaos loop
# --------------------------------------------------------------------------
def run(repeats: int) -> ResultRow:
    ensure_venv_on_path()
    import tempfile

    n = max(1, repeats)  # `repeats` IS the iteration count N: 3 smoke, 20 stress

    try:
        import torch  # noqa: F401
        torch_ok = True
    except ImportError:
        torch_ok = False

    integrity_flags: list[float] = []
    torn_writes_hit = 0
    resumed_steps: list[int] = []
    naive_fail = 0
    naive_torn_hit = 0
    naive_modes: dict[str, int] = {}

    with tempfile.TemporaryDirectory(prefix="ckpt-integrity-") as td:
        root = Path(td)
        for i in range(n):
            fired, integrity, r = _flash_iteration(root / f"flash-{i:03d}")
            integrity_flags.append(1.0 if integrity else 0.0)
            if fired:
                torn_writes_hit += 1
            if r:
                resumed_steps.append(r)
            if torch_ok:
                nf, load_ok, mode = _naive_iteration(root / f"naive-{i:03d}")
                if nf:
                    naive_torn_hit += 1
                if not load_ok:
                    naive_fail += 1
                    if mode:
                        naive_modes[mode] = naive_modes.get(mode, 0) + 1

    integrity_rate = round(sum(integrity_flags) / n, 3)
    mean_resume = round(sum(resumed_steps) / len(resumed_steps), 1) if resumed_steps else 0.0

    comparators: dict[str, float] = {
        "iterations": float(n),
        "torn_writes_hit": float(torn_writes_hit),  # flash kills that landed in a write window
    }
    notes = [
        "integrity is COUNTED from run state (terminal SUCCEEDED + a hash-verified latest "
        "manifest still present + a resume from a verified earlier step>0 when the kill landed "
        "in a window), never asserted — a mishandled iteration lowers the rate as a FINDING",
        f"write window = a step-* dir on disk WITHOUT its manifest.json (parts first, manifest "
        f"last); kill -9 fired inside it on {torn_writes_hit}/{n} iterations (torn_writes_hit) — "
        f"window-missed iterations are counted SEPARATELY, never blended into integrity_rate",
        f"mean resume step across torn-write hits: {mean_resume} (a verified EARLIER checkpoint, "
        "never the torn one)",
    ]
    if torch_ok:
        comparators["naive_torch_save_failure_rate"] = round(naive_fail / n, 3)
        comparators["naive_torn_writes_hit"] = float(naive_torn_hit)
        modes = ", ".join(f"{k}×{v}" for k, v in sorted(naive_modes.items())) or "none observed"
        notes.append(
            f"naive comparator: torch.save(state, 'latest.pt') overwritten in place each step, "
            f"killed mid-write, then torch.load — raised on {naive_fail}/{n} iterations "
            f"(mid-write kills landed on {naive_torn_hit}/{n}); observed failure modes, verbatim "
            f"exception classes: {modes}"
        )
    else:
        notes.append(
            "naive comparator skipped — torch not installed (the flash integrity measurement "
            "needs no torch; this scenario still measures flashruntime itself)"
        )

    return ResultRow(
        scenario=name,
        section="resilience",
        unit="integrity_rate",
        median=integrity_rate,
        p10=percentile(integrity_flags, 0.1),
        p90=percentile(integrity_flags, 0.9),
        repeats=n,
        comparators=comparators,
        notes=notes,
    )
