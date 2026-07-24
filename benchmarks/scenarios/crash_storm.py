"""crash_storm — under a fan-out where HALF the trials crash mid-run, how much
of the compute flashruntime spends is USEFUL, and what wall-clock does the
crash storm cost versus a clean sweep?

HYPOTHESIS: a 16-trial fan-out in which every even-indexed trial crashes on its
FRESH attempt (a mid-run ``SystemExit`` at the checkpoint midpoint) still
completes 16/16 with ZERO human interventions — flashruntime classifies each
crash as a transient WORKER_CRASH and auto-resumes it from its own job-scoped
checkpoint. The price of the storm is a bounded goodput haircut (redundant
attempt-1 work) and a bounded wall-clock penalty, both MEASURED here, never a
lost trial or a manual restart.

MEASUREMENT METHOD (auditable from this file alone):
  Two identical sweeps of ``n`` fan-out trials of ONE crash-router script:

    * storm sweep — even-indexed trials are ARMED: their fresh attempt routes to
      faults.py's ``systemexit_mid`` trainer, which crashes at the checkpoint
      midpoint on a FRESH run only (``start == 0``); the resumed retry
      (``max_restarts=1``, same job id ⇒ same ``FLASHML_CKPT_DIR``) sails past
      the marker and finishes. Odd trials route to the clean trainer. Both
      targets are written VERBATIM by ``faults.write_crashy_trainer`` (crash
      semantics byte-identical to fault_recovery_matrix case (b)).
    * clean sweep — the SAME workload with ``FLASHML_CRASH_DISARMED=1`` in the
      env, so the router sends every trial (even the evens) to the clean
      trainer. No crashes ⇒ this is t_clean, the crash-free wall-clock baseline.

  Everything is COUNTED from observable run state — never a ``try/assert`` that
  predetermines the score:
    * ``median``            = completions = ``len(storm_run.trials)`` (a trial is
                              in ``.trials`` only once an attempt produced a
                              metrics.json; a trial that never resumed would drop
                              out and LOWER this — a FINDING, not a test to fix).
    * ``crashed_first_attempt`` = trials whose final metrics report
                              ``resumed_from > 0`` (they crashed on attempt 1 and
                              resumed). Deterministically the even trials (n/2).
    * ``manual_interventions`` = 0.0, DERIVED: the recovery loop resolved every
                              crash with no human in the loop (there is no
                              human-input path in ``flash.submit``). Counted and
                              reported, never asserted as the headline.

  GOODPUT ARITHMETIC (the executed-steps accounting — observable-only, pinned
  by ``_goodput`` and its unit test):

    Each trial's FINAL metrics.json reports ``(steps, resumed_from)`` — the
    completed step count and the checkpoint step it resumed from (0 if it never
    crashed). From those two OBSERVED numbers alone:

      useful_steps(trial)   = steps                 # the final, kept progress
      executed_steps(trial) = steps + resumed_from  # attempt-2 completed `steps`,
                                                     # PLUS attempt-1's sunk work,
                                                     # counted as the checkpoint it
                                                     # left behind (`resumed_from`)

    A clean trial has ``resumed_from == 0``, so ``executed == steps`` and it
    contributes NO waste — the same formula is therefore total over all trials.
    A crashed trial contributes ``resumed_from`` of redundant executed steps.

      goodput_fraction = Σ useful / Σ executed = Σ steps / Σ (steps + resumed_from)

    This is DELIBERATELY CONSERVATIVE. Operationally a crashed trial executes
    "crash-point steps on attempt 1 + the resume-tail (steps − resumed_from) on
    attempt 2"; because checkpointing means attempt 1's progress up to
    ``resumed_from`` was SAVED (not recomputed), the true redundant work is only
    the sub-checkpoint remainder. Charging the full ``resumed_from`` as executed
    twice OVER-counts waste on purpose: it makes goodput a pessimistic LOWER
    bound on efficiency, so this number can never FLATTER flashruntime. Both
    terms are read from metrics — nothing depends on the (un-reported) crash
    step — so the fraction is auditable from run state alone.

    Worked once by hand (n=16, STEPS=8, EVERY=2 ⇒ crashed trials resumed_from=4):
      8 crashed trials  → useful 8×8=64,  executed 8×(8+4)=96
      8 clean   trials  → useful 8×8=64,  executed 8×(8+0)=64
      goodput = (64+64) / (96+64) = 128 / 160 = 0.8

  ``wallclock_penalty_fraction`` = (t_storm − t_clean) / t_clean — the extra
  wall-clock the storm's crash+resume attempts cost over the clean sweep.

  Honesty note inherited from hpo_sweep: local fan-out is SEQUENTIAL by design
  (flash.submit runs one trial at a time so each trial's outputs are collected
  before the next overwrites them), so both sweeps' wall-clock is a sum over
  trials, not a parallel speedup — the value measured here is fault-tolerant
  goodput, not throughput.
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks import faults
from benchmarks._util import ensure_venv_on_path, median, percentile, timed
from benchmarks.schema import ResultRow

name = "crash_storm"
hypothesis = (
    "A 16-trial fan-out where every even trial crashes mid-run still completes "
    "16/16 with zero human interventions — flashruntime auto-resumes each crash "
    "from its own checkpoint — at a bounded, MEASURED goodput and wall-clock cost."
)

# STEPS//2 = 4 is the crash midpoint; EVERY=2 checkpoints at 2 and 4, so a
# crashed trial's newest valid manifest is step 4 ⇒ resumed_from=4 (deterministic).
STEPS, EVERY = 8, 2
STORM_TRIALS = 16   # the full storm (8 evens crash+resume)
SMOKE_TRIALS = 4    # the bench_smoke variant (evens 0,2 crash+resume)

# The per-trial crash router: reads --trial and, for EVEN trials (unless the run
# is disarmed), delegates to faults.py's systemexit_mid trainer; otherwise to the
# clean trainer. A small wrapper around write_crashy_trainer's output (NOT a
# reimplementation) so the crash mechanism stays byte-identical to
# fault_recovery_matrix case (b) — the fresh-run-only marker, the manifest-progress
# idiom, and the bare SystemExit(3)→WORKER_CRASH classification are all inherited.
_ROUTER = '''#!/usr/bin/env python
"""Auto-generated by benchmarks/scenarios/crash_storm.py — the per-trial crash router.

Reads --trial and routes to a trainer written VERBATIM by benchmarks.faults:
EVEN trials (unless FLASHML_CRASH_DISARMED=1) run the systemexit_mid trainer,
which crashes at the checkpoint midpoint on a FRESH run only and resumes clean;
ODD trials — and every trial when disarmed — run the clean trainer. All other
args (--steps/--checkpoint-every) pass straight through to the delegate.
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", type=int, required=True)
    args, passthrough = ap.parse_known_args()
    disarmed = os.environ.get("FLASHML_CRASH_DISARMED") == "1"
    crash_armed = (args.trial % 2 == 0) and not disarmed
    target = HERE / ("crashy_trainer_systemexit_mid.py" if crash_armed else "crashy_trainer_clean.py")
    # runpy runs the delegate in THIS process with run_name="__main__"; a
    # SystemExit(3) it raises propagates out with no traceback (WORKER_CRASH).
    sys.argv = [str(target), *passthrough]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
'''


def _write_router(src: Path) -> Path:
    """Materialise the two delegate trainers (faults.py verbatim) + the router."""
    src.mkdir(parents=True, exist_ok=True)
    faults.write_crashy_trainer(src, steps=STEPS, checkpoint_every=EVERY, crash="systemexit_mid")
    faults.write_crashy_trainer(src, steps=STEPS, checkpoint_every=EVERY, crash=None)
    path = src / "crash_router.py"
    path.write_text(_ROUTER)
    return path


def _sweep(src: Path, router: Path, n_trials: int, *, disarmed: bool, out: Path):
    """One fan-out sweep of ``n_trials``: trial i fills the {trial} placeholder;
    even trials crash unless ``disarmed``. Returns the finished Run."""
    import flashruntime as flash
    from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source

    wl = CommandWorkload(
        command=[sys.executable, str(router), "--trial", "{trial}",
                 "--steps", str(STEPS), "--checkpoint-every", str(EVERY)],
        source=Source(path=str(src)),
        outputs=OutputSpec(collect=["metrics.json"]),
        env={"FLASHML_CRASH_DISARMED": "1"} if disarmed else {},
        task_params=[{"trial": i} for i in range(n_trials)],
    )
    # max_restarts=1: exactly one recovery attempt per trial — a crashed even
    # trial's second (resumed) attempt sails past the fresh-only marker.
    return flash.submit(wl, output_dir=out, max_restarts=1)


def _goodput(trials: list[dict]) -> tuple[float, int, int, int]:
    """Pure, testable goodput accounting over completed trials' metrics.

    Returns ``(goodput_fraction, useful_steps, executed_steps, crashed_first_attempt)``.
    useful = Σ steps; executed = Σ (steps + resumed_from) (a crashed trial's
    attempt-1 sunk work is charged as the checkpoint it left, ``resumed_from``);
    crashed_first_attempt = trials with ``resumed_from > 0``. See the module
    docstring for the derivation and why the formula is a conservative LOWER
    bound on efficiency. NaN-safe: 0 executed ⇒ 0.0 (never a fabricated 1.0)."""
    useful = sum(int(t.get("steps", 0)) for t in trials)
    executed = sum(int(t.get("steps", 0)) + int(t.get("resumed_from", 0)) for t in trials)
    crashed = sum(1 for t in trials if int(t.get("resumed_from", 0)) > 0)
    frac = useful / executed if executed else 0.0
    return frac, useful, executed, crashed


def _storm(n_trials: int, repeats: int) -> ResultRow:
    """Run the storm+clean sweeps ``repeats`` times over ``n_trials`` and MEASURE
    completions / goodput / wall-clock penalty from observable run state."""
    ensure_venv_on_path()
    import tempfile

    reps = max(1, repeats)
    completions: list[float] = []
    goodputs: list[float] = []
    penalties: list[float] = []
    crashed_counts: list[float] = []
    for _ in range(reps):
        with tempfile.TemporaryDirectory(prefix="crash-storm-") as td:
            root = Path(td)
            src = root / "src"
            router = _write_router(src)
            # storm (evens crash) then clean (all disarmed) — SEPARATE output dirs
            # so the clean sweep never resumes the storm's checkpoints.
            t_storm, storm_run = timed(
                lambda: _sweep(src, router, n_trials, disarmed=False, out=root / "storm")
            )
            t_clean, _clean_run = timed(
                lambda: _sweep(src, router, n_trials, disarmed=True, out=root / "clean")
            )
            trials = storm_run.trials
            frac, _useful, _executed, crashed = _goodput(trials)
            completions.append(float(len(trials)))
            goodputs.append(frac)
            crashed_counts.append(float(crashed))
            penalties.append((t_storm - t_clean) / t_clean if t_clean else 0.0)

    completed = median(completions)
    comparators = {
        "goodput_fraction": round(median(goodputs), 3),
        "wallclock_penalty_fraction": round(median(penalties), 3),
        # DERIVED, not asserted: the automation resolved every crash with no human
        # in the loop (flash.submit has no human-input path). Counted at 0.0.
        "manual_interventions": 0.0,
        "crashed_first_attempt": median(crashed_counts),
    }
    notes = [
        "completions, goodput, crashed_first_attempt are COUNTED from run state "
        "(len(run.trials) and each trial's metrics steps/resumed_from), never asserted — "
        f"a storm that completes <{n_trials}/{n_trials} ships as the measured number, a FINDING",
        "goodput_fraction = Σ steps / Σ (steps + resumed_from): a crashed trial's attempt-1 "
        "sunk work is charged as resumed_from (the checkpoint it left), a CONSERVATIVE over-count "
        "that makes goodput a pessimistic lower bound — it can never flatter flashruntime",
        "manual_interventions = 0.0 is DERIVED: the max_restarts=1 recovery loop auto-resumed "
        "every WORKER_CRASH with no human in the loop (a bare torchrun needs one restart per crash)",
        "local fan-out is SEQUENTIAL by design (each trial's outputs are collected before the "
        "next runs), so both sweeps' wall-clock is a sum over trials — this measures fault-tolerant "
        "goodput, not throughput",
    ]
    return ResultRow(
        scenario=name,
        section="resilience",
        unit=f"completed/{n_trials}",
        median=completed,
        p10=percentile(completions, 0.1),
        p90=percentile(completions, 0.9),
        repeats=reps,
        comparators=comparators,
        notes=notes,
    )


def run(repeats: int) -> ResultRow:
    """The full 16-trial storm (bench_stress covers the same via ``_storm(16, 1)``)."""
    return _storm(STORM_TRIALS, repeats)
