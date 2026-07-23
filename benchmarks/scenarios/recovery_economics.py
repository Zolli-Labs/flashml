"""recovery_economics — what does automatic resume actually save on a crash?

HYPOTHESIS: after a crash at 40/80 steps, flashruntime's checkpoint-backed
auto-resume finishes the job in materially less wall-clock than a raw torchrun,
which has no checkpoint and must rerun from step 0 — and the steps between the
last checkpoint and the finish line are never recomputed.

MEASUREMENT METHOD (auditable from this file alone):
  Real 2-process CPU DDP runs (the example crashes itself with --kill-at-step).
  Per repeat we measure three wall-clocks:
    t_crash = submit(crash@40, max_restarts=0)  → FAILED at step 40
    t_auto  = submit(crash@40, max_restarts=1)  → crash, then resume from the
              step-40 checkpoint to 80 (verified: trials[0].resumed_from == 40)
    t_full  = submit(80 steps, no crash)        → a clean full run from 0
  flashruntime cost (a): t_auto.
  raw-torchrun cost (b): you lose the crashed run and rerun the whole thing,
    so b = t_crash + t_full.
  seconds_saved = b - a ; steps_not_recomputed = 40 (the 40→80 tail the resume
  skips). We report the median seconds saved; both totals are in comparators.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from benchmarks._util import (
    EXAMPLES,
    ScenarioUnavailable,
    ensure_venv_on_path,
    median,
    percentile,
    timed,
)
from benchmarks.schema import ResultRow

name = "recovery_economics"
hypothesis = "Auto-resume from a checkpoint finishes a crashed run faster than a raw rerun-from-zero, and never recomputes the steps past the last checkpoint."

_SOURCE = EXAMPLES / "user_pytorch"
_STEPS, _EVERY, _KILL = 80, 20, 40  # resume point (40) is an epoch boundary — see train.py


def _require() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ScenarioUnavailable("torch not installed") from exc
    if shutil.which("torchrun") is None:  # pragma: no cover - env dependent
        raise ScenarioUnavailable("torchrun not on PATH")


def run(repeats: int) -> ResultRow:
    _require()
    ensure_venv_on_path()
    import flashruntime as flash
    from flashruntime.integrations import pytorch as fr_torch

    def ddp(extra: str = "") -> object:
        args = f"--steps {_STEPS} --checkpoint-every {_EVERY} {extra}".strip()
        return fr_torch.ddp("train.py", source=str(_SOURCE), nproc_per_node=2, script_args=args)

    saved: list[float] = []
    autos: list[float] = []
    raws: list[float] = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as td:
            t_crash, crash = timed(
                lambda: flash.submit(ddp(f"--kill-at-step {_KILL}"), output_dir=Path(td) / "crash")
            )
            if crash.state.value != "FAILED":
                raise RuntimeError("expected the crash run to FAIL at step 40")
            t_auto, auto = timed(
                lambda: flash.submit(
                    ddp(f"--kill-at-step {_KILL}"), output_dir=Path(td) / "auto", max_restarts=1
                )
            )
            if auto.state.value != "SUCCEEDED" or auto.trials[0].get("resumed_from") != _KILL:
                raise RuntimeError(f"auto-resume did not resume from {_KILL}:\n{auto.logs()}")
            t_full, full = timed(
                lambda: flash.submit(ddp(), output_dir=Path(td) / "full")
            )
            if full.state.value != "SUCCEEDED":
                raise RuntimeError(f"full run failed:\n{full.logs()}")

            autos.append(t_auto)
            raws.append(t_crash + t_full)
            saved.append((t_crash + t_full) - t_auto)

    return ResultRow(
        scenario=name,
        unit="seconds saved",
        median=round(median(saved), 3),
        p10=round(percentile(saved, 0.1), 3),
        p90=round(percentile(saved, 0.9), 3),
        repeats=repeats,
        comparators={
            "auto_resume_s": round(median(autos), 3),
            "raw_rerun_from_zero_s": round(median(raws), 3),
            "steps_not_recomputed": float(_STEPS - _KILL),
        },
        notes=[
            "raw-torchrun cost modelled as t_crash + t_full (no checkpoint ⇒ rerun the whole job)",
            "auto-resume verified to restart from the step-40 checkpoint (trials[0].resumed_from == 40)",
            "seconds-saved is small here because torchrun startup (~2 s) dominates and the 40 recomputed "
            "steps of this tiny model cost only a fraction of a second; the saving scales with the compute "
            "between the last checkpoint and the crash — negligible at smoke size, hours on a real job",
            "steps_not_recomputed (40) is the size-INDEPENDENT guarantee: resume never re-does work past "
            "the last valid checkpoint, whatever a step costs",
        ],
    )
