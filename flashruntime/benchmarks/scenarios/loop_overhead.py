"""loop_overhead — what does the ft training loop cost per step and per checkpoint?

HYPOTHESIS: flashruntime.torch adds negligible per-step overhead, and a
checkpoint costs single-digit-to-low-tens of milliseconds — cheap enough to
take often.

MEASUREMENT METHOD (auditable from this file alone):
  Three single-process CPU runs, each timed by subprocess wall-clock, isolated
  with a fresh FLASHML_CKPT_DIR/OUTPUT_DIR (no cross-run resume, no pollution):
    A. examples/user_pytorch  --steps 200, periodic checkpoints DISABLED
    B. examples/user_pytorch  --steps 200 --checkpoint-every 50  (4 checkpoints)
    C. examples/user_pytorch_vanilla  (plain torch, no flashruntime import)
  per-checkpoint cost = (B - A) / (200 // 50) — startup+loop cancel in the delta,
  leaving only the marginal cost of a checkpoint write. This is the clean number.
  The ft-vs-vanilla ratio (A vs C, per wall-step) is reported too, with the
  honest caveat that at 200/100 steps of a tiny model interpreter+import startup
  dominates wall-clock and the two scripts do different per-step work — so the
  ratio bounds loop overhead loosely, it is not identical-work.

  "Disabled" uses a sentinel checkpoint-every > steps, NOT 0: ft.checkpoint()
  computes `step % every`, so every=0 raises ZeroDivisionError (a real edge in
  the example) — a sentinel skips every periodic write instead.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

from benchmarks._util import (
    EXAMPLES,
    ScenarioUnavailable,
    bench_env,
    median,
    percentile,
    time_subprocess,
)
from benchmarks.schema import ResultRow

name = "loop_overhead"
hypothesis = "The ft training loop adds negligible per-step cost; a checkpoint is a few to low-tens of ms."

_STEPS = 200
_EVERY = 50
_DISABLED = _STEPS + 1  # > steps ⇒ no periodic checkpoint (every=0 would divide by zero)
_FT = EXAMPLES / "user_pytorch"
_VANILLA = EXAMPLES / "user_pytorch_vanilla"


def _require() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ScenarioUnavailable("torch not installed") from exc


def _time_python(script_dir: Path, args: list[str]) -> float:
    with tempfile.TemporaryDirectory() as td:
        env = bench_env(FLASHML_CKPT_DIR=f"{td}/ckpt", FLASHML_OUTPUT_DIR=td)
        return time_subprocess([sys.executable, "train.py", *args], cwd=script_dir, env=env)


def run(repeats: int) -> ResultRow:
    _require()
    per_ckpt_ms: list[float] = []
    ratios: list[float] = []
    ft_sps: list[float] = []
    van_sps: list[float] = []
    for _ in range(repeats):
        a = _time_python(_FT, ["--steps", str(_STEPS), "--checkpoint-every", str(_DISABLED)])
        b = _time_python(_FT, ["--steps", str(_STEPS), "--checkpoint-every", str(_EVERY)])
        c = _time_python(_VANILLA, [])
        per_ckpt_ms.append((b - a) / (_STEPS // _EVERY) * 1000.0)
        ft_sps.append(_STEPS / a)
        van_sps.append(100 / c)  # the vanilla script runs a fixed 100 iterations
        ratios.append((a / _STEPS) / (c / 100))  # ft wall/step ÷ vanilla wall/step

    return ResultRow(
        scenario=name,
        unit="ms/checkpoint",
        median=round(median(per_ckpt_ms), 3),
        p10=round(percentile(per_ckpt_ms, 0.1), 3),
        p90=round(percentile(per_ckpt_ms, 0.9), 3),
        repeats=repeats,
        comparators={
            "ft_steps_per_s": round(median(ft_sps), 1),
            "vanilla_steps_per_s": round(median(van_sps), 1),
            "ft_vs_vanilla_wall_per_step_ratio": round(median(ratios), 3),
        },
        notes=[
            "per-checkpoint = (ckpt=50 run - checkpoint-disabled run) / 4; startup+loop cancel in the delta",
            "a per-checkpoint figure at or below zero means the write cost is BELOW the run-to-run "
            "noise floor for this tiny model (a few-KB state dict) — checkpoints are effectively free "
            "here; a larger model would surface a positive cost",
            "steps/sec and the ratio are startup-dominated at this step count and compare "
            "DIFFERENT scripts (ft MLP vs vanilla Linear) — indicative, not identical-work",
            "checkpoint-every=0 raises ZeroDivisionError in the example; a sentinel > steps disables instead",
        ],
    )
