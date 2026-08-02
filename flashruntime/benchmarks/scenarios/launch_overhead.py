"""launch_overhead — what does flash.submit cost over a bare torchrun?

HYPOTHESIS: flash.submit(ddp(...)) adds well under a second of wall-clock over
the identical bare `torchrun` launch — the fault-tolerance wrapper is cheap.

MEASUREMENT METHOD (auditable from this file alone):
  For each repeat we time two launches of the SAME 1-step DDP job:
    (1) flash.submit(pytorch.ddp("train.py", steps=1)) end to end, and
    (2) an identical `torchrun` invocation — the SAME argv the ddp() helper
        builds, same cwd — run via subprocess.run.
  Both children get the SAME FLASHML_CKPT_DIR / FLASHML_OUTPUT_DIR (a fresh temp
  per launch) so the ONLY difference measured is flash.submit's own machinery
  (Run bookkeeping, launcher wrapper, output collection) vs a raw process spawn.
  overhead = flash_time - torchrun_time, paired per repeat; we report the
  median/p10/p90 of the paired overhead. The two medians are in `comparators`.

  Isolation note: torchrun's PATH and the child's FLASHML_* vars are set
  EXPLICITLY (bench_env / a temp dir) — never inherited — because torchrun lives
  in the venv bin and an inherited PATH silently can't find it, and a shared
  checkpoint dir would let one launch resume another's state.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from benchmarks._util import (
    EXAMPLES,
    ScenarioUnavailable,
    bench_env,
    ensure_venv_on_path,
    median,
    percentile,
    time_subprocess,
    timed,
)
from benchmarks.schema import ResultRow

name = "launch_overhead"
hypothesis = "flash.submit adds under a second of wall-clock over a bare torchrun launch."

_SOURCE = EXAMPLES / "user_pytorch"
_SCRIPT_ARGS = "--steps 1"  # one optimizer step: isolate launch cost, not training


def _require() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ScenarioUnavailable("torch not installed") from exc
    if shutil.which("torchrun") is None:  # pragma: no cover - env dependent
        raise ScenarioUnavailable("torchrun not on PATH")


def run(repeats: int) -> ResultRow:
    _require()
    ensure_venv_on_path()  # so flash.submit's launcher (inherits os.environ) finds torchrun
    import flashruntime as flash
    from flashruntime.integrations import pytorch as fr_torch

    workload = fr_torch.ddp("train.py", source=str(_SOURCE), nproc_per_node=2, script_args=_SCRIPT_ARGS)
    bare_argv = list(workload.command)  # identical argv to what the launcher runs

    flash_times: list[float] = []
    torchrun_times: list[float] = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as td:
            elapsed, run_obj = timed(
                lambda: flash.submit(workload, output_dir=Path(td) / "flash")
            )
            if run_obj.state.value != "SUCCEEDED":
                raise RuntimeError(f"flash.submit failed:\n{run_obj.logs()}")
            flash_times.append(elapsed)

            ckpt = Path(td) / "bare"
            env = bench_env(FLASHML_CKPT_DIR=str(ckpt / "ckpt"), FLASHML_OUTPUT_DIR=str(ckpt))
            torchrun_times.append(time_subprocess(bare_argv, cwd=_SOURCE, env=env))

    overheads = [f - t for f, t in zip(flash_times, torchrun_times)]
    return ResultRow(
        scenario=name,
        unit="seconds",
        median=round(median(overheads), 4),
        p10=round(percentile(overheads, 0.1), 4),
        p90=round(percentile(overheads, 0.9), 4),
        repeats=repeats,
        comparators={
            "flash_submit_s": round(median(flash_times), 4),
            "bare_torchrun_s": round(median(torchrun_times), 4),
        },
        notes=[
            "overhead = flash.submit wall-clock minus identical bare torchrun, paired per repeat",
            "both launches share the same FLASHML_CKPT_DIR/OUTPUT_DIR isolation (a fresh temp)",
        ],
    )
