"""hpo_sweep — flashruntime vs a hand-rolled loop vs ray.tune, on 8 trials.

HYPOTHESIS: flashruntime runs an 8-trial sklearn sweep with a fraction of the
setup code of a hand-rolled loop or ray.tune, at comparable local wall-clock —
its value is orchestration + result collection + fault-tolerance, not (locally)
raw speed, since it runs trials sequentially.

MEASUREMENT METHOD (auditable from this file alone):
  Grid = model×C×n_estimators = 2×2×2 = 8 trials of examples/user_sklearn.
    * flashruntime: flash.submit(sklearn.hpo(grid)) end to end — wall-clock.
    * sequential:   a plain for-loop of subprocess.run over the same 8 combos.
    * ray.tune:     LOC-only — wall-clock is NOT measured (ray is a heavy,
                    throwaway-venv dependency we don't install into the project
                    venv); its minimal setup is counted from the committed,
                    cited snippet ray_tune_hpo.py. Choosing to measure the
                    sequential baseline for real and skip the ray timing is the
                    honest trade the brief allows.
  Peak child RSS via resource.getrusage(RUSAGE_CHILDREN) (a shared high-water
  mark across all spawned trials). setup_loc counted from snippets/*.py.
"""

from __future__ import annotations

import itertools
import subprocess
import sys
import tempfile
from pathlib import Path

from benchmarks._util import (
    EXAMPLES,
    SNIPPETS,
    ScenarioUnavailable,
    bench_env,
    count_loc,
    maxrss_mb,
    median,
    percentile,
    timed,
)
from benchmarks.schema import ResultRow

name = "hpo_sweep"
hypothesis = "flashruntime runs an 8-trial sweep with far less setup code than a hand-rolled loop or ray.tune, at comparable local wall-clock."

_SOURCE = EXAMPLES / "user_sklearn"
_GRID = {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50, 100]}


def _require() -> None:
    try:
        import sklearn  # noqa: F401
    except ImportError as exc:  # pragma: no cover - env dependent
        raise ScenarioUnavailable("scikit-learn not installed") from exc


def _combos() -> list[dict]:
    keys = sorted(_GRID)
    return [dict(zip(keys, c)) for c in itertools.product(*(_GRID[k] for k in keys))]


def _sequential_sweep() -> None:
    env = bench_env()
    for params in _combos():
        argv = [sys.executable, "train.py"]
        for k, v in params.items():
            argv += [f"--{k}", str(v)]
        proc = subprocess.run(argv, cwd=str(_SOURCE), env=env, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"sequential trial {params} failed:\n{proc.stderr}")


def run(repeats: int) -> ResultRow:
    _require()
    import flashruntime as flash
    from flashruntime.integrations import sklearn as fr_sklearn

    flash_times: list[float] = []
    seq_times: list[float] = []
    for _ in range(repeats):
        with tempfile.TemporaryDirectory() as td:
            t_flash, run_obj = timed(
                lambda: flash.submit(
                    fr_sklearn.hpo("train.py", _GRID, source=str(_SOURCE)), output_dir=Path(td)
                )
            )
            if run_obj.state.value != "SUCCEEDED" or len(run_obj.trials) != 8:
                raise RuntimeError(f"flash sweep failed:\n{run_obj.logs()}")
            flash_times.append(t_flash)
            seq_times.append(timed(_sequential_sweep)[0])

    comparators = {
        "sequential_s": round(median(seq_times), 3),
        "peak_child_rss_mb": maxrss_mb(),
        "flash_setup_loc": float(count_loc(SNIPPETS / "flashruntime_hpo.py")),
        "sequential_setup_loc": float(count_loc(SNIPPETS / "sequential_hpo.py")),
        "ray_tune_setup_loc": float(count_loc(SNIPPETS / "ray_tune_hpo.py")),
    }
    notes = [
        "flash.submit runs trials SEQUENTIALLY locally, so wall-clock ≈ the for-loop baseline; "
        "flashruntime's HPO value is orchestration + result collection + fault-tolerance, not local parallelism",
        "peak RSS is a shared RUSAGE_CHILDREN high-water mark across all trials (largest single child)",
        "ray.tune wall-clock is NOT measured here (ray not installed — a heavy, throwaway-venv "
        "comparator); its setup LOC is counted from the committed, cited snippet ray_tune_hpo.py",
    ]
    return ResultRow(
        scenario=name,
        unit="seconds",
        median=round(median(flash_times), 3),
        p10=round(percentile(flash_times, 0.1), 3),
        p90=round(percentile(flash_times, 0.9), 3),
        repeats=repeats,
        comparators=comparators,
        notes=notes,
    )
