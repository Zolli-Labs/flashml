"""sklearn adapter: distribute across runs, never inside .fit().

The contract with the user's script is pure convention: CLI flags in,
metrics.json out. No sklearn import here — the script owns the estimator.
"""

from __future__ import annotations

import itertools

from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source


def sweep(
    script: str,
    task_params: list[dict],
    *,
    source: str = ".",
    metric: str = "accuracy_mean",
    maximize: bool = True,
    python: str = "python",
) -> CommandWorkload:
    """One independent task per params dict. Every dict must carry every
    key (the CLI flags are built from the union)."""
    keys = sorted({k for p in task_params for k in p})
    command = [python, script]
    for key in keys:
        command += [f"--{key}", "{" + key + "}"]
    return CommandWorkload(
        command=command,
        source=Source(path=source),
        task_params=task_params,
        mode="independent_tasks",
        outputs=OutputSpec(collect=["metrics.json"], primary_metric=metric, maximize=maximize),
    )


def hpo(script: str, grid: dict[str, list], **kwargs) -> CommandWorkload:
    """Cartesian grid search: {"model": ["logreg","rf"], "C": [0.1, 1]} → 4 trials."""
    keys = sorted(grid)
    trials = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    return sweep(script, trials, **kwargs)
