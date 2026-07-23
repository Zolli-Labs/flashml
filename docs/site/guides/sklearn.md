# scikit-learn guide

FlashRuntime **operates** your scikit-learn job — it never rewrites your
estimator. You keep the model and the scoring; FlashRuntime fans a grid out
into independent tasks, runs them, collects each `metrics.json`, and ranks the
results.

The rule that shapes this whole adapter: sklearn work is *embarrassingly
parallel across runs*, **never inside a single `.fit()`**. FlashRuntime fans a
grid into one independent task per trial — it never tries to split one `.fit()`
call, which would change the math.

For a worked walkthrough, do the
[sklearn sweeps tutorial](../tutorials/sklearn-sweeps.md).

---

## The contract: flags in, `metrics.json` out

Your script needs **zero FlashRuntime imports**. It reads hyperparameters from
CLI flags and writes a flat `metrics.json` to its working directory. That is
the entire contract — the same one every framework uses.
`examples/user_sklearn/train.py` is plain sklearn end to end.

---

## Fan a grid out

The `integrations.sklearn` adapter builds the workload from that script:

```python
import flashruntime as flash
from flashruntime.integrations import sklearn as fr_sklearn

run = flash.submit(fr_sklearn.hpo(
    "train.py",
    {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50]},
    source="examples/user_sklearn",
))
print(f"state={run.state.value}  trials={len(run.trials)}")
print("best:", run.best_trial())     # ranks by outputs.primary_metric
```

- **`hpo(script, grid, **kwargs)`** expands a Cartesian grid
  (`{"model": ["logreg", "rf"], "C": [0.1, 1]}` → 4 trials) and delegates to
  `sweep`.
- **`sweep(script, task_params, *, source=".", metric="accuracy_mean",
  maximize=True, python="python")`** takes an explicit list of param dicts —
  use it when you want a hand-picked, non-Cartesian set.

Each `{placeholder}` in the built command is filled from the trial's params, so
`train.py` receives `--model rf --C 1.0` and friends. Because `sweep` sets
`outputs.primary_metric=metric`, `run.best_trial()` needs no arguments — it
returns the trial with the highest `accuracy_mean` (or lowest, when
`maximize=False`).

---

## Why the fan-out is correct by construction

- **Sequential and isolated.** `flash.submit()` runs one trial at a time and
  copies each trial's `metrics.json` out **before** the next trial can
  overwrite it.
- **Independent trees.** Each trial gets its own job-scoped checkpoint tree, so
  trials never cross-contaminate.
- **Add fault tolerance the usual way.** `flash.submit(..., max_restarts=1)`
  retries a transient trial failure and fails fast on a deterministic one (a
  bad flag combination that raises the same error every time). See the
  [fault-tolerance tutorial](../tutorials/fault-tolerance.md).

---

## Adding another framework

The sklearn adapter is a ~40-line function that builds a `CommandWorkload` with
`task_params` set for fan-out. A new framework adapter follows the same
pattern: a small function under `flashruntime/integrations/` that returns a
`CommandWorkload` describing *what to run*, then reuses the same
launch/collect/rank machinery. The
[PyTorch adapter](pytorch.md#adding-another-framework) is the coordinated-run
counterpart, and [Hugging Face](huggingface.md) is a thin wrapper over it — no
core change is needed to teach FlashRuntime a new framework.
