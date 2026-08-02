# Tutorial: parallel scikit-learn sweeps

scikit-learn work is *embarrassingly parallel across runs* — a grid of
independent fits, never a single `.fit()` you split internally. FlashRuntime
fans a grid out into one independent task per trial, runs them, and ranks the
results, while your script stays plain sklearn with no FlashRuntime import.

This tutorial builds a sweep from an ordinary script. You need
`pip install "flashruntime[sklearn]"` (numpy + scikit-learn) — see
[Get started](../get-started.md).

---

## 1. A plain sklearn script (flags in, `metrics.json` out)

The only contract FlashRuntime asks: read hyperparameters from CLI flags, write
a flat `metrics.json`. No FlashRuntime import anywhere.

```python
import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=50)
    args = parser.parse_args()

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=600, n_features=12, random_state=0)
    if args.model == "logreg":
        estimator = LogisticRegression(C=args.C, max_iter=500, random_state=0)
    elif args.model == "rf":
        estimator = RandomForestClassifier(n_estimators=args.n_estimators, random_state=0)
    else:
        raise SystemExit(f"unknown model {args.model!r} (logreg|rf)")

    scores = cross_val_score(estimator, X, y, cv=3)
    metrics = {
        "model": args.model,
        "C": args.C,
        "n_estimators": args.n_estimators,
        "accuracy_mean": round(float(scores.mean()), 4),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
```

Run one trial by hand to confirm it works:

```bash
python train.py --model rf --n_estimators 100
```

---

## 2. Fan a grid out with `fr_sklearn.hpo(...)`

The `integrations.sklearn` adapter builds the workload from that script. Give
it a grid; it expands the Cartesian product into one task per trial:

```python
import flashruntime as flash
from flashruntime.integrations import sklearn as fr_sklearn

run = flash.submit(fr_sklearn.hpo(
    "train.py",
    {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50]},
    source=".",
))
print(f"state={run.state.value}  trials={len(run.trials)}")
print("best:", run.best_trial())
```

`{"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50]}` expands to
`2 × 2 × 1 = 4` trials. Each `{placeholder}` in the built command is filled
from the trial's params, so one task receives `--model rf --C 1.0
--n_estimators 50` and so on. Every trial's `metrics.json` is collected and
recorded on `run.trials`, with its params merged in.

---

## 3. Read the winner

Because `hpo` (via `sweep`) sets `outputs.primary_metric="accuracy_mean"`,
`run.best_trial()` needs no arguments — it returns the trial with the highest
`accuracy_mean`:

```python
best = run.best_trial()                 # ranks by accuracy_mean (maximize=True)
worst = run.best_trial(maximize=False)  # or flip it
by_other = run.best_trial(metric="C")   # or rank by any reported key
```

`best_trial(metric=None, maximize=None)` falls back to the `OutputSpec`
defaults the adapter set; pass `metric=` / `maximize=` to override. It returns
`None` if no trial reported the metric.

---

## How the fan-out stays correct

- **Sequential and isolated.** `flash.submit()` runs one trial at a time and
  copies each trial's `metrics.json` out **before** the next trial can
  overwrite it — so a trial's outputs are always its own.
- **Independent checkpoint trees.** Each trial gets its own job-scoped
  checkpoint tree, so trials never cross-contaminate. (This matters for
  checkpointed workloads; a pure sklearn fit has none.)
- **Two API shapes.** `hpo(script, grid, **kwargs)` is grid sugar over
  `sweep(script, task_params, *, source=".", metric="accuracy_mean",
  maximize=True, python="python")`. Use `sweep` directly to pass an explicit
  list of param dicts (e.g. a hand-picked, non-Cartesian set).

Add a restart budget the same way as any run: `flash.submit(..., max_restarts=1)`
retries a *transient* trial failure and fails fast on a deterministic one — see
the [fault-tolerance tutorial](fault-tolerance.md).

---

## Where to go next

- **[scikit-learn guide](../guides/sklearn.md)** — the adapter reference and
  the "distribute across runs, never inside `.fit()`" rule.
- **[ConvNet tutorial](convnet.md)** — the PyTorch DDP + checkpoint-resume
  story.
- **[SDK reference](../reference/sdk.md)** — every `Run` attribute and
  `submit()` argument.
