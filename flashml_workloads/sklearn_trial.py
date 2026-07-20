"""One hyperparameter-search trial, runnable as a leased task.

Executor contract (Tier-1 subprocess runner):

    python -m flashml_workloads.sklearn_trial --spec spec.json --out OUTDIR

`spec.json` (written by the executor from the lease payload):
    {
      "task_id": "trial-003",
      "params":  {"model": "logreg", "C": 0.1},          # this trial's config
      "inputs":  {"dataset": "/abs/path/to/dataset.csv"} # downloaded shared data
    }

The dataset is a headerless CSV, features then label in the last column —
shared data hosted by the coordinator's local artifact endpoint, downloaded
once per task. Output: `OUTDIR/metrics.json` with the cross-validated
accuracy; the executor uploads it and commits its sha256.

Supported params: model ∈ {logreg, rf}; logreg: C; rf: n_estimators,
max_depth. Deterministic (fixed random_state) so retried attempts produce
identical results — which is what makes idempotent commit meaningful.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path


def run_trial(spec: dict) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    params = spec.get("params", {})
    dataset_path = spec["inputs"]["dataset"]

    features: list[list[float]] = []
    labels: list[float] = []
    with open(dataset_path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            features.append([float(x) for x in row[:-1]])
            labels.append(float(row[-1]))

    model_kind = params.get("model", "logreg")
    if model_kind == "logreg":
        model = LogisticRegression(C=float(params.get("C", 1.0)), max_iter=500, random_state=0)
    elif model_kind == "rf":
        model = RandomForestClassifier(
            n_estimators=int(params.get("n_estimators", 50)),
            max_depth=int(params["max_depth"]) if params.get("max_depth") else None,
            random_state=0,
        )
    else:
        raise ValueError(f"unsupported model {model_kind!r} (logreg|rf)")

    start = time.monotonic()
    scores = cross_val_score(model, features, labels, cv=3)
    return {
        "task_id": spec.get("task_id", ""),
        "params": params,
        "accuracy_mean": round(float(sum(scores) / len(scores)), 4),
        "accuracy_folds": [round(float(s), 4) for s in scores],
        "n_samples": len(labels),
        "train_seconds": round(time.monotonic() - start, 3),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sklearn_trial")
    parser.add_argument("--spec", required=True, help="path to spec.json")
    parser.add_argument("--out", required=True, help="output directory")
    args = parser.parse_args(argv)

    spec = json.loads(Path(args.spec).read_text())
    metrics = run_trial(spec)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
