"""Checkpointable logistic-regression SGD, runnable as a leased task.

Executor contract (same CLI as the other task modules):

    python -m flashml_workloads.sgd_trainer --spec spec.json --out OUTDIR

`spec.json`:
    params: steps, checkpoint_every, lr, seed,
            kill_at_step (optional — fires only on a FRESH start, so a
            resumed retry never re-crashes; this is the deterministic
            kill-and-recover test hook)
    inputs: dataset (headerless CSV, label last), resume (optional path to
            a checkpoint json downloaded by the executor)

Checkpoints go to OUTDIR/ckpt/step-NNNNNN.json ({step, weights, bias}); the
executor's relay uploads them and commits manifests as they appear.

Determinism is the contract: batches are indexed by step number (cyclic
slices, no RNG state to carry), so resuming from step k reproduces the
uninterrupted run bit-for-bit — recovery must never silently change the
result. Pure stdlib; runs anywhere, including `--network none` containers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

BATCH = 4


def _load(path: str) -> tuple[list[list[float]], list[float]]:
    xs, ys = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if row:
                xs.append([float(v) for v in row[:-1]])
                ys.append(float(row[-1]))
    return xs, ys


def _loss(xs, ys, w, b) -> float:
    total = 0.0
    for x, y in zip(xs, ys):
        z = sum(wi * xi for wi, xi in zip(w, x)) + b
        p = 1.0 / (1.0 + math.exp(-z))
        p = min(max(p, 1e-12), 1 - 1e-12)
        total += -(y * math.log(p) + (1 - y) * math.log(1 - p))
    return total / len(xs)


def run_trainer(spec: dict, out: Path) -> dict:
    params = spec.get("params", {})
    steps = int(params["steps"])
    every = int(params.get("checkpoint_every", 100))
    lr = float(params.get("lr", 0.1))
    kill_at = params.get("kill_at_step")

    xs, ys = _load(spec["inputs"]["dataset"])
    dims = len(xs[0])

    resume_path = spec.get("inputs", {}).get("resume")
    if resume_path:
        ckpt = json.loads(Path(resume_path).read_text())
        w, b, start = list(ckpt["weights"]), float(ckpt["bias"]), int(ckpt["step"])
        resumed = True
    else:
        w, b, start = [0.0] * dims, 0.0, 0
        resumed = False

    out = Path(out)
    ckpt_dir = out / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    n = len(xs)
    for step in range(start + 1, steps + 1):
        # batch indexed by step ⇒ no RNG state; resume is trivially exact
        base = ((step - 1) * BATCH) % n
        idx = [(base + i) % n for i in range(BATCH)]
        gw, gb = [0.0] * dims, 0.0
        for i in idx:
            z = sum(wi * xi for wi, xi in zip(w, xs[i])) + b
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - ys[i]
            for d in range(dims):
                gw[d] += err * xs[i][d]
            gb += err
        for d in range(dims):
            w[d] -= lr * gw[d] / BATCH
        b -= lr * gb / BATCH

        if step % every == 0 and step < steps:
            (ckpt_dir / f"step-{step:06d}.json").write_text(
                json.dumps({"step": step, "weights": w, "bias": b})
            )
        if kill_at is not None and not resumed and step >= int(kill_at):
            # simulated crash on a fresh run only — retries resume, not re-die
            sys.exit(3)

    metrics = {
        "task_id": spec.get("task_id", ""),
        "steps": steps,
        "started_from_step": start,
        "resumed": resumed,
        "final_loss": round(_loss(xs, ys, w, b), 12),
        "weights": [round(v, 12) for v in w],
        "bias": round(b, 12),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, sort_keys=True))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sgd_trainer")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    run_trainer(json.loads(Path(args.spec).read_text()), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
