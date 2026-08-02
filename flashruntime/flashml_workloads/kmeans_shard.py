"""One K-means map step over one data shard, runnable as a leased task.

Executor contract (same as sklearn_trial):

    python -m flashml_workloads.kmeans_shard --spec spec.json --out OUTDIR

`spec.json`: {"params": {"centroids": [[…], …]}, "inputs": {"shard": path}}
The shard is a headerless CSV of points. Output `metrics.json` carries the
*partial* statistics — per-centroid coordinate sums, counts, and inertia —
which the driver reduces into the next round's centroids. Pure stdlib:
assignment is just nearest-squared-distance; no numpy required on devices.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def run_shard(spec: dict) -> dict:
    centroids: list[list[float]] = spec["params"]["centroids"]
    k = len(centroids)
    dims = len(centroids[0])
    sums = [[0.0] * dims for _ in range(k)]
    counts = [0] * k
    inertia = 0.0
    n = 0

    with open(spec["inputs"]["shard"], newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            point = [float(x) for x in row]
            best, best_d = 0, float("inf")
            for j, c in enumerate(centroids):
                d = sum((p - q) ** 2 for p, q in zip(point, c))
                if d < best_d:
                    best, best_d = j, d
            for dim in range(dims):
                sums[best][dim] += point[dim]
            counts[best] += 1
            inertia += best_d
            n += 1

    return {
        "task_id": spec.get("task_id", ""),
        "sums": [[round(v, 10) for v in s] for s in sums],
        "counts": counts,
        "inertia": inertia,
        "n": n,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kmeans_shard")
    parser.add_argument("--spec", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    spec = json.loads(Path(args.spec).read_text())
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(run_shard(spec), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
