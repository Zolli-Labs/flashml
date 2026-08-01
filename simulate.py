"""Run the whole federated loop locally, before spending anyone's laptop on it.

Submitting a repo and watching five rounds crawl across volunteer machines is
a slow way to discover that round 0 wrote the wrong shape. This runs the same
train.py the platform runs, shards and rounds and all, in a temp directory —
and averages the deltas with FlashML's own reduce_deltas rather than a
reimplementation, so if the encoding is wrong it fails here rather than
looking fine here and failing there.

    python simulate.py                 # 5 rounds x 3 shards, as flashml.yaml
    python simulate.py --rounds 3

Requires flashruntime (`pip install "flashruntime @ git+https://github.com/
Zolli-Labs/flashruntime"`). Not needed to submit the job — only to rehearse
it.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from flashml_workloads.fedavg_weights import apply_delta, reduce_deltas
except ImportError:  # pragma: no cover - guidance, not logic
    sys.exit(
        "simulate.py needs flashruntime:\n"
        '  pip install "flashruntime @ git+https://github.com/Zolli-Labs/flashruntime"'
    )

HERE = Path(__file__).resolve().parent


def run_round(root: Path, round_index: int, shards: int, weights: dict | None):
    """One round: every shard trains, then the platform averages."""
    contributions: list[tuple[dict, int]] = []
    losses: list[float] = []

    for shard in range(shards):
        work = root / f"r{round_index}-s{shard}"
        (work / "inputs").mkdir(parents=True, exist_ok=True)
        (work / "out").mkdir(parents=True, exist_ok=True)
        # Round 0 gets no weights.json at all — the absence is the signal,
        # so writing an empty file here would test the wrong thing.
        if weights is not None:
            (work / "inputs" / "weights.json").write_text(json.dumps(weights))

        proc = subprocess.run(
            [
                sys.executable, str(HERE / "train.py"),
                "--round", str(round_index),
                "--num-shards", str(shards),
                "--shard", str(shard),
            ],
            env={**dict(__import__("os").environ), "FLASHML_WORK_DIR": str(work)},
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            print(proc.stdout)
            print(proc.stderr, file=sys.stderr)
            sys.exit(f"shard {shard} failed on round {round_index}")

        delta = json.loads((work / "out" / "delta.json").read_text())
        metrics = json.loads((work / "out" / "metrics.json").read_text())
        # The platform reads exactly these two fields; assert them here so a
        # missing one is a clear failure rather than a mysterious zero
        # weighting in the average.
        assert isinstance(metrics.get("samples"), int) and metrics["samples"] > 0, metrics
        assert isinstance(metrics.get("loss"), (int, float)), metrics
        contributions.append((delta, metrics["samples"]))
        losses.append(float(metrics["loss"]))

    averaged = reduce_deltas(contributions)
    # Round 0's "deltas" are whole models, so there is nothing to add them
    # to; from round 1 they are changes applied to the round's base.
    new_weights = averaged if weights is None else apply_delta(weights, averaged)
    return new_weights, losses


def main() -> int:
    cfg = {}
    yaml_path = HERE / "flashml.yaml"
    for line in yaml_path.read_text().splitlines():
        line = line.strip()
        for key in ("rounds", "shards"):
            if line.startswith(f"{key}:"):
                cfg[key] = int(line.split(":", 1)[1].split("#")[0].strip())

    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=cfg.get("rounds", 5))
    parser.add_argument("--shards", type=int, default=cfg.get("shards", 3))
    args = parser.parse_args()

    root = Path(tempfile.mkdtemp(prefix="flashml-sim-"))
    print(f"simulating {args.rounds} rounds x {args.shards} shards in {root}\n")

    weights = None
    try:
        for r in range(args.rounds):
            weights, losses = run_round(root, r, args.shards, weights)
            mean = sum(losses) / len(losses)
            spread = " ".join(f"{l:.4f}" for l in losses)
            print(f"round {r}: mean loss {mean:.4f}   (shards: {spread})")
    finally:
        shutil.rmtree(root, ignore_errors=True)

    print("\nIf the mean loss falls across rounds, the averaging is working.")
    print("If it is flat, every shard is training but nothing is being")
    print("combined — check that delta.json is the CHANGE from round 1 on.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
