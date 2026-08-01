"""One federated-averaging round on one shard, runnable as a leased task.

Executor contract (same as kmeans_shard / sgd_trainer):

    python -m flashml_workloads.fedavg_worker --spec spec.json --out OUTDIR

`spec.json`:
    params: round, shard, num_shards, local_steps, lr, batch_size, seed,
            in_dim, hidden, out_dim, dataset_size
    inputs: weights (optional path to the round's weights JSON; absent on
            round 0, where the seed determines the starting point)

Outputs `OUTDIR/delta.json` (this worker's weight change) and
`OUTDIR/metrics.json` (the commit artifact — only a root-level
metrics.json sets the commit hash).

Why a delta and not the new weights: the driver averages contributions,
and averaging deltas keeps the arithmetic correct when a worker joins a
round late with stale weights — its delta is still a valid direction from
the weights it actually saw.

torch is imported inside functions so this module can be inspected (and
the rest of flashml_workloads used) without torch installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashml_workloads.fedavg_weights import decode, encode, require_finite, subtract


def build_model(seed: int, in_dim: int, hidden: int, out_dim: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
    )


def state_to_blob(model) -> dict:
    state = {}
    for name, t in model.state_dict().items():
        state[name] = (list(t.shape), [float(x) for x in t.flatten().tolist()])
    return encode(state)


def blob_to_state(model, blob: dict) -> None:
    """Load a weight blob into `model`, failing loudly on any mismatch."""
    import torch

    from flashml_workloads.fedavg_weights import WeightShapeMismatch

    # This is the worker's entry point for weights that arrived over the
    # network as a staged input file — the last line of defence on the node
    # side before a NaN/Inf is loaded straight into the torch model. Gate it
    # before anything else so a corrupted or attacker-written weights.json
    # never reaches `load_state_dict`.
    require_finite(blob, "blob_to_state")

    current = state_to_blob(model)
    if current.keys() != blob.keys():
        raise WeightShapeMismatch(
            f"weights name mismatch: expected {sorted(current)}, got {sorted(blob)}"
        )
    new_state = {}
    for name, (shape, data) in decode(blob).items():
        if shape != list(current[name]["shape"]):
            raise WeightShapeMismatch(
                f"parameter {name!r} shape {shape} != {current[name]['shape']}"
            )
        new_state[name] = torch.tensor(data, dtype=torch.float32).reshape(shape)
    model.load_state_dict(new_state)


def _make_shard(params: dict):
    """Deterministic synthetic data, sliced by shard.

    Strided slicing (`x[shard::num_shards]`) rather than contiguous blocks
    so every shard sees the same label distribution — a contiguous split of
    sorted data would give workers disjoint classes and FedAvg would
    diverge for reasons unrelated to the runtime.
    """
    import torch

    g = torch.Generator().manual_seed(params["seed"])
    n, d = params["dataset_size"], params["in_dim"]
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = ((x @ w).squeeze(1) > 0).long()
    shard, num = params["shard"], params["num_shards"]
    return x[shard::num], y[shard::num]


def run_worker(spec: dict, outdir: Path) -> dict:
    import torch
    from torch import nn

    p = spec["params"]
    outdir = Path(outdir)

    model = build_model(p["seed"], p["in_dim"], p["hidden"], p["out_dim"])
    weights_path = (spec.get("inputs") or {}).get("weights")
    if weights_path:
        blob_to_state(model, json.loads(Path(weights_path).read_text()))
    base = state_to_blob(model)

    x, y = _make_shard(p)
    samples = int(x.shape[0])
    if samples == 0:
        raise ValueError(
            f"shard {p['shard']} of num_shards={p['num_shards']} is empty "
            f"for dataset_size={p['dataset_size']}: num_shards must not "
            f"exceed dataset_size"
        )
    opt = torch.optim.SGD(model.parameters(), lr=p["lr"])
    loss_fn = nn.CrossEntropyLoss()
    batch = p["batch_size"]

    # Batches are indexed by step via a wrapped, step-indexed gather (each
    # index individually wrapped mod `samples`, not a single truncating
    # slice) so every step sees exactly `batch` rows even when `batch` does
    # not divide `samples` evenly — same rule as sgd_trainer. Pure function
    # of `step`, no RNG state, so a retried attempt reproduces the same delta.
    last_loss = 0.0
    for step in range(p["local_steps"]):
        idx = [(step * batch + i) % samples for i in range(batch)]
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    delta = subtract(state_to_blob(model), base)
    (outdir / "delta.json").write_text(json.dumps(delta))

    metrics = {
        "round": p["round"],
        "shard": p["shard"],
        "samples": samples,
        "loss": last_loss,
        "local_steps": p["local_steps"],
        "delta_file": "delta.json",
    }
    (outdir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_worker(json.loads(Path(args.spec).read_text()), outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
