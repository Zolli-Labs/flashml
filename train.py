"""Federated training on FlashML — a complete, runnable example.

WHAT THIS IS NOT
----------------
This is not DistributedDataParallel. DDP needs every rank to talk to every
other rank (NCCL/Gloo all-reduce on each backward pass), and FlashML task
containers run with `--network none` — a volunteer lends you a machine, not
an open socket. There are no peers to reach.

Instead the platform runs federated averaging. Each round:

    1. every machine gets the SAME starting weights and a DIFFERENT slice
       of the data
    2. each trains locally for a few epochs, alone
    3. each uploads how much it changed the weights
    4. the platform averages those changes, weighted by how many samples
       each machine actually trained on, and that becomes round N+1's
       starting weights

Weights move once per round rather than once per batch, so it tolerates
laptops on home wifi that close their lids mid-round. It is not a drop-in
replacement for DDP — convergence per round is worse than per-step
synchronisation — but it is what works over the public internet on hardware
nobody is paying for.

THE CONTRACT THIS FILE IMPLEMENTS
---------------------------------
FlashML runs this file once per shard per round, as:

    python /work/inputs/code/train.py --round R --num-shards K --shard N

and requires exactly three things of it:

    read   /work/inputs/weights.json   the round's starting weights.
                                       ABSENT on round 0.
    write  /work/out/delta.json        how the weights changed:
                                       {"<param>": {"shape": [...],
                                                    "data": [...]}}
                                       On round 0, where you were given no
                                       weights, write the trained weights
                                       themselves.
    write  /work/out/metrics.json      at least {"samples": <positive int>,
                                       "loss": <number>}. `samples` is the
                                       weight this machine's contribution
                                       gets in the average, so it must be
                                       the number of samples actually
                                       trained on.

Everything else is yours.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn

# --- the contract's paths --------------------------------------------------
# /work is where FlashML mounts a task's world, and the default is what runs
# in production. FLASHML_WORK_DIR only exists so `simulate.py` can run the
# whole federated loop on a laptop without root — the alternative is testing
# a *copy* of this logic, which is how you ship a demo that has never been
# run. FlashML never sets it.
WORK = Path(os.environ.get("FLASHML_WORK_DIR", "/work"))
WEIGHTS_IN = WORK / "inputs" / "weights.json"   # /work/inputs/weights.json
OUT_DIR = WORK / "out"                          # /work/out
DELTA_OUT = OUT_DIR / "delta.json"
METRICS_OUT = OUT_DIR / "metrics.json"

# --- a small, deterministic problem ---------------------------------------
# Synthetic rather than a real dataset because task containers have no
# network: nothing can be downloaded at runtime, and a dataset large enough
# to be interesting is too large to stage as a repo input. The relationship
# below is genuinely learnable, so a falling loss across rounds means the
# averaging is working rather than that the numbers happen to shrink.
N_SAMPLES = 4096
N_FEATURES = 20
HIDDEN = 32

# The seed that MUST be shared. Every shard has to start round 0 from
# byte-identical weights: averaging N independently-initialised networks
# gives you the mean of N unrelated models, which is worse than any of them
# and looks like "federated learning doesn't work" rather than like a bug.
# Deliberately not derived from --shard.
INIT_SEED = 12345
# Separate seed for the dataset, so the data is identical everywhere and
# only the *slice* differs per shard.
DATA_SEED = 999


def build_model() -> nn.Module:
    torch.manual_seed(INIT_SEED)
    return nn.Sequential(
        nn.Linear(N_FEATURES, HIDDEN),
        nn.ReLU(),
        nn.Linear(HIDDEN, 1),
    )


def build_dataset() -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(DATA_SEED)
    x = torch.randn(N_SAMPLES, N_FEATURES, generator=gen)
    true_w = torch.randn(N_FEATURES, 1, generator=gen)
    y = torch.tanh(x @ true_w) + 0.05 * torch.randn(N_SAMPLES, 1, generator=gen)
    return x, y


def shard_of(x, y, shard: int, num_shards: int):
    """Strided, not contiguous.

    A contiguous split hands each machine a different region of the feature
    space, so every shard trains on a biased sample and the averaged model
    is worse than any single one. Striding keeps each shard's distribution
    close to the whole, which is the friendly (IID) case — real federated
    data is not IID, and handling that is a research problem, not a demo.
    """
    idx = torch.arange(shard, len(x), num_shards)
    return x[idx], y[idx]


# --- the weight <-> JSON encoding the platform expects ---------------------


def encode(state: dict) -> dict:
    """Flatten a state_dict to {"name": {"shape": [...], "data": [...]}}."""
    return {
        name: {"shape": list(t.shape), "data": t.detach().flatten().tolist()}
        for name, t in state.items()
    }


def decode_into(model: nn.Module, payload: dict) -> None:
    state = model.state_dict()
    for name, entry in payload.items():
        if name not in state:
            raise SystemExit(
                f"weights.json has a parameter this model does not: {name!r}. "
                "The model definition must not change between rounds."
            )
        state[name] = torch.tensor(entry["data"], dtype=state[name].dtype).reshape(
            tuple(entry["shape"])
        )
    model.load_state_dict(state)


def subtract(new: dict, base: dict) -> dict:
    """new - base, in the platform's encoding."""
    return {
        name: {
            "shape": list(new[name]["shape"]),
            "data": [a - b for a, b in zip(new[name]["data"], base[name]["data"])],
        }
        for name in new
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    # These three are supplied by FlashML on every task. --shard is the only
    # one that differs between machines within a round.
    parser.add_argument("--round", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--shard", type=int, required=True)
    # Yours. Anything under `args:` in flashml.yaml lands here.
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.05)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = build_model()

    # Round 0 has no weights.json — that absence IS the signal to start from
    # your own initialisation, not an error to retry.
    if WEIGHTS_IN.exists():
        base = json.loads(WEIGHTS_IN.read_text())
        decode_into(model, base)
        print(f"round {args.round}: resumed from averaged weights", flush=True)
    else:
        base = None
        print(f"round {args.round}: starting from initialisation", flush=True)

    x, y = build_dataset()
    xs, ys = shard_of(x, y, args.shard, args.num_shards)
    samples = int(len(xs))
    if samples == 0:
        # More shards than samples. Fail loudly: a task that reports
        # samples=0 would be given zero weight and silently contribute
        # nothing, which looks like the machine was slow rather than wrong.
        raise SystemExit(
            f"shard {args.shard} of {args.num_shards} is empty "
            f"({N_SAMPLES} samples total) — reduce `shards` in flashml.yaml"
        )

    loss_fn = nn.MSELoss()
    opt = torch.optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    last = float("nan")
    for epoch in range(args.epochs):
        opt.zero_grad()
        loss = loss_fn(model(xs), ys)
        loss.backward()
        opt.step()
        last = float(loss.item())
    print(
        f"shard {args.shard}/{args.num_shards}: {samples} samples, "
        f"loss {last:.4f}",
        flush=True,
    )

    trained = encode(model.state_dict())
    # On round 0 the platform has no base to add a delta to, so the
    # "delta" IS the trained model. From round 1 it is the change.
    delta = trained if base is None else subtract(trained, base)

    DELTA_OUT.write_text(json.dumps(delta))
    METRICS_OUT.write_text(
        json.dumps(
            {
                # Must be the count actually trained on: it is this
                # machine's weight in the average.
                "samples": samples,
                "loss": last,
                "round": args.round,
                "shard": args.shard,
                # Useful when the job view names who contributed to which
                # round; ignored by the platform.
                "host": os.environ.get("HOSTNAME", "unknown"),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
