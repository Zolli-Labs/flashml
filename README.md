# flashml-example-federated

A complete, runnable FlashML job: a small PyTorch model trained across
several machines by federated averaging.

Submit it by pasting this repo's URL into the FlashML console.

## This is not DDP

DistributedDataParallel needs every rank to reach every other rank — NCCL or
Gloo all-reduce on each backward pass. FlashML task containers run with
`--network none`, because a volunteer is lending you a machine, not an open
socket. There are no peers to reach, and there is no rank 0 to rendezvous
with.

What runs instead is **federated averaging**. Each round:

1. every machine gets the same starting weights and a different slice of the
   data
2. each trains locally for a few epochs, alone
3. each uploads how much it changed the weights
4. the platform averages those changes, weighted by how many samples each
   machine actually trained on, and that becomes the next round's weights

Weights cross the network once per round instead of once per batch. That is
what makes it survivable on laptops over home wifi — someone can close their
lid mid-round, miss it, and rejoin the next one. It converges more slowly per
round than synchronous DDP. That is the trade, and it is the one that works
on hardware nobody is paying for.

## What the platform requires of `train.py`

FlashML runs your entrypoint once per shard per round:

```
python /work/inputs/code/train.py --round R --num-shards K --shard N
```

and requires three things:

| | |
|---|---|
| read `/work/inputs/weights.json` | the round's starting weights. **Absent on round 0** — that absence is the signal to use your own initialisation |
| write `/work/out/delta.json` | `{"<param>": {"shape": [...], "data": [...]}}`. On round 0, where you were given nothing, write the trained weights themselves |
| write `/work/out/metrics.json` | at least `{"samples": <positive int>, "loss": <number>}`. `samples` is your weight in the average, so it must be what you actually trained on |

Everything else is yours.

## Rehearse it before spending anyone's battery

```bash
pip install "flashruntime @ git+https://github.com/Zolli-Labs/flashruntime" torch
python simulate.py
```

```
round 0: mean loss 0.6245   (shards: 0.6378 0.6158 0.6197)
round 1: mean loss 0.4832   (shards: 0.4940 0.4742 0.4813)
round 2: mean loss 0.3671   (shards: 0.3741 0.3580 0.3691)
round 3: mean loss 0.2843   (shards: 0.2868 0.2757 0.2904)
round 4: mean loss 0.2360   (shards: 0.2355 0.2279 0.2446)
```

`simulate.py` runs the same `train.py`, with the same shards and rounds, and
averages with FlashML's own `reduce_deltas` — not a reimplementation. If the
encoding is wrong it fails here, on your machine, in seconds.

**A flat loss is the failure to watch for.** It means every shard trained and
nothing was combined — usually `delta.json` holding whole weights instead of
the change from round 1 onward.

## Two things that are easy to get wrong

**Every shard must start round 0 from identical weights.** Averaging N
independently-initialised networks gives the mean of N unrelated models,
which is worse than any of them. `train.py` seeds initialisation from a fixed
`INIT_SEED` that is deliberately *not* derived from `--shard`. This failure
looks like "federated learning doesn't work" rather than like a bug.

**Shard by striding, not by slicing.** A contiguous split gives each machine a
different region of the feature space, so every shard trains on a biased
sample and the average is worse than any single one. Real federated data is
not IID; handling that properly is a research problem, not a demo.

## Why the data is synthetic

Task containers have no network. Nothing can be downloaded at runtime, and a
dataset large enough to be interesting is too large to ship as a repo input.
The relationship in `train.py` is genuinely learnable, so a falling loss means
the averaging works rather than that the numbers happen to shrink.

## `flashml.yaml`

```yaml
mode: federated       # without this the shards run once and never talk
rounds: 5             # how many times to average
min_participants: 2   # quorum — set BELOW `shards`, or one slow laptop
                      # gates every round
shards: 3             # data slices, and therefore parallel machines
```

`min_participants: 2` with `shards: 3` is what makes a closed lid survivable:
the round completes on two, and the third rejoins next round.
