# Get started

This page takes you from an empty environment to a fault-tolerant job and your
first 2-process DDP run — on CPU, with no cluster and no GPU. For the what and
why, see the **[overview](index.md)**.

## Install

The core is deliberately tiny — `pip install flashruntime` brings **only
pydantic**, and every core module (planner, leases, checkpoints, recovery, the
`flash.submit()` SDK) works with zero infrastructure:

```bash
pip install flashruntime
```

Infrastructure integrations are opt-in extras, never core imports:

```bash
pip install "flashruntime[service]"   # the FastAPI coordinator + CLI
pip install "flashruntime[sklearn]"   # numpy + scikit-learn for the sweep examples
```

**Torch is not a dependency.** FlashRuntime *launches* PyTorch; it never
imports it (the four-axes rule — launching is orthogonal to your framework). To
run the DDP example below, install PyTorch yourself. A CPU-only build is enough
— DDP works over the `gloo` backend with no GPU:

```bash
pip install torch   # CPU build is fine; gloo needs no CUDA
```

## Your first run

`flash.submit()` operates any command. The one convention your script owes
FlashRuntime is to write a `metrics.json` (a flat JSON object) into its working
directory; FlashRuntime collects it and records it as a trial:

```python
import flashruntime as flash

run = flash.submit(flash.CommandWorkload(
    command="python -c \"import json; json.dump({'accuracy': 0.91}, open('metrics.json','w'))\"",
    source=flash.Source(path="."),
))

print(run.state.value)   # "SUCCEEDED"
print(run.trials)        # [{'accuracy': 0.91}]
print(run.artifacts)     # [PosixPath('.../metrics.json')]
```

Rerunning `flash.submit(workload, output_dir=...)` against the **same**
`output_dir` reuses the job id, so a checkpointed script resumes instead of
restarting. Pair that with `max_restarts=N` and a crash mid-run is recovered
automatically from the last valid checkpoint.

## Your first DDP run

The `integrations.pytorch` adapter builds the `torchrun` command for you. If
your script already calls `torch.distributed.init_process_group()` and wraps
its model in `DistributedDataParallel`, there are **zero code changes** — the
adapter just launches it:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(fr_torch.ddp(
    "train.py",
    source="examples/user_pytorch_vanilla",
    nproc_per_node=2,          # 2 processes on this host — gloo/CPU works
    script_args="--steps 100",
))

print(run.state.value, run.trials)
```

`ddp(script, *, source=".", nproc_per_node=2, nnodes=1, script_args="",
env=None)` emits `torchrun --nproc-per-node=N --nnodes=1 --standalone
--local-addr=127.0.0.1 <script> <args>`. The `--local-addr=127.0.0.1` pins the
advertised rendezvous address to loopback (otherwise torchrun advertises
`socket.getfqdn()`, which on some macOS DNS setups is unresolvable and the run
hangs before spawning a process). `nproc_per_node=2` starts two worker
processes that rendezvous on loopback and hand each rank its `RANK` /
`WORLD_SIZE` / `LOCAL_RANK` — a real distributed run on a single machine, no
GPU required.

`nnodes > 1` raises `NotImplementedError` today: multi-node rendezvous is a
launcher concern for a later slice. `--standalone` is single-node by
definition.

> Want fault-tolerant checkpointing inside a script you *are* willing to touch?
> `import flashruntime.torch as ft` gives you `ft.prepare(...)`,
> `ft.checkpoint(...)`, and `ft.log_metrics(...)` — torch's own DDP wrapped
> under the parts-first / manifest-last checkpoint contract, so a killed run
> resumes bit-identically. It is optional sugar on the same launch-only
> contract, never required.

## Watch it run

Pass `watch=True` (or just run at an interactive terminal, where it is the
default) and `flash.submit()` opens a live run page in your browser and prints
its URL:

```python
import flashruntime as flash

run = flash.submit(
    flash.CommandWorkload(command="python train.py", source=flash.Source(path=".")),
    watch=True,
)
print(run.viewer_url)   # http://127.0.0.1:<port>
```

The page draws the run's topology, loss curve, verified checkpoints, and every
recovery decision, refreshing every couple of seconds — served entirely from a
loopback server with **no external assets**, so it renders with the network
cut. These docs are served from that same viewer at `/docs`.
