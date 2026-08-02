# PyTorch guide

FlashRuntime **operates** your PyTorch job — it never rewrites your model. You
keep the framework, the model, the loop, and the loss you already have;
FlashRuntime launches your command, injects the environment it promises, tracks
metrics, validates checkpoints, retries on failure, and collects artifacts.

This is ADR-0003's fourth axis in practice: **recipes integrate user code**.
FlashRuntime plans, launches, observes, and recovers; the distributed math is
always done by PyTorch (`torchrun`, DDP). There are two paths, and both operate
*unmodified* torch code.

For a full worked walkthrough, do the [ConvNet tutorial](../tutorials/convnet.md).

---

## Path 1 — a script that is already DDP-ready

If your script already calls `dist.init_process_group()` and wraps its model in
`DistributedDataParallel` itself, there are **zero code changes**. The adapter
just builds the `torchrun` command:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(fr_torch.ddp(
    "train.py",
    source="examples/user_pytorch_vanilla",
    nproc_per_node=2,           # 2 processes on this host — gloo/CPU works, no GPU
    script_args="--steps 100",
))
print(run.state.value, run.trials)
```

`ddp(script, *, source=".", nproc_per_node=2, nnodes=1, script_args="",
env=None)` emits `torchrun --nproc-per-node=N --nnodes=1 --standalone
--local-addr=127.0.0.1 <script> <args>`. `--local-addr=127.0.0.1` pins the
advertised rendezvous address to loopback (otherwise torchrun advertises
`socket.getfqdn()`, which on some macOS DNS setups is unresolvable and the run
hangs before spawning a process).

`nnodes > 1` raises `NotImplementedError` today — multi-node rendezvous is a
launcher concern for a later slice. `--standalone` is single-node by
definition.

---

## Path 2 — `flashruntime.torch`, the optional in-script helper

For a script you *are* willing to touch, one import makes it both
launch-anywhere and fault-tolerant, without rebuilding any framework machinery.
The surface is three verbs plus read-only launch-fact accessors:

```python
import flashruntime.torch as ft

model, optimizer, loader = ft.prepare(model, optimizer, loader)
start = ft.start_step()                        # 0 fresh, >0 after a resume
...
ft.checkpoint(model, optimizer, step=step, every=100)
ft.log_metrics({"step": step, "loss": float(loss)})
if ft.is_main():                               # ft.rank(), ft.world_size() too
    ...
```

- **`prepare(model, optimizer=None, dataloader=None)`** — launched distributed
  (`WORLD_SIZE > 1`) it initializes torch's *own* process group (`nccl` on GPU,
  `gloo` on CPU), wraps the model in `DistributedDataParallel`, and **swaps the
  DataLoader's sampler for a seed-0 `DistributedSampler`** so each rank sees a
  disjoint, deterministically-shuffled shard. It then restores the newest
  *valid* checkpoint manifest if one exists, setting the resume step. Launched
  as plain `python train.py`, it is a no-op passthrough.

  **CUDA device placement is wired.** `prepare` selects `nccl` when CUDA is
  present, moves your model onto this rank's GPU *before* the DDP wrap, and
  binds DDP with `device_ids` / `output_device` — so a single-GPU box "just
  works" and you no longer call `model.to(device)` yourself. The everyday e2e
  tests exercise CPU / `gloo`; real-GPU validation is tracked in the workspace
  progress log. **Multi-node DDP (`nnodes > 1`) is a later slice.**

  One caveat: `prepare` rebuilds the DataLoader carrying over `batch_size`,
  `collate_fn`, `num_workers`, and `drop_last` — **`shuffle` and `pin_memory`
  are not carried over** (the `DistributedSampler` owns shuffling, at seed 0).

- **`checkpoint(model, optimizer=None, *, step, every=None)`** — rank 0 writes
  a checkpoint under the parts-first / manifest-last contract (the manifest is
  written last, so a half-written checkpoint is never `latest_valid`). `every=N`
  no-ops except on multiples of N. Every rank synchronizes on a barrier so no
  one races past a partial write.

- **`log_metrics(dict)`** — rank 0 appends one JSON record per call to
  `metrics.jsonl` (streaming history; the live page's loss curve reads it). It
  never raises — metrics must never kill training. This is *separate* from the
  final `metrics.json` your script writes for `run.trials`.

- **`start_step()` / `rank()` / `world_size()` / `is_main()` / `device()` /
  `backend()`** — the small read-only helpers (`device()`/`backend()` let a
  script report where it actually trained, e.g. into `metrics.json` — how
  the GPU e2e proves the CUDA/nccl path). Full signatures in the
  [torch helper reference](../reference/torch-helper.md).

The same file, three ways:

| Command | What runs |
|---|---|
| `python train.py --steps 200` | single process, `prepare` is a passthrough |
| `torchrun --nproc-per-node=2 --standalone train.py` | DDP by hand |
| `flash.submit(fr_torch.ddp("train.py", ...))` | operated by FlashRuntime |

### Determinism / bit-exact resume

Keep the script deterministic on CPU (fixed seeds; the seed-0
`DistributedSampler` repeats its order every epoch) and a killed-and-resumed
run reproduces the uninterrupted result — recovery must not change the math.
There is one alignment constraint: on resume the `for` loop restarts the
dataloader at batch 0, so the resumed step must land on an **epoch boundary** —
a multiple of batches-per-rank-per-epoch. Keep `--checkpoint-every` a multiple
of that.

> **One `output_dir` is one workload.** Resume works by reusing the job-scoped
> checkpoint tree under `output_dir`. Point a *different* workload at an
> `output_dir` that already holds another workload's checkpoints and
> `prepare()` will happily restore those foreign weights — silent wrong
> results, not an error. Use a **fresh `output_dir` per workload**; reusing one
> for the *same* workload is exactly how kill-and-resume is meant to work.
> (Fan-out sweeps are safe automatically: each trial gets its own checkpoint
> tree.)

### Guardrail (ADR-0003 — do not rebuild Accelerate)

`flashruntime.torch` wraps torch's *own* DDP and stops. There are no FSDP
policies, no autocast, no DeepSpeed config in this surface. Users who want those
use the real framework features directly — the launcher still launches such a
script correctly, because launching is orthogonal to the strategy your code
chooses.

---

## Adding another framework

The PyTorch adapter is ~50 lines: it builds a `torchrun` command string and
returns a `CommandWorkload`. That is the whole extensibility pattern — a new
framework adapter is a small function under `flashruntime/integrations/` that
returns a `CommandWorkload`, reusing the same launch/collect/recover machinery.
`integrations.huggingface` is literally a thin wrapper over this
`ddp()` — see the [Hugging Face guide](huggingface.md) — and
`integrations.sklearn` is the same shape for the fan-out case. No core change is
needed to teach FlashRuntime a new framework; you describe *what to run*, and
the four axes handle the rest.
