# Reference: torch helper (`flashruntime.torch`)

The optional in-training-script helper: one import
(`import flashruntime.torch as ft`) makes a PyTorch script both
launch-anywhere and fault-tolerant. `torch` is imported **inside** these
functions only — FlashRuntime's core never depends on it.

The surface is three verbs plus read-only launch-fact accessors — a
deliberate guardrail (ADR-0003: do not rebuild Accelerate). The boundary is
*capability*, not count: there are no FSDP policies, no autocast, no
DeepSpeed config here, and there never will be. Signatures are exact; each
says, in one line, *why it exists*.

```python
def prepare(model, optimizer=None, dataloader=None): ...
def checkpoint(model, optimizer=None, *, step, every=None) -> None: ...
def log_metrics(metrics: dict) -> None: ...
def start_step() -> int: ...
def rank() -> int: ...
def world_size() -> int: ...
def is_main() -> bool: ...
def device() -> str: ...          # "cpu" or "cuda:N" — where prepare() put the model
def backend() -> str | None: ...  # "gloo"/"nccl", None when single-process
```

---

### `prepare(model, optimizer=None, dataloader=None)`

Wires distributed execution and restores the newest valid checkpoint; returns
the possibly-wrapped `(model, optimizer, dataloader)` triple.

- Launched distributed (`WORLD_SIZE > 1`): initializes torch's own process
  group (`nccl` on GPU, `gloo` on CPU), wraps the model in
  `DistributedDataParallel`, and swaps the DataLoader's sampler for a seed-0
  `DistributedSampler` (each rank sees a disjoint, deterministic shard).
- On CUDA: selects `nccl`, moves the model onto this rank's GPU **before** the
  DDP wrap, and binds `device_ids` / `output_device` — you no longer call
  `model.to(device)`.
- Restores the newest **valid** checkpoint manifest if one exists, setting the
  resume step (read it with `start_step()`).
- Launched as plain `python train.py`: a no-op passthrough.

> The rebuilt DataLoader carries over `batch_size`, `collate_fn`,
> `num_workers`, and `drop_last`. **`shuffle` and `pin_memory` are not carried
> over** — the seed-0 `DistributedSampler` owns shuffling.

### `checkpoint(model, optimizer=None, *, step, every=None)`

rank 0 writes a resumable checkpoint under the **parts-first / manifest-last**
contract (the manifest is written last, so a half-written checkpoint is never
`latest_valid`). `every=N` no-ops except on multiples of N. Every rank
synchronizes on a barrier so no one races past a partial write.

### `log_metrics(metrics: dict)`

rank 0 appends one JSON record per call to `metrics.jsonl` — the streaming
history the live page draws as a loss curve. **Never raises** (metrics must
never kill training). Separate from the final `metrics.json` your script writes
for `run.trials`.

### `start_step() -> int`

The first step the loop should run: `0` fresh, `>0` after a resume (set by
`prepare()` when it restores a checkpoint). Start your loop from it.

### `rank()` / `world_size()` / `is_main()`

The small positional helpers, read from the launch environment: `rank()` is
this process's `RANK` (default `0`), `world_size()` is `WORLD_SIZE` (default
`1`), and `is_main()` is `rank() == 0` — guard rank-0-only work (writing the
final `metrics.json`, printing) with it.

---

## Usage shape

```python
import flashruntime.torch as ft

model, optimizer, loader = ft.prepare(model, optimizer, loader)
start = ft.start_step()

step = start
while step < total_steps:
    for x, y in loader:
        ...
        step += 1
        ft.checkpoint(model, optimizer, step=step, every=100)
        ft.log_metrics({"step": step, "loss": float(loss)})

if ft.is_main():
    ...   # write the final metrics.json
```

The full worked example is the [ConvNet tutorial](../tutorials/convnet.md); the
launch side is the [PyTorch guide](../guides/pytorch.md).
