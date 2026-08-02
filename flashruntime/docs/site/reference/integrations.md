# Reference: integrations (`flashruntime.integrations`)

Framework adapters that build a `CommandWorkload` for you. Each is a small,
framework-neutral function — the recipe axis of the four-axis architecture. No
adapter imports its framework at the top level (the import is paid only in your
training process).

Signatures below are exact; each entry says, in one line, *why it exists*.

---

## `integrations.pytorch` (`fr_torch`)

```python
def ddp(
    script,
    *,
    source=".",
    nproc_per_node=2,
    nnodes=1,
    script_args="",
    env=None,
) -> CommandWorkload: ...
```

Builds the `torchrun` launch command for a PyTorch script — the launch
conventions only; DDP is wired by your code (or `flashruntime.torch.prepare`).
Emits `torchrun --nproc-per-node=N --nnodes=1 --standalone
--local-addr=127.0.0.1 <script> <args>`. `nnodes > 1` raises
`NotImplementedError` (multi-node rendezvous is a later slice). See the
[PyTorch guide](../guides/pytorch.md).

---

## `integrations.sklearn` (`fr_sklearn`)

```python
def sweep(
    script,
    task_params,
    *,
    source=".",
    metric="accuracy_mean",
    maximize=True,
    python="python",
) -> CommandWorkload: ...

def hpo(script, grid, **kwargs) -> CommandWorkload: ...
```

- **`sweep`** — one independent task per params dict; sets
  `outputs.primary_metric=metric` so `run.best_trial()` needs no arguments.
- **`hpo`** — Cartesian-grid sugar over `sweep`
  (`{"model": ["logreg", "rf"], "C": [0.1, 1]}` → 4 trials).

Distributes across runs, never inside a single `.fit()`. See the
[scikit-learn guide](../guides/sklearn.md).

---

## `integrations.huggingface` (`fr_hf`)

```python
def trainer(script, *, source=".", nproc_per_node=1, script_args="") -> CommandWorkload: ...

def latest_checkpoint(output_dir) -> str | None: ...

def flashruntime_callback(): ...             # returns a transformers TrainerCallback
```

- **`trainer`** — a thin wrapper over `pytorch.ddp()`; launching an HF Trainer
  job is just the PyTorch path (Trainer wraps DDP/FSDP internally under
  `torchrun`).
- **`latest_checkpoint`** — newest `checkpoint-*` dir with a **valid** manifest
  (or `None`); pass straight to `trainer.train(resume_from_checkpoint=...)`.
- **`flashruntime_callback`** — a `TrainerCallback` whose `on_save` commits a
  verified manifest and whose `on_log` relays metrics through
  `flashruntime.torch.log_metrics`. The `transformers` import is paid inside
  this factory, in your process.

See the [Hugging Face guide](../guides/huggingface.md).

---

## The extensibility pattern

Every adapter returns a `CommandWorkload` and reuses the same
launch/collect/recover machinery. To teach FlashRuntime a new framework, write a
function under `flashruntime/integrations/` that returns a `CommandWorkload`
describing *what to run* (and, if the framework has hooks, maps them onto
`write_manifest` / `log_metrics`). No core change is required — that is the
four-axis payoff. The [SDK reference](sdk.md) documents the `CommandWorkload`
shape you build.
