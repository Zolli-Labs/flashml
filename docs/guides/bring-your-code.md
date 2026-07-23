# Bring your own code

FlashRuntime **operates** your training job — it never rewrites your model.
You keep the framework, the model, the loop, and the loss you already have.
FlashRuntime wraps a reliability and reproducibility layer around them: it
launches your command, injects the environment it promises, tracks metrics,
validates checkpoints, retries on failure, and collects the artifacts.

| You own | FlashRuntime owns |
|---|---|
| model, training loop, loss, data, framework | launch, env vars, metric tracking, checkpoint validity, retry, recovery, artifact collection |

The contract at the boundary is deliberately thin: **CLI flags in,
`metrics.json` out.** A script that already reads its hyperparameters from
`argparse` and writes a small JSON file of results needs *zero* FlashRuntime
imports to be operated (see `examples/user_sklearn/train.py`). The PyTorch
and Hugging Face helpers below are optional sugar on top of that same
contract — never a required dependency.

This is ADR-0003's fourth axis in practice: **recipes integrate user code**.
FlashRuntime plans, launches, observes, and recovers; the distributed math is
always done by your framework (PyTorch DDP, `torchrun`, HF Trainer, sklearn).

---

## Run any repository (no changes to your code)

The local entry point is `flash.submit()`. It compiles your description into a
launch spec, runs it as a real subprocess, waits, and hands back a `Run`:

```python
import flashruntime as flash

run = flash.submit(flash.CommandWorkload(
    command="python train.py --config configs/train.yaml",
    source=flash.Source(path="~/my-project"),
    outputs=flash.OutputSpec(collect=["metrics.json", "checkpoints/**"]),
))
print(run.state, run.artifacts)   # LaunchState.SUCCEEDED, [Path(...), ...]
print(run.logs())                 # captured stdout+stderr (tail)
```

Notes that matter:

- **`command` is `shlex`-split — there is no shell.** `"python a.py | tee log"`
  will *not* pipe; the `|` becomes an argument. For pipes, redirection, or
  globs that the shell would expand, pass an explicit
  `command="bash -c 'python train.py | tee log'"` (or an argv list).
- **`source` is a `flash.Source`, not a bare string.** Use
  `flash.Source(path="~/my-project")`; `~` is expanded for you. It defaults to
  the current directory.
- **`outputs.collect` decides what survives.** Globs are resolved against the
  script's working directory and copied into the run's output dir after the
  process exits. The default is `["metrics.json"]`.
- **`metrics.json` must be in `outputs.collect` (it is, by default) to appear
  in `run.trials`.** FlashRuntime reads the collected `metrics.json` and, if
  it parses to a JSON object, records it as a trial. Your script's one
  convention: write a `metrics.json` (a flat JSON object) to its working
  directory.

`flash.submit(workload, output_dir=None)` is **synchronous and sequential** in
v1: compile → launch → wait → collect, once per parameter set. Rerunning with
the same `output_dir` reuses the same job id, so checkpoints resume (see the
kill-and-resume demo below). There is no `provider=` or `wait=` argument —
remote providers and async submission are later slices (spec §10).

The `Run` handle, fully populated by the time `submit()` returns:

| Attribute | Meaning |
|---|---|
| `run.state` | `LaunchState.SUCCEEDED` if every task succeeded, else `FAILED` |
| `run.trials` | list of parsed `metrics.json` dicts (one per task; fan-out merges its `params`) |
| `run.artifacts` | list of collected file `Path`s |
| `run.output_dir` | root the run wrote under |
| `run.logs(tail_lines=200)` | captured process output |
| `run.best_trial(metric=None, maximize=None)` | best trial by `metric` (defaults from `outputs`) |

---

## scikit-learn: hyperparameter sweeps

sklearn work is *embarrassingly parallel across runs*, never inside a single
`.fit()`. FlashRuntime fans a grid out into one independent task per trial —
it never tries to split one `.fit()` call, which would change the math.

The `integrations.sklearn` adapter builds the workload for you from an
ordinary script that takes `--flag value` and writes `metrics.json`:

```python
import flashruntime as flash
from flashruntime.integrations import sklearn as fr_sklearn

run = flash.submit(fr_sklearn.hpo(
    "train.py",
    {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50]},
    source="examples/user_sklearn",
))
print(f"state={run.state.value}  trials={len(run.trials)}")
print("best:", run.best_trial())     # ranks by outputs.primary_metric
```

`hpo(script, grid, **kwargs)` expands a Cartesian grid
(`{"model": ["logreg","rf"], "C": [0.1, 1]}` → 4 trials) and delegates to
`sweep(script, task_params, *, source=".", metric="accuracy_mean",
maximize=True, python="python")`. Each `{placeholder}` in the built command is
filled from the trial's params, so `train.py` receives `--model rf --C 1.0`
and friends. Because `sweep` sets `outputs.primary_metric=metric`,
`run.best_trial()` needs no arguments — it returns the trial with the highest
`accuracy_mean` (or lowest, when `maximize=False`).

The user script has no FlashRuntime import at all — `examples/user_sklearn/
train.py` is plain sklearn: flags in, `metrics.json` out. That is the whole
contract.

---

## PyTorch: DDP

There are two paths, and both operate *unmodified* torch code — the launcher
starts `torchrun`, which spawns N processes and hands each its `RANK` /
`WORLD_SIZE` / `LOCAL_RANK`.

### Path 1 — a script that is already DDP-ready

If your script already calls `dist.init_process_group()` and wraps its model
in `DistributedDataParallel` itself, there are **zero code changes**. The
adapter just builds the `torchrun` command:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(fr_torch.ddp(
    "train.py",
    source="examples/user_pytorch_vanilla",
    nproc_per_node=2,                      # 2 processes on this host
    script_args="--steps 100",
))
print(run.state.value, run.trials)
```

`ddp(script, *, source=".", nproc_per_node=2, nnodes=1, script_args="",
env=None)` emits `torchrun --nproc-per-node=N --nnodes=1 --standalone <script>
<args>`. `examples/user_pytorch_vanilla/train.py` proves the point: plain
`torch.distributed`, no FlashRuntime import anywhere.

`nnodes > 1` raises `NotImplementedError` today — multi-node rendezvous is a
launcher concern for a later slice (spec §10). `--standalone` is single-node by
definition.

### Path 2 — `flashruntime.torch`, the optional in-script helper

For a script you *are* willing to touch, one import makes it both
launch-anywhere and fault-tolerant, without rebuilding any framework
machinery. The complete surface is seven functions:

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

What each does:

- **`prepare(model, optimizer=None, dataloader=None)`** — when launched
  distributed (`WORLD_SIZE > 1`) it initializes torch's *own* process group
  (`nccl` on GPU, `gloo` on CPU), wraps the model in `DistributedDataParallel`,
  and **swaps the DataLoader's sampler for a seed-0 `DistributedSampler`** so
  each rank sees a disjoint, deterministically-shuffled shard. It then restores
  the newest *valid* checkpoint manifest if one exists, setting the resume
  step. Launched as plain `python train.py`, it is a no-op passthrough.

  **GPU DDP is a later slice.** `prepare` *selects* `nccl` when CUDA is present
  but does **not** yet move your model to the device for you — multi-GPU DDP is
  not exercised end-to-end. The proven path today is CPU / `gloo` (what the e2e
  tests run); on GPU you still call `model.to(device)` yourself, and treat the
  distributed GPU wiring as unverified until that slice lands.

  One caveat worth knowing: `prepare` rebuilds the DataLoader carrying over
  `batch_size`, `collate_fn`, `num_workers`, and `drop_last` — **`shuffle` and
  `pin_memory` are not carried over** (the `DistributedSampler` owns shuffling,
  at seed 0).

- **`checkpoint(model, optimizer=None, *, step, every=None)`** — rank 0 writes
  a checkpoint under the parts-first / manifest-last contract (the manifest is
  written last, after the parts, so a half-written checkpoint is never
  `latest_valid`). `every=N` no-ops except on multiples of N. Every rank
  synchronizes on a barrier so no one races past a partial write.

- **`log_metrics(dict)`** — rank 0 appends one JSON record per call to
  `metrics.jsonl` (streaming history). It never raises — metrics must never
  kill training. This is *separate* from the final `metrics.json` your script
  writes for `run.trials`.

- **`start_step()` / `rank()` / `world_size()` / `is_main()`** — the small
  positional helpers.

The same file, three ways (from `examples/user_pytorch/train.py`):

| Command | What runs |
|---|---|
| `python train.py --steps 200` | single process, `prepare` is a passthrough |
| `torchrun --nproc-per-node=2 --standalone train.py` | DDP by hand |
| `flash.submit(fr_torch.ddp("train.py", ...))` | operated by FlashRuntime |

**Determinism / bit-exact resume.** The example is deterministic on CPU
(fixed seeds; the seed-0 `DistributedSampler` repeats its order every epoch),
so a killed-and-resumed run reproduces the uninterrupted result — recovery
must not change the math. There is one alignment constraint: on resume the
`for` loop restarts the dataloader at batch 0, so the resumed step must land on
an **epoch boundary** — a multiple of batches-per-rank-per-epoch (in the
example, 512 samples / 32 batch / 2 ranks = 8). Keep `--checkpoint-every` a
multiple of that, or the `1e-6` loss comparison in `tests/test_examples_e2e.py`
would drift.

**Kill-and-resume demo.** `examples/bring_your_code_demo.py` runs the whole
story end to end: an sklearn sweep, a 2-process CPU DDP run, then a crash at
step 60, a resubmit against the *same* `output_dir`, and a resume from the
last valid checkpoint manifest:

```bash
.venv/bin/python examples/bring_your_code_demo.py
```

The resumed run reports `resumed_from` > 0 — recovery, not a restart. (The
torch parts skip automatically when `torch`/`torchrun` are not installed.)

> **Warning — one `output_dir` is one workload.** Resume works by reusing the
> job-scoped checkpoint tree at `<output_dir>/local/ckpt`. Point a *different*
> workload at an `output_dir` that already holds another workload's checkpoints
> and `prepare()` will happily restore those foreign weights — silent, wrong
> results, not an error. Use a **fresh `output_dir` per workload**; reusing one
> for the *same* workload is exactly how kill-and-resume is meant to work.
> (Fan-out sweeps are safe automatically: each trial gets its own
> `local-NNN/ckpt` tree, so trials never cross-contaminate.)

**Guardrail (ADR-0003 — do not rebuild Accelerate).** `flashruntime.torch`
wraps torch's *own* DDP and stops. There are no FSDP policies, no autocast, no
DeepSpeed config in this surface. Users who want those use the real framework
features directly — the launcher still launches such a script correctly,
because launching is orthogonal to the strategy your code chooses.

---

## Hugging Face

HF Trainer already wraps DDP/FSDP internally when it is launched by
`torchrun`, so *launching* an HF job is just the PyTorch path. What
`integrations.huggingface` adds is the **callback seam** that commits Trainer
checkpoints as verified manifests and relays Trainer metrics.

```python
import flashruntime as flash
from flashruntime.integrations import huggingface as fr_hf

run = flash.submit(fr_hf.trainer(
    "train_hf.py",
    source="~/hf-project",
    nproc_per_node=1,
    script_args="--model_name_or_path bert-base-uncased",
))
```

Inside your training script, wire the callback and the resume in the usual HF
way — the `transformers` import is paid only in your training process, never in
FlashRuntime's core:

```python
from flashruntime.integrations import huggingface as fr_hf

trainer.add_callback(fr_hf.flashruntime_callback())    # on_save → manifest, on_log → metrics

resume = fr_hf.latest_checkpoint(training_args.output_dir)   # newest VALID checkpoint dir, or None
trainer.train(resume_from_checkpoint=resume)
```

`flashruntime_callback()` builds a `TrainerCallback` whose `on_save` writes a
verified manifest for `checkpoint-<step>/` (rank 0 only) and whose `on_log`
relays metrics through `flashruntime.torch.log_metrics`. `latest_checkpoint
(output_dir)` returns the storage prefix of the newest checkpoint dir with a
*valid* manifest (`None` means fresh start) — pass it straight to
`resume_from_checkpoint`.

`trainer(script, *, source=".", nproc_per_node=1, script_args="")` is a thin
wrapper over the PyTorch `ddp()` adapter, so everything in the PyTorch section
about launching and multi-process applies unchanged.

---

## Submitting to a coordinator (JobSpec)

`flash.submit()` runs on the local machine. To hand a command workload to a
FlashRuntime **coordinator** — so nodes pull and run it under leases,
heartbeats, and recovery — compile it to the wire form and POST it:

```python
from flashruntime.workloads.command import to_jobspec
from flashruntime.protocol.v1alpha1 import ImageSpec

jobspec = to_jobspec(
    workload,
    name="my-sweep",
    image=ImageSpec(repository="myrepo/trainer", tag="2026.07-a1b2c3"),
)
# POST jobspec.model_dump() to  POST /v1alpha1/jobs  (or: flashruntime submit spec.json)
```

`to_jobspec` produces `JobSpec{execution.backend: "leases", workload.type:
"command"}`. **A pinned image is required** — remote runs must be reproducible,
and the schema already rejects the tag `latest`. On the coordinator the
`command` recipe expands the job into one `TaskSpec` per `task_params` entry
(or a single task), each carrying an `argv` payload, its env, its
`artifact://` inputs, and its isolation requirement.

### Isolation tiers (fail closed)

Every command task carries an isolation tier from `workload.isolation`:

| Tier | Where it runs | Meaning |
|---|---|---|
| `standard` (default) | your own machines, RunPod, trusted pools | ordinary placement — runs anywhere |
| `sandboxed` | community / untrusted machines | may only be leased to a node advertising `sandbox_capable is True` |

The placement gate (`scheduler.IsolationAwarePlacement`) is **fail-closed** on
the security-relevant field, per AGENTS.md rule 3:

- A node counts as capable **only** when `sandbox_capable is True` — a truthy
  stand-in (the string `"false"`, `1`, `"yes"`) does **not** count.
- Any tier that is not `None` / `""` / `"standard"` (including a mistyped
  `"Sandboxed"`) is treated as requiring capability — no silent downgrade.
- A `sandboxed` task never falls back to an uncapable node unless the workload
  explicitly sets `isolation.allowFallback = True`.

So a `sandboxed` task will sit unclaimed rather than land on a node that cannot
isolate it. That is the intended behavior: unsafe placement fails closed.

### What runs where today

> **Local SDK path — works now.** `flash.submit()` runs sklearn sweeps,
> 2-process CPU DDP (via `gloo`), and kill-and-resume from checkpoints on this
> machine. All three are proven by `tests/test_examples_e2e.py` (four real
> end-to-end tests).
>
> **Service-side command jobs — expansion works, execution pending.** POSTing
> a `to_jobspec()` workload expands it into leased tasks and places them
> fail-closed by isolation tier (proven by `tests/test_service_command_recipe.py`).
> But **running** an `argv` payload needs FlashNode's argv runner tier, a
> cross-repo, versioned change that has not landed yet. Until it ships, command
> jobs *expand and lease* correctly but only an argv-aware executor can execute
> them.
>
> **Later slices.** Multi-node DDP (`nnodes > 1` rendezvous), remote providers
> (RunPod) with source packaging (`git_revision`), and `flash.run(StrategyPlan)`
> wiring are open follow-ups (spec §10).

---

## Built-in task modules

FlashRuntime ships a few ready-made **task modules** in `flashml_workloads/`
that the Mode A coordinator expands a job into and leases out — worked
examples of the reliability contract, no bring-your-own-code wiring required:

- `sklearn_trial` — one scikit-learn fit/score trial; the unit a
  `hyperparameter_search` job fans out into.
- `kmeans_shard` + `kmeans_driver` — deterministic sharded K-means
  (`sharded_kmeans`), retriable per shard with honest recovery evidence.
- `sgd_trainer` — a checkpointable SGD loop with bit-identical resume from
  the last valid checkpoint manifest.

They install with the `[sklearn]` extra and run through the coordinator's
job→task expansion (`execution.backend: leases`); the `e2e/` suite drives
them through the kill-a-machine sweep and cross-machine resume demos. They
demonstrate the mechanics — they are not the required path. The supported
way to run your own work is always: bring your code and let FlashRuntime
operate it.

---

**See also:** the strategy planner walkthrough
([`docs/planner/README.md`](../planner/README.md)) for choosing *how* a job
should run before you submit it, and ADR-0003
([`docs/adr/0003-reliability-runtime-first-planner-second.md`](../adr/0003-reliability-runtime-first-planner-second.md))
for the four-axis architecture this guide's contract comes from.
