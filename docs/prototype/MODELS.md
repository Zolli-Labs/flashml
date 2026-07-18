# FlashML Model Contract — Bringing Any ML/DL Model

This is the model-side counterpart to [PROVIDERS.md](PROVIDERS.md). PROVIDERS.md
is the contract for plugging in a new **compute backend** (RunPod, SSH, ...).
This document is the contract for plugging in a new **model** — scikit-learn,
PyTorch, or anything else — without FlashML hand-integrating each library.

**Core position:** FlashML does not "support library X." It supports one small
protocol (`DistributedAlgorithm`, already defined in
`flashml/algorithms/base.py`). Any model — regardless of framework — that can
be expressed as that protocol runs on every current and future `Provider`
with zero engine changes. Library breadth is a question of how many
*generic adapters* implement that protocol well, not how many libraries get
bespoke code.

---

## 1. The insight that makes this cheap

Look again at what `DistributedAlgorithm` actually requires
(`flashml/algorithms/base.py`):

```python
plan(dataset, storage, job_id) -> shards
initialize(shards, storage) -> state
make_tasks(state, shards, iteration) -> [payload, ...]
reduce(state, results) -> (new_state, metrics)
converged(old_state, new_state, metrics, threshold) -> bool
finalize(state, shards, storage) -> result
```

`state` is just a Python dict. Nothing about the interface requires it to be
JSON-serializable floats — that's just what `KMeansAlgorithm` and
`SklearnPartialFitAlgorithm` chose to do, because sklearn's `coef_`/centroids
are small enough to inline. **The engine's `MAX_PAYLOAD_BYTES` check
(`flashml/engine/loop.py`) only fires on the per-task `payload`, never on
`state` itself.** So the exact pattern the engine already uses for
*datasets* — write it to `Storage` once, pass a small key around — works
identically for *model weights*:

```python
# initialize(): build a fresh model, checkpoint it, return a pointer
model = model_fn()
storage.put(f"{job_id}/ckpt_0.pt", serialize(model.state_dict()))
return {"checkpoint_key": f"{job_id}/ckpt_0.pt", "iteration": 0}

# make_tasks(): payload stays tiny — a shard key + a checkpoint key
return [{"shard": shard, "checkpoint_key": state["checkpoint_key"]} for shard in shards]

# worker-side map_task(): loads both from Storage, trains locally, checkpoints its own result
model.load_state_dict(deserialize(storage.get(payload["checkpoint_key"])))
train_locally(model, load_shard(payload["shard"], storage))          # real framework training loop
key = f"{job_id}/worker_{worker_id}_iter_{iteration}.pt"
storage.put(key, serialize(model.state_dict()))
return {"checkpoint_key": key, "n_samples": ..., "loss": ...}        # small payload

# reduce(): coordinator loads each worker's checkpoint, averages, writes the new global checkpoint
```

**No `Provider`, `WorkerPool`, `Storage`, or engine change is required for
this.** This is the single most important conclusion of this document —
deep learning support is an *algorithm* to write, not new infrastructure.

---

## 2. Three canonical distributed shapes

Every model — classical or deep — fits one of these. Pick the shape by what
the model needs, not by which library it's from.

| Shape | What happens each iteration | Fits |
|---|---|---|
| **MapReduce** | Map a stateless function over shards, reduce to shared statistics | K-Means, histograms, counting-based algorithms |
| **Parameter Server (FedAvg)** | Broadcast global state → each worker runs *K local training steps* on its shard → weighted-average the results back into global state | Anything with a "local step" (sklearn `partial_fit`, PyTorch/any-DL mini-batch training, gradient boosting with a mergeable histogram) — **this is the workhorse for "any trainable model"** |
| **Embarrassingly Parallel** | Each worker independently does a full, unrelated unit of work; no cross-worker state at all | Per-series time-series models (sktime/darts/statsmodels/Prophet), ensembles, hyperparameter/cross-validation sweeps |

Today only MapReduce (`KMeansAlgorithm`) and Parameter Server
(`SklearnPartialFitAlgorithm`) are implemented — both against sklearn.
**Embarrassingly Parallel doesn't exist yet** (it was in the original
`IMPLEMENTATION.md` scope as "Parallel" and never got built) and unlocks time
series + hyperparameter search cheaply — see [ROADMAP.md](ROADMAP.md) Phase 2.

**Deliberately not a fourth shape here: true DDP/NCCL all-reduce.** See §5 —
that one needs a different `Provider` capability, not a new algorithm, and is
explicitly out of scope for serverless connectors.

---

## 3. Built-in generic adapters (the "supported libraries" list)

Rather than one adapter per library, ship a small number of *generic*
adapters, each parameterized by what the user passes in:

| Adapter | Shape | Status | Covers |
|---|---|---|---|
| `SklearnPartialFitAlgorithm` | Parameter Server | **Done** | Any sklearn estimator with `partial_fit` — `SGDRegressor/Classifier`, `Perceptron`, `PassiveAggressive*`, and anything added to `sklearn_registry.py` |
| `TorchParameterServerAlgorithm` | Parameter Server | **Planned (Roadmap Phase 2)** | Any `torch.nn.Module` + optimizer + loss — the deep-learning entry point |
| `EmbarrassinglyParallelAlgorithm` | Embarrassingly Parallel | **Planned (Roadmap Phase 2)** | Any `fit(X, y) -> fitted_model` callable run independently per shard/series — sktime, darts, statsmodels, Prophet, XGBoost/LightGBM full-fit-per-shard, hyperparameter sweeps |
| Full custom `DistributedAlgorithm` subclass | any | Always available (escape hatch) | Anything the generic adapters don't fit — e.g. a custom MapReduce-shaped algorithm |

This is the direct answer to "should I start with sklearn and add libraries
later": **yes, but add *shapes*, not libraries.** Once
`TorchParameterServerAlgorithm` and `EmbarrassinglyParallelAlgorithm` exist,
adding sktime, darts, XGBoost, or a HuggingFace fine-tuning loop is
"pass the right constructor," not new algorithm code — the same way adding
`Perceptron` today was a 5-line subclass of `SklearnPartialFitAlgorithm`.

### Target user-facing API (Phase 2, not yet built)

```python
import torch, torch.nn as nn
import flashml

with flashml.Cluster(provider="runpod", workers=8, accelerator="gpu-24gb") as cluster:
    job = cluster.train(
        algorithm=flashml.algorithms.TorchModel(
            model_fn="mypackage.models:MyModel",       # dotted path, see §4 — not a live object
            optimizer_fn="mypackage.models:make_optimizer",
            loss_fn="torch.nn:CrossEntropyLoss",
            local_steps=50,       # local mini-batches per sync round (federated-style)
            batch_size=64,
            n_shards=8,
        ),
        dataset=my_dataset,
        max_iterations=100,
    )
```

```python
job = cluster.train(
    algorithm=flashml.algorithms.EmbarrassinglyParallel(
        fit_fn="mypackage.ts:fit_one_series",   # (shard_df) -> fitted model bytes
        n_shards=len(all_series),               # one shard per series, e.g.
    ),
    dataset=all_series,
)
```

---

## 4. The real gap: getting *user* code onto the worker

Built-in algorithms work today because `flashml.algorithms.kmeans:map_task`
ships inside the `flashml` package itself, so `resolve_entrypoint()`
(`flashml/providers/entrypoint.py`) can `importlib` it on any worker that has
`flashml` installed. A user's own `MyModel(nn.Module)` is **not** installed on
a fresh RunPod worker by default — this is the actual unsolved problem for
"bring any model," independent of which distributed shape is used.

Two options, in order of implementation cost — recommend building (a) first:

**(a) Dotted-path / installable-package convention (recommended first):**
The user's model/optimizer/loss are referenced the same way `entrypoint`
already is — importable dotted paths. FlashML ships the user's training code
to the worker as a `WorkerSpec.dependencies` entry (already a field on
`WorkerSpec` in `flashml/providers/base.py` — a RunPod `@Endpoint` already
takes a `dependencies` list per `TOOLS.md`). Concretely this means either the
user's package is `pip`-installable (from PyPI or a private index) and listed
in `dependencies`, or FlashML uploads the user's module source as a blob via
`Storage` and the generic worker entrypoint downloads + imports it before
resolving `model_fn`. Matches how the RunPod connector (Phase 1) is already
being designed — see the Phase 1 requirements in
[`phases/01-runpod-connector/README.md`](phases/01-runpod-connector/README.md).
Same constraint as
`AuthError`/`ProvisionError`: fail loud at `provision()` time if the package
can't be resolved, not mid-task.

**(b) `cloudpickle` convenience shim (later, opt-in):** let the user pass a
*live* object (`model_fn=MyModel()`) and have FlashML `cloudpickle` it,
`Storage.put()` the bytes, and have the worker `cloudpickle.load()` it back.
Much nicer DX, but: requires identical Python/library versions on both ends,
is fragile for classes with CUDA state or open resources, and is a real
concern once bytes are coming from a remote NetworkVolume (unpickling
executes arbitrary code — treat this the same as any other "deserializing
untrusted input" boundary, and only enable it for a job's own coordinator ↔
its own workers, never across jobs/tenants). Ship this only as a documented
opt-in once (a) is solid, not as the default path.

The `local` connector sidesteps this whole problem — it's the same process,
so a live object just works with no serialization boundary at all. This is
worth remembering: **the entrypoint/dependency problem is connector-specific,
not algorithm-specific** — it will resurface identically for the `ssh`
connector in Phase 4 and needs exactly one solution reused by both.

---

## 5. Explicitly out of scope: true DDP / NCCL all-reduce

Real PyTorch DDP synchronizes gradients via collective communication
(NCCL/gloo all-reduce) between workers that can address each other directly
over a fast interconnect, every training step. RunPod Flash-style serverless
endpoints are ephemeral HTTP services with no guaranteed direct networking
between workers and no persistent process across calls — this is
structurally the opposite of what NCCL wants, on any serverless GPU
marketplace, not just RunPod.

**Strategic call: don't chase real DDP on serverless.** The Parameter
Server / FedAvg shape in §2 — sync every `local_steps` mini-batches through
the coordinator over HTTP+Storage instead of every step over NCCL — is the
version of distributed deep learning that actually fits serverless
infrastructure, and it's already a proven pattern (this is standard
Federated Learning). The tradeoff is real and worth stating plainly: FedAvg
with infrequent syncs is an approximation of single-machine SGD (same caveat
already documented for the sklearn Parameter Server path in
`sklearn_partial_fit.py`), not identical to it. It's a good fit for
fine-tuning, mid-size models, and classical ML at scale. It is a poor fit for
large-scale foundation-model pretraining, which genuinely needs tight
per-step gradient sync — that workload is out of scope for this project's
serverless-first design, and would need a fundamentally different `Provider`
capability (persistent, directly-networked pods, not serverless endpoints —
see `ROADMAP.md` Phase 7 for where this is tracked as research, not a
committed deliverable).

---

## 6. Checklist: adding a new generic adapter

- [ ] Confirm it fits one of the three shapes in §2 — if it doesn't, that's
      a sign a fourth shape is genuinely needed; don't force it
- [ ] `flashml/algorithms/<name>.py`: `DistributedAlgorithm` subclass +
      worker-side entrypoint function
- [ ] Large state (model weights) goes through `Storage` as a checkpoint key,
      never inlined into `make_tasks()` payloads (§1)
- [ ] Prefer a safe binary codec for weights (e.g. `safetensors`) over
      `pickle`-based serialization where the bytes may have come from a
      shared/remote store
- [ ] User-supplied model/optimizer/loss are dotted import paths (§4a), not
      live objects, until the cloudpickle shim (§4b) exists
- [ ] Add an entry to whichever registry pattern fits (see
      `sklearn_registry.py` for the existing example)
- [ ] Example script proving numerical correctness against a known result,
      same bar as `examples/local_kmeans_and_linear_regression.py`
