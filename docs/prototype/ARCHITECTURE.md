# FlashML Architecture (As-Built)

> This document describes what actually exists in the code today, not the plan.
> For the pitch, read [README.md](../../README.md). For the forward build plan
> phase-by-phase, read [ROADMAP.md](ROADMAP.md). For the connector contract
> (how a new compute backend plugs in), read [PROVIDERS.md](PROVIDERS.md). For
> the model contract (how a new ML/DL framework plugs in — PyTorch, sktime,
> etc.), read [MODELS.md](MODELS.md). For RunPod-specific technical depth
> (Flash SDK, NetworkVolume, `runpodctl`), read [TOOLS.md](TOOLS.md).
> [IMPLEMENTATION.md](IMPLEMENTATION.md) is the original MVP design doc that
> this was built from — some details there (a separate `Strategy` abstraction,
> a `flashml/server/` layer) haven't been built yet or were simplified; this
> doc is the source of truth for what's real.

---

## 1. Status snapshot

| Piece | Status | Where |
|---|---|---|
| `flashml` package skeleton, public API | **Done** | `flashml/` |
| `Provider` / `WorkerPool` / `Storage` / `DistributedAlgorithm` interfaces | **Done** | `flashml/providers/base.py`, `flashml/storage/base.py`, `flashml/algorithms/base.py` |
| Local connector (thread pool, temp-dir storage) | **Done** | `flashml/providers/local/` |
| Algorithms: K-Means (MapReduce), Linear/Logistic Regression, Perceptron, Passive-Aggressive (all Parameter Server via generic sklearn `partial_fit` bridge) | **Done** | `flashml/algorithms/` |
| Tests (conformance-style, against `local`) + runnable example | **Done, passing** | `tests/`, `examples/` |
| RunPod connector | **Not started** | legacy code to port lives in `legacy/coordinator/flash_worker/`, target design in `TOOLS.md` |
| `flashml serve` + dashboard reattach | **Not started** | legacy code lives in `legacy/coordinator/server.py`; `apps/dashboard/` still points at it |
| Conformance test suite (as a public `flashml.testing` module) | **Not started** | today's tests hit this informally but aren't packaged as one |
| SSH connector | **Not started** | |

**⚠️ Important:** `git status` shows `flashml/`, `tests/`, `examples/`, and
`pyproject.toml` as **untracked**. The only thing committed so far is the
planning docs (README, IMPLEMENTATION.md, PROVIDERS.md, TOOLS.md). All of the
"Done" rows above exist only in the working tree. Committing this baseline is
the first task in [ROADMAP.md](ROADMAP.md) — do it before starting new work,
so there's a clean history to diff against.

---

## 2. Repository map

```text
flashml/              the library — everything under "Done" above
apps/dashboard/       retained Next.js product client. It builds today but still
                      calls the legacy API until Phase 3.
legacy/coordinator/   fused FastAPI/RunPod implementation retained only as the
                      migration source for Phases 1 and 3.
archive/experiments/  pre-library POCs and local datasets; not package code.
docs/
    INDEX.md          documentation map — start here if you're new to the repo
    status/           live verified stage and missing requirements
    phases/           one folder per phase with requirements and exit gates
    guides/           learning-oriented walkthroughs of real code
    ARCHITECTURE.md   this document — as-built code map
    ROADMAP.md        compact phase index
    PROVIDERS.md      compute-backend (connector) contract
    MODELS.md         ML/DL model/framework contract
    TOOLS.md          RunPod Flash / runpodctl technical reference — the spec
                      the RunPod connector (Phase 1) needs to satisfy
    IMPLEMENTATION.md original MVP design doc (partially superseded, see above)
tests/                pytest suite, currently exercises the local connector
examples/             runnable end-to-end scripts (no pytest needed)
```

---

## 3. The public API

```python
import flashml

with flashml.Cluster(provider="local", workers=4) as cluster:
    job = cluster.train(
        algorithm=flashml.algorithms.KMeans(k=3, n_shards=4),
        dataset=X,                     # numpy array, or (X, y) for supervised algorithms
        max_iterations=20,
        convergence_threshold=1e-3,
    )
    for event in job.stream():
        print(event.iteration, event.metrics)
    result = job.result()
```

Exported from `flashml/__init__.py`: `Cluster`, `Job`, `JobEvent`, `algorithms`,
`registered_providers`. There is intentionally no separate `strategy=`
argument: the loop shape is fully determined by the `algorithm` object (see
§5). This is a deliberate simplification versus the original two-abstraction
plan (see §6).

---

## 4. End-to-end execution flow

`flashml/engine/cluster.py:Cluster.train()` does five things, in order:

1. **Provision** — `self._provider.provision(WorkerSpec(accelerator, entrypoint), workers)`
   asks the connector for a `WorkerPool` of the requested size. `entrypoint`
   comes from `algorithm.entrypoint`, a dotted `"module:function"` string —
   this is the *only* thing a connector ever learns about the algorithm
   (`flashml/providers/entrypoint.py:resolve_entrypoint`).
2. **Get storage** — `self._provider.storage()` returns a `Storage` reachable
   from that provider's workers.
3. **Run the loop** — `flashml/engine/loop.py:run_job()`:
   ```text
   shards = algorithm.plan(dataset, storage, job_id)     # writes shards to storage ONCE
   state  = algorithm.initialize(shards, storage)
   for iteration in range(max_iterations):
       payloads = algorithm.make_tasks(state, shards, iteration)   # small dicts only
       # >MAX_PAYLOAD_BYTES (1MB) raises before it reaches the connector
       results  = pool.gather([pool.submit(Task(...)) for p in payloads], timeout)
       state, metrics = algorithm.reduce(state, results)
       job._emit(JobEvent(iteration, metrics, worker_states))
       if algorithm.converged(old_state, state, metrics, threshold):
           break
   final = algorithm.finalize(state, shards, storage)
   ```
   `run_job` never raises — failures are captured on the `Job` via `_fail()`
   and surface when the caller calls `job.result()`.
4. **Teardown** — `provider.teardown(pool)` unconditionally, whether the loop
   succeeded or raised.
5. `Job` is returned; `job.stream()` replays the events recorded during the
   (currently synchronous) run, `job.result()` returns the final value or
   re-raises the captured exception.

Today the whole loop runs synchronously inside `cluster.train()` before it
returns — `Job` looks async (queue + `stream()`) so the public API doesn't
have to change when a real background/async engine lands later.

---

## 5. Core interfaces

### `Provider` / `WorkerPool` / `Storage` — `flashml/providers/base.py`, `flashml/storage/base.py`

The contract every connector implements. Full spec with conformance checklist
lives in [PROVIDERS.md](PROVIDERS.md) — that document is written for someone
adding a new connector and is more complete than this summary. Key point for
reading the code: `WorkerPool.submit()` returns a `TaskHandle`; `gather()`
waits on a batch and always returns `TaskResult` (never raises for a
task-level failure — `ok=False` + `error` instead).

### `DistributedAlgorithm` — `flashml/algorithms/base.py`

```text
plan(dataset, storage, job_id) -> shards        # once, writes data to storage
initialize(shards, storage) -> state            # seed model state
make_tasks(state, shards, iteration) -> [payload]  # one per shard, small
reduce(state, results) -> (new_state, metrics)
converged(old_state, new_state, metrics, threshold) -> bool
finalize(state, shards, storage) -> result
```

**Design deviation from IMPLEMENTATION.md worth knowing:** the original plan
had two separate abstractions — a provider-facing `Strategy` (plan/dispatch
shape: MapReduce vs. Parameter Server vs. Parallel) and an algorithm-facing
`DistributedAlgorithm` (the math). The actual code merged them into one
`DistributedAlgorithm` per §above — a K-Means instance *is* the MapReduce
strategy, an `SklearnPartialFitAlgorithm` instance *is* the Parameter Server
strategy. This removed a layer of indirection with no loss of the "strategies
never import a provider" invariant, at the cost of the `strategies/` package
in IMPLEMENTATION.md's target layout never existing. If a future strategy
shape genuinely needs to be reused across multiple unrelated algorithms,
that's the signal to reintroduce the split — don't reintroduce it
speculatively.

### `resolve_entrypoint` — `flashml/providers/entrypoint.py`

The one place a connector is allowed to know about algorithm code: it
`importlib`-loads whatever dotted path the algorithm declared and calls it
with `(payload, storage)`. `LocalWorkerPool` calls this in-process. A remote
connector (RunPod) will need workers that can do the same resolution
*inside the deployed worker process* — see the Phase 1 requirements in
[phases/01-runpod-connector/README.md](phases/01-runpod-connector/README.md).

---

## 6. Package-by-package reference

```text
flashml/
    __init__.py                 public API surface
    engine/
        cluster.py               Cluster: provision -> run -> teardown
        job.py                   Job, JobEvent — the event/result handle
        loop.py                  run_job(): the strategy-agnostic training loop
    providers/
        base.py                  Provider/WorkerPool/Storage/Task/* dataclasses + errors
        registry.py               name -> Provider factory, @register decorator
        entrypoint.py             resolve_entrypoint(): "module:func" -> callable
        local/provider.py         LocalProvider/LocalWorkerPool (ThreadPoolExecutor)
    algorithms/
        base.py                  DistributedAlgorithm interface
        kmeans.py                 KMeansAlgorithm (MapReduce) + map_task worker fn
        sklearn_partial_fit.py   SklearnPartialFitAlgorithm (generic Parameter Server / FedAvg)
        sklearn_registry.py      name -> sklearn estimator class (keeps payloads JSON-safe)
        linear_regression.py     LinearRegression = SklearnPartialFitAlgorithm(sgd_regressor)
        linear_models.py         LogisticRegression / Perceptron / PassiveAggressive*
    storage/
        base.py                  Storage interface (re-exported from providers/base.py)
        local.py                  LocalStorage: temp-dir-backed blob store
```

---

## 7. Algorithms catalog

| Algorithm | Strategy shape | Backing math | Entrypoint |
|---|---|---|---|
| `KMeans` | MapReduce | hand-rolled Lloyd's algorithm, sklearn `kmeans_plusplus` for seeding | `flashml.algorithms.kmeans:map_task` |
| `LinearRegression` | Parameter Server (FedAvg over one local `partial_fit` step per iteration) | `sklearn.linear_model.SGDRegressor` | `flashml.algorithms.sklearn_partial_fit:map_task` |
| `LogisticRegression`, `Perceptron`, `PassiveAggressiveClassifier`, `PassiveAggressiveRegressor` | Parameter Server, same mechanism | corresponding sklearn linear estimator | same |
| `SklearnPartialFitAlgorithm` | Parameter Server (generic) | any sklearn estimator with `partial_fit` + array-valued state attrs | same |

All four "linear" algorithms are thin `__init__`-only subclasses of
`SklearnPartialFitAlgorithm` (`flashml/algorithms/linear_regression.py`,
`linear_models.py`) — adding another sklearn `partial_fit` estimator is a
~5-line subclass plus a registry entry in `sklearn_registry.py`, not a new
algorithm implementation.

`finalize()` on the Parameter Server path returns a real, usable fitted
sklearn estimator (`.predict()` works), not just a dict of numbers — see
`sklearn_partial_fit.py:143`.

---

## 8. Providers catalog

| Connector | Status | Compute model | Storage |
|---|---|---|---|
| `local` | **Done** | `ThreadPoolExecutor`, in-process | temp directory (`flashml/storage/local.py`) |
| `runpod` | Not started (Phase 1) | RunPod Flash serverless `@Endpoint` | NetworkVolume (S3-compatible) |
| `ssh` | Not started (Phase 4) | bring-your-own hosts over SSH | rsync/scp staging dir |

The `local` connector is the reference every future connector is tested
against — the tests in `tests/test_local_provider.py` are a lightweight,
inline version of the conformance suite `PROVIDERS.md` describes (that
suite doesn't exist as a reusable `flashml.testing` module yet — see
Phase 4 in the roadmap).

---

## 9. Tests & examples

```bash
# Runtime, lifecycle, and documentation-structure tests
.venv/bin/python -m pytest tests/ -q

# End-to-end runnable demo (K-Means, linear regression, logistic regression)
.venv/bin/python examples/local_kmeans_and_linear_regression.py
```

Both exercise the full stack — `Cluster` → `Provider` → `WorkerPool` →
`DistributedAlgorithm` → `Storage` — with real (if synthetic) data and assert
on numerical convergence, not just "it ran."

---

## 10. The legacy system: `legacy/coordinator/` + `apps/dashboard/`

`legacy/coordinator/server.py` (1,631 lines) is the pre-library FastAPI app: it owns
the training loop, the RunPod Flash dispatch, and the job/event API the
dashboard consumes, all fused together with no `Provider`/`Storage`
boundary. `legacy/coordinator/flash_worker/` holds the actual `@Endpoint`-decorated
worker functions per accelerator tier (`worker_gpu_ada24.py`,
`worker_cpu_large.py`, etc.) — this is the ported-from source for the RunPod
connector's Phase 1 work, and `TOOLS.md` documents the design decisions in it
(the 10MB payload cap, the multi-tier GPU fallback list, NetworkVolume
staging) that the connector must preserve.

`apps/dashboard/` (Next.js) is a complete, working dashboard that currently talks
to `legacy/coordinator/server.py`'s routes directly. It is explicitly **not**
being rewritten — Phase 3 repoints it at `flashml serve` by making that new
FastAPI layer expose the same routes the frontend already consumes.

Both are live and runnable today; they are not being deleted until the
functionality they hold is proven equivalent inside `flashml/` (coordinator
becomes a deprecation shim per IMPLEMENTATION.md, then goes away).

---

## 11. Doc map

| Question | Read |
|---|---|
| Why does this project exist, what's the pitch? | `README.md` |
| What phase are we in, what's next, what are the tasks? | `ROADMAP.md` |
| How do I write a new connector (compute backend)? | `PROVIDERS.md` |
| How do I bring a new model/framework (PyTorch, sktime, ...)? | `MODELS.md` |
| How does the RunPod Flash SDK / NetworkVolume / `runpodctl` actually work? | `TOOLS.md` |
| What does the code actually do right now? | this document |
| What was the original MVP design (partially superseded)? | `IMPLEMENTATION.md` |
