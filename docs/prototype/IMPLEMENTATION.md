# FlashML MVP Implementation Guide

> **Status:** M0 (this doc's first milestone) is built — see
> [ARCHITECTURE.md](ARCHITECTURE.md) for what actually exists in the
> code, and [ROADMAP.md](ROADMAP.md) for the living, phase-by-phase
> plan going forward (M1/RunPod and generalized model support are the current
> phases). This document remains as the original MVP design rationale; a few
> details below (the separate `strategies/` package, exact file layout) were
> simplified during implementation — ARCHITECTURE.md documents the deviation.
> Full documentation map: [INDEX.md](INDEX.md).

## MVP Goal

Turn the existing FlashML playground into an installable library:

> **`pip install flashml` → connect to any GPU provider → run a distributed training job through one interface.**

The MVP proves the core claim — *provider-agnostic distributed training* — with the smallest honest demonstration:

- One Python package (`flashml`) with a stable public API
- One provider contract (`Provider`) that every backend implements
- **Two working connectors**: `local` (thread pool, zero-setup) and `runpod` (real serverless GPUs — ported from the existing code)
- The three existing strategies (MapReduce, Parameter Server, Embarrassingly Parallel) running unchanged on **both** connectors
- `flashml serve` exposing the job API the existing dashboard already consumes

The acceptance test for the whole MVP is one line of diff:

```python
flashml.Cluster(provider="local", workers=8)    # runs
flashml.Cluster(provider="runpod", workers=8)   # runs, same job, real GPUs
```

---

## Where the Pieces Come From

Almost everything the MVP needs already exists — it's currently fused together inside the coordinator. The MVP is an **extraction and interface-hardening exercise**, not a rewrite:

| Existing code | Becomes |
|---|---|
| `legacy/coordinator/server.py` — training loop, reduce step, job state, FastAPI routes | `flashml/engine/` (loop, scheduler, job store) + `flashml/server/` (the FastAPI layer, now optional) |
| `legacy/coordinator/server.py` — `map_reduce`, `gradient_sync`, `parallel_search` branches | `flashml/strategies/` — three `Strategy` plugins behind one interface |
| `legacy/coordinator/flash_worker/` — `worker.py` + GPU/CPU tier variants | `flashml/providers/runpod/` — the RunPod connector |
| Coordinator's ThreadPoolExecutor simulation paths | `flashml/providers/local/` — the Local connector (a first-class citizen now, not a simulation) |
| NetworkVolume shard staging (see [TOOLS.md](TOOLS.md)) | `flashml/storage/` — `Storage` interface; RunPod implements it with NetworkVolume, Local with a temp dir |
| `apps/dashboard/` | Unchanged — points at `flashml serve` instead of `legacy/coordinator/server.py` |

---

## Target Package Layout

```text
flashml/
    __init__.py            # public API: Cluster, Job, connect(), algorithms
    engine/
        job.py             # Job, JobStore, events
        loop.py            # the strategy-agnostic training loop
        scheduler.py       # dispatch tasks to a WorkerPool, gather results, retries
    strategies/
        base.py            # Strategy interface
        mapreduce.py       # K-Means (from coordinator map_reduce path)
        parameter_server.py# Linear Regression gradient sync
        parallel.py        # hyperparameter search
    algorithms/
        base.py            # DistributedAlgorithm plugin interface
        kmeans.py
        linear_regression.py
    providers/
        base.py            # Provider, WorkerSpec, Offer, WorkerPool, TaskHandle
        registry.py        # name -> connector ("runpod", "local", ...)
        local/
            provider.py    # thread/process pool workers
        runpod/
            provider.py    # Flash @Endpoint provisioning + runsync dispatch
            tiers.py       # normalized accelerator -> GpuGroup / CpuInstanceType
            storage.py     # NetworkVolume-backed Storage
    storage/
        base.py            # Storage interface: put/get/exists shard blobs
        local.py
    server/
        app.py             # FastAPI: POST /api/train, GET /api/jobs/{id} (existing routes)
        cli.py             # `flashml serve`

coordinator/               # shrinks to a thin shim, then deleted post-MVP
frontend/                  # dashboard, unchanged API contract
docs/
    PROVIDERS.md           # connector spec (the contract new providers implement)
```

---

## The Public API

```python
import flashml

# 1. Connect — pick a provider by name, normalized resources
cluster = flashml.Cluster(
    provider="runpod",           # any registered connector
    workers=8,
    accelerator="gpu-24gb",      # normalized tier; connector maps it (see PROVIDERS.md)
    credentials=...,             # or env vars, per connector convention
)

# 2. Train — strategy + algorithm + data
job = cluster.train(
    strategy="mapreduce",
    algorithm=flashml.algorithms.KMeans(k=5, max_iterations=10),
    dataset="s3://bucket/embeddings.parquet",   # or local path / ndarray
)

# 3. Observe — stream events (the same events the dashboard renders)
for event in job.stream():
    print(event.iteration, event.metrics, event.worker_states)

result = job.result()            # final model state + metrics
cluster.close()                  # teardown: nothing left renting GPUs
```

Design rules the MVP must not break:

- **Strategies never import a provider.** They see only `WorkerPool.submit(task) -> TaskHandle` and `Storage`.
- **Providers never see algorithms.** They move opaque task payloads and blobs.
- **Small state over the wire, big data through Storage.** This is the lesson already learned on RunPod (10MB payload cap, no re-uploading shards per iteration) — it is now a library-level invariant that applies to every connector.
- **`Cluster` is a context manager.** GPUs cost money by the minute; teardown must be impossible to forget (`with flashml.Cluster(...) as cluster:`).

---

## Core Interfaces (MVP versions)

### Provider (full spec in [PROVIDERS.md](PROVIDERS.md))

```python
class Provider(ABC):
    name: str

    def offers(self, spec: ResourceSpec) -> list[Offer]:
        """Availability + pricing for a normalized resource request."""

    def provision(self, spec: WorkerSpec, count: int) -> WorkerPool:
        """Bring up `count` workers able to execute FlashML tasks."""

    def storage(self) -> Storage:
        """Blob store reachable by this provider's workers (shards, state)."""

    def teardown(self, pool: WorkerPool) -> None:
        """Release everything. Idempotent. Must succeed on partial provisions."""


class WorkerPool(ABC):
    def submit(self, task: Task) -> TaskHandle: ...
    def gather(self, handles: list[TaskHandle], timeout: float) -> list[TaskResult]: ...
```

### Strategy

```python
class Strategy(ABC):
    def plan(self, dataset: Dataset, config: JobConfig) -> list[Shard]:
        """Partition once; shards go to provider Storage, not payloads."""

    def make_tasks(self, state: ModelState, iteration: int) -> list[Task]: ...

    def reduce(self, results: list[TaskResult]) -> tuple[ModelState, Metrics]: ...

    def converged(self, old: ModelState, new: ModelState, metrics: Metrics) -> bool: ...
```

The engine's loop is then strategy- and provider-agnostic:

```python
def run(job, strategy, pool, storage):
    shards = strategy.plan(job.dataset, job.config)     # written to storage once
    state = strategy.initialize(shards, job.config)

    for iteration in range(job.config.max_iterations):
        tasks = strategy.make_tasks(state, iteration)
        results = pool.gather([pool.submit(t) for t in tasks], job.config.task_timeout)
        state, metrics = strategy.reduce(results)
        job.emit(iteration, metrics)                     # dashboard event stream
        if strategy.converged(...):
            break

    return strategy.finalize(state, shards)
```

---

## Milestones

### M0 — Library skeleton + Local connector — **Done**

*Everything through the interface, nothing on a cloud yet.*

- [x] Create the `flashml/` package with a stable public API (`Cluster`, `Job`, `algorithms`)
- [x] Define `Provider`, `WorkerPool`, `Storage`, `DistributedAlgorithm` interfaces (the `Strategy`/`DistributedAlgorithm` split below was merged into one interface during implementation — see `docs/ARCHITECTURE.md` §6)
- [x] Implement `providers/local/` on a thread pool — the connector every algorithm is tested against
- [x] Port K-Means (MapReduce) and a generic sklearn `partial_fit` bridge covering Linear/Logistic Regression, Perceptron, Passive-Aggressive (Parameter Server)
- [x] **Exit criterion met:** all algorithms run via `flashml.Cluster(provider="local")`, verified by `tests/` and `examples/`

### M1 — RunPod connector

*The first real GPU marketplace behind the same interface.*

- Port `legacy/coordinator/flash_worker/` into `providers/runpod/`: `provision` deploys/attaches the Flash `@Endpoint` worker pool, `submit` calls `/runsync`, `storage()` wraps the NetworkVolume (S3-compatible API), `teardown` scales to zero.
- `tiers.py`: map normalized accelerators (`gpu-24gb`, `cpu-small`, …) to `GpuGroup`/`CpuInstanceType`, keeping the multi-tier fallback trick (`[AMPERE_24, ADA_24]`) documented in TOOLS.md.
- Handle credentials via `RUNPOD_API_KEY` env var + explicit `credentials=` override.
- **Exit criterion:** the M0 acceptance script runs on RunPod by changing only `provider="runpod"`; K-Means executes on real Flash nodes as it does today.

### M2 — `flashml serve` + dashboard reattach

*The observability story survives the refactor.*

- `flashml/server/app.py` re-exposes the exact routes the frontend consumes today (`POST /api/train`, `GET /api/jobs/{id}`, worker/iteration events) on top of the engine's event stream.
- `flashml serve` CLI entry point; `legacy/coordinator/server.py` becomes a deprecation shim that imports it.
- **Exit criterion:** the existing Next.js dashboard renders live jobs from both connectors with no frontend changes.

### Phase 4 — Prove the plug-in claim

*A third connector written by following the docs alone.*

- Write `PROVIDERS.md` conformance checklist into an actual test suite (`flashml.testing.provider_conformance`) that any connector must pass.
- Implement the `ssh` connector (bring-your-own-machines: a list of hosts, workers launched over SSH, storage over rsync/scp) **using only the public interface and the spec doc** — this is the dogfood test that the abstraction is real.
- **Exit criterion:** conformance suite green on `local`, `runpod`, `ssh`; a demo runs the same job on all three.

---

## MVP Checklist

Must have:

- [x] `flashml` package installable (`pip install -e .`) — not yet committed to git, see `ROADMAP.md`
- [x] `Provider` / `WorkerPool` / `Storage` / `DistributedAlgorithm` interfaces
- [x] Local connector
- [ ] RunPod connector (provision, dispatch, NetworkVolume storage, teardown) — current phase, see `ROADMAP.md`
- [x] MapReduce and Parameter Server strategies on the new interfaces (Parallel/hyperparameter-search not yet ported)
- [x] Shard-once-to-storage invariant enforced at the engine level
- [x] Teardown guaranteed via context manager + idempotent `teardown()`
- [ ] `flashml serve` + dashboard compatibility
- [ ] Provider conformance test suite (informal version exists in `tests/`, not packaged)
- [x] `PROVIDERS.md` connector spec

Nice to have:

- [ ] SSH connector (Phase 4 — do it if time allows; it's the strongest demo)
- [ ] `flashml.offers("gpu-24gb")` — cross-provider price/availability comparison
- [ ] Worker retry / straggler re-dispatch in the scheduler
- [ ] Cost reporting per job (RunPod: `runpodctl billing serverless`)
- [ ] Export trained state (centroids/weights) as artifacts

Explicitly **out of scope** for MVP (roadmap, not now):

- PyTorch DDP / NCCL strategies (MVP strategies are HTTP-coordinated; direct worker-to-worker networking is a different provisioning contract)
- Vast.ai / Lambda / CoreWeave connectors
- Cost-aware automatic provider routing
- Multi-provider single job (spanning one job across two clouds)

---

## Demo Script

1. `pip install flashml` — one library, no cluster to stand up.
2. Show the 10-line training script with `provider="local"` — it runs instantly on a laptop.
3. Change one word: `provider="runpod"`. Same script, now on real serverless GPUs.
4. Open the dashboard — watch shards, workers, and convergence live, identical view for both providers.
5. (If Phase 4 landed) Change the word again: `provider="ssh"` with two rented boxes — same job, third backend.
6. Close: "Every GPU marketplace is a connector. The training code never changes."

---

## Core Story

Scikit-learn made machine learning simple on one machine. The GPU rental market made compute abundant but fragmented — every provider is another SDK, another glue layer, another lock-in.

FlashML is the connection layer: distributed training strategies above one Provider interface, GPU marketplaces below it. Write the job once; the provider is a config value.
