# Code Walkthrough — One Real FlashML Job

This guide follows code that exists and passes today. Planned RunPod, PyTorch,
and server behavior is deliberately excluded from the main path.

## 1. Start at the public API

Run:

```bash
.venv/bin/python examples/local_kmeans_and_linear_regression.py
```

The example creates `flashml.Cluster(provider="local", workers=4)` and calls
`Cluster.train(...)`. Public names are exported from `flashml/__init__.py`.

## 2. Resolve the provider

`flashml/engine/cluster.py:Cluster.__init__` calls
`flashml/providers/registry.py:get_provider`. Importing
`flashml.providers.local` registers `LocalProvider` under `local`.

The registry is the extension seam: a future connector implements `Provider`
and registers a factory name; the engine does not add provider-specific branches.

## 3. Provision workers and reachable storage

`Cluster.train` builds a `WorkerSpec` containing an accelerator tier and the
algorithm's dotted worker entrypoint. `LocalProvider.provision` creates a
thread-backed `LocalWorkerPool`; `LocalProvider.storage` returns temporary
filesystem storage visible to those threads.

A remote provider must preserve this contract even if its implementation is
HTTP, SSH, or a serverless endpoint.

## 4. Run the provider-independent loop

`flashml/engine/loop.py:run_job` owns the invariant execution sequence:

```text
algorithm.plan(...)          write dataset shards to Storage once
algorithm.initialize(...)    create initial model state
repeat:
  algorithm.make_tasks(...)  create small JSON-safe payloads
  pool.submit(...)            dispatch without knowing the algorithm
  pool.gather(...)            normalize worker results
  algorithm.reduce(...)       build new model state and metrics
  algorithm.converged(...)    decide whether to stop
algorithm.finalize(...)      return a usable result
```

Payloads larger than 1 MB are rejected before dispatch. Large datasets and,
in future, model checkpoints belong in `Storage`; messages carry keys.

## 5. Resolve algorithm code on a worker

Each algorithm declares an entrypoint such as
`flashml.algorithms.kmeans:map_task`. The local worker calls
`flashml/providers/entrypoint.py:resolve_entrypoint`, then invokes the function
with `(payload, storage)`.

Phase 1 must perform the same resolution inside a generic RunPod worker. A
connector must not contain `if algorithm == "kmeans"` branches.

## 6. Reduce and return a usable result

K-Means reduces partial sums/counts into centroids. The sklearn adapter reduces
worker model states using sample-weighted averaging. `finalize` returns either
the centroid result or a fitted sklearn estimator whose `predict()` works.

`Job.stream()` replays recorded iteration events. Today training is synchronous:
the work finishes inside `Cluster.train` before the `Job` is returned. Phase 3
moves execution into a background job runner without changing the algorithm or
provider contracts.

## The two abstractions to extend

### Add compute

Implement the abstract classes in `flashml/providers/base.py`:

- `Provider`: offers, provision, storage, teardown;
- `WorkerPool`: submit, gather, size;
- `TaskHandle`: wait;
- `Storage`: put, get, exists, delete_prefix.

Use `flashml/providers/local/provider.py` as the executable reference and
`docs/PROVIDERS.md` as the contract.

### Add a model or distributed shape

Implement `DistributedAlgorithm` from `flashml/algorithms/base.py`:

- plan and initialize;
- make tasks;
- reduce and detect convergence;
- finalize.

Use `kmeans.py` for MapReduce and `sklearn_partial_fit.py` for parameter-server
behavior. Read `docs/MODELS.md` before adding a new shape.

## What not to learn from

- `legacy/coordinator/` proves older RunPod/API mechanics but violates the new
  provider boundary. Port behavior from it; do not copy its architecture.
- `archive/experiments/` is historical evidence only.
- `apps/dashboard/` is a retained client, not the source of engine truth.
