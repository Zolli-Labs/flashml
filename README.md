# FlashRuntime

> **The open fault-tolerant distributed ML runtime.** FlashRuntime defines
> what a job is, how it becomes tasks, how tasks are leased to nodes, how
> progress is reported, how artifacts are committed, and how recovery happens
> when nodes disappear.

FlashRuntime is one of three components in the FlashML system by
[Zolli Labs](https://github.com/Zolli-Labs):

- **[flashnode](https://github.com/Zolli-Labs/flashnode)** — open host agent
  installed by resource contributors.
- **flashruntime** (this repo) — the open workload protocol and execution
  layer. Self-hostable: useful without the cloud.
- **flashml-cloud** (private) — the managed control plane, marketplace, and
  dashboard.

Read [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) for the full
product architecture, and [`AGENTS.md`](AGENTS.md) if you are an AI coding
agent working in this repo.

## Status

**Pre-release.** This repo was seeded (July 2026) from the working FlashML
prototype — a provider-agnostic distributed training library with a local
multi-worker backend. That code runs today (see Quickstart) and is being
evolved into the lease/heartbeat/recovery architecture described below.

## Quickstart (what works today)

```bash
uv venv && uv pip install -e ".[sklearn,dev]"
python examples/local_kmeans_and_linear_regression.py
pytest
```

```python
import flashruntime

with flashruntime.Cluster(provider="local", workers=4) as cluster:
    job = cluster.train(
        algorithm=flashruntime.algorithms.KMeans(k=5, n_shards=4),
        dataset=numpy_array,
        max_iterations=20,
    )
    for event in job.stream():
        print(event.iteration, event.metrics)
    fitted = job.result()
```

## Package layout

```
flashruntime/
├── protocol/    # versioned public schemas (scaffold — being built)
├── leases/      # lease + heartbeat semantics (scaffold)
├── checkpoint/  # checkpoint manifests and compatibility (scaffold)
├── recovery/    # failure taxonomy + recovery state machine (scaffold)
├── scheduler/   # pluggable placement interfaces (scaffold)
├── engine/      # working prototype: Cluster / Job / sync loop
├── algorithms/  # working prototype: sharded K-means, sklearn partial_fit
├── adapters/    # working prototype: provider interface + local backend
└── storage/     # working prototype: artifact storage interface
```

The `engine`/`algorithms`/`adapters`/`storage` packages are the functioning
prototype carried over from the original FlashML library. The scaffold
packages are the target architecture; prototype code migrates into it as the
protocol lands.

## The dependency rule

`flashruntime` owns the public protocol and imports **neither** application
repo. `flashnode` and `flashml-cloud` both depend on `flashruntime`; they
never import each other.

## License

[Apache-2.0](LICENSE). Contributions via Developer Certificate of Origin
(`git commit -s`).
