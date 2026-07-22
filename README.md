# FlashRuntime

> **The open fault-tolerant distributed ML runtime.** FlashRuntime takes an
> ML job from *submitted* to *verifiably completed* on unreliable machines:
> it defines what a job is, how it becomes tasks, how tasks are leased to
> nodes, how progress is reported, how artifacts and checkpoints are
> committed, and how recovery happens when nodes disappear — and a
> **strategy planner** that turns model + hardware + budget + deadline into
> a ranked, explained execution plan. It plans, launches, observes, and
> recovers; the actual distributed computation is always done by established
> libraries (PyTorch DDP/FSDP2/torchrun/DCP, Ray, Hugging Face).

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
agent working in this repo. The strategy planner's code walkthrough lives
in [`docs/planner/`](docs/planner/README.md); docs for the pre-K8s
prototype engine are archived in
[`docs/archive-prototype/`](docs/archive-prototype/README.md).

## Status

**Pre-release.** Two working generations coexist here:

- The original prototype (local multi-worker training library — see
  Quickstart) that seeded the repo.
- The July 2026 POC layer: the versioned `protocol/v1alpha1`, the KubeRay
  execution backend (JobSpec → ephemeral Ray cluster), the artifact store
  (MinIO/S3-compatible + Alibaba OSS), the FastAPI service + event ledger,
  and the sharded K-Means Ray workload — proven end-to-end with a
  kill-a-worker recovery demo.
- The **strategy planner** (July 2026): `flash.plan()` / `flashruntime plan`
  — deterministic, explainable strategy selection for transformer
  fine-tuning (full/LoRA/QLoRA), generic PyTorch training, classical ML,
  and independent task sets. No cluster required. See
  [`docs/planner/README.md`](docs/planner/README.md) for the architecture
  and code walkthrough.
- The **local reliability loop** (July 2026): the lease coordinator over
  HTTP (claim/heartbeat/idempotent *validated* commit), durable SQLite
  lease state (in-flight leases survive coordinator restarts), a minimal
  node registry with optional join codes, coordinator-hosted local
  artifacts, per-task **checkpoint manifests over the wire**
  (parts-first/manifest-last), three runnable workloads (hyperparameter
  search, sharded K-means, checkpointable SGD trainer with bit-identical
  resume), and a built-in dashboard at `GET /`. Proven end to end by the
  workspace `e2e/` suite: kill-a-machine sweep recovery, distributed
  K-means convergence, and cross-machine training resume.

Active work follows a staged rebuild (Mode A lease runtime first, then the
planner, then checkpointed training recovery) — see
[`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) §10 and ADR-0003
([`docs/adr/0003-reliability-runtime-first-planner-second.md`](docs/adr/0003-reliability-runtime-first-planner-second.md)).

## Quickstart 1 — plan a job (no cluster required)

```bash
uv venv && uv pip install -e ".[dev]"
flashruntime plan examples/plan-qwen7b-lora.yaml
python examples/plan_quickstart.py
```

```python
import flashruntime as flash

report = flash.plan(flash.PlanRequest(
    workload=flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="lora",
                                       train_tokens_m=25),
    resources=flash.Resources(gpus=4, gpu_type="RTX4090",
                              hourly_cost_usd_per_gpu=0.44),
    objective=flash.Objective(mode="balanced", max_cost_usd=20,
                              deadline_minutes=240),
))
print(flash.render(report))
# → SELECTED: ddp + lora(r=16), 4 workers, torchrun, 21.6 GB/GPU, ~96 min,
#   $2.82 — plus every rejected candidate with its arithmetic
```

The report names the distributed method, the libraries it is built from
(torchrun, DDP/FSDP2, transformers+peft, bitsandbytes, PyTorch DCP) and
their roles, the knobs, and *why* — including why the alternatives lost.
Details: [`docs/planner/README.md`](docs/planner/README.md).

## Quickstart 2 — manage work: leases, checkpoints, recovery

The reliability core is an embeddable library, not a service you must run:

```python
from flashruntime.leases import LeaseManager
from flashruntime.checkpoint import CheckpointCatalog
from flashruntime.recovery import FailureSignals, classify, decide
from flashruntime.protocol.v1alpha1 import TaskSpec

mgr = LeaseManager()
mgr.add_task(TaskSpec(task_id="trial-01", job_id="sweep", commit_key="sweep/trial-01"))
lease = mgr.claim(node_id="laptop-1")        # pull, never push
mgr.heartbeat(lease.lease_id)                # prove liveness
mgr.complete(lease.lease_id, output_sha256="…")  # first valid commit wins

decision = decide(classify(FailureSignals(heartbeat_lost=True)),
                  mode="independent_tasks")  # → RETRY_TASK, cordon node
```

The FlashRuntime service exposes these same objects over HTTP; FlashNode's
device executor is their remote client.

## Quickstart 3 — prototype execution engine

```bash
uv pip install -e ".[sklearn,dev]"
python examples/local_kmeans_and_linear_regression.py
```

## Bring your own code

FlashRuntime operates *your* training job — no rewrite. Run an existing
repository locally with `flash.submit()` (sklearn sweeps, PyTorch DDP, Hugging
Face), or compile it to a JobSpec for a coordinator:

- **Bring your own code** (sklearn / PyTorch / Hugging Face): [docs/guides/bring-your-code.md](docs/guides/bring-your-code.md)

## Testing

```bash
pytest                  # unit tests — pure Python, no infrastructure
pytest -m integration   # opt-in: Docker / Kubernetes / MinIO environments
```

Integration tests live in [`tests/integration/`](tests/integration/README.md)
and skip themselves (with instructions) when their environment is absent.
The local kind + KubeRay + MinIO stack is owned by the workspace Makefile,
not this repo — the library stays `pip install`-clean.

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

**Pure-Python core** — `pip install flashruntime` brings pydantic and
nothing else; every module below works with zero infrastructure:

```
flashruntime/
├── protocol/    # versioned public schemas (v1alpha1: JobSpec, Event, Lease,
│                #   TaskAttempt, CheckpointManifest, RecoveryDecision, plans)
├── planner/     # strategy planner: catalog, memory/comm/time estimators,
│                #   curated candidate menus, deterministic selector, explain
├── leases/      # Mode A core: claim / heartbeat / expiry sweep / idempotent
│                #   commit; InMemory + SQLite stores (restart-durable)
├── checkpoint/  # manifest catalog: parts-first/manifest-last commit,
│                #   validation ladder, topology-compatible selection
└── recovery/    # failure-signal classifier + versioned deterministic policy
```

**Infrastructure integrations** (opt-in extras, never core imports):

```
├── service/     # [service] the coordinator: job submission + task expansion
│   │            #   (hyperparameter_search, sharded_kmeans), lease + node +
│   │            #   checkpoint HTTP endpoints, local artifact hosting with
│   │            #   size caps + commit-time sha256 validation, join codes,
│   │            #   SQLite ledger, dashboard at GET /
├── backends/    # [k8s] ExecutionBackend contract + KubeRay backend (Mode B)
├── artifacts/   # [artifacts]/[oss] MinIO/S3-compatible + Alibaba OSS stores
├── engine/      # [prototype] pre-K8s local training engine
├── algorithms/  #   ⤷ sharded K-means, sklearn partial_fit
└── deploy/      # container image definitions (not part of the package)

flashml_workloads/   # runnable task modules (pure stdlib/sklearn):
│                    #   sklearn_trial, kmeans_shard + kmeans_driver,
│                    #   sgd_trainer (checkpointable, bit-identical resume)
```

Still to come (each lands with its vertical slice, no empty scaffolds):
`strategies/` (StrategyPlan → torchrun/DeepSpeed config compilers),
`launchers/`, `recipes/` (HF Trainer + PEFT LoRA riding the same
checkpoint contract sgd_trainer proves), manifest persistence (catalog is
in-memory; checkpoint *files* are durable), Stage-8 ledger metrics
(MTTD/MTTR/goodput), Postgres state, and SSE event streaming.

Four orthogonal axes keep the architecture honest: **providers** get
machines, **launchers** start processes, **strategies** configure execution,
**recipes** integrate user code. Hugging Face lives in recipes — it is the
workload layer, not an execution backend. The planner never imports
framework code; it emits a backend-neutral `StrategyPlan` that strategy
compilers translate.

## What FlashRuntime builds on (and what it owns)

| Layer | Libraries reused | What FlashRuntime adds |
|---|---|---|
| ML math & models | PyTorch, HF Transformers + PEFT, bitsandbytes, scikit-learn | Nothing — untouched, wrapped by recipes |
| Distribution strategies | DDP, FSDP2 (`fully_shard`); DeepSpeed ZeRO later | The *choice* and its explanation, compiled from a StrategyPlan |
| Launching | torchrun / Torch Elastic (Mode B); Ray Core on clusters (Mode A) | Checkpoint-aware restart around torchrun; the **lease/heartbeat protocol** for machines outside any cluster — the layer with no existing library |
| Checkpointing | PyTorch Distributed Checkpoint (parallel save/load, resharding) | The manifest **catalog**: written-last commit, validation status, compatibility, recovery-time selection |
| Infrastructure | Kubernetes + KubeRay today; SkyPilot / Slurm / provider APIs later | Provider adapters + one cross-environment job model |

Deliberately not built on: HF Accelerate (overlaps torchrun and strategy
config — user scripts that use it still work via the launch-only contract)
and Ray Train (mid V1→V2 migration).

## The dependency rule

`flashruntime` owns the public protocol and imports **neither** application
repo. `flashnode` and `flashml-cloud` both depend on `flashruntime`; they
never import each other.

## License

[Apache-2.0](LICENSE). Contributions via Developer Certificate of Origin
(`git commit -s`).
