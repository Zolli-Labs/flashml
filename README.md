# FlashRuntime

> **The open fault-tolerant distributed ML runtime.** FlashRuntime *operates*
> your training job — it never rewrites it. You keep the model, the training
> loop, the loss, the data, and the framework you already have. FlashRuntime
> wraps the reliability layer around them: it launches your command, injects
> the environment it promised, tracks your metrics, validates your checkpoints,
> **recovers from crashes in one call**, and collects your artifacts. The
> distributed computation itself is always done by established libraries
> (PyTorch DDP/FSDP2/torchrun, Ray, Hugging Face, scikit-learn).

FlashRuntime is one of three components in the FlashML system by
[Zolli Labs](https://github.com/Zolli-Labs):

- **[flashnode](https://github.com/Zolli-Labs/flashnode)** — open host agent
  installed by resource contributors.
- **flashruntime** (this repo) — the open workload protocol and execution
  layer. Self-hostable: useful without the cloud.
- **flashml-cloud** (private) — the managed control plane, marketplace, and
  dashboard.

Read [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) for the full product
architecture, and [`AGENTS.md`](AGENTS.md) if you are an AI coding agent working
in this repo.

## Install

The core is deliberately tiny — `pip install flashruntime` brings **only
pydantic**. Every core module (the `flash.submit()` SDK, planner, leases,
checkpoints, recovery) works with zero infrastructure:

```bash
pip install flashruntime
```

Torch is **not** a dependency. FlashRuntime *launches* PyTorch; it never imports
it. Install it yourself to run the DDP example — a CPU-only build is enough,
since DDP works over the `gloo` backend with no GPU:

```bash
pip install torch          # CPU build is fine
```

## 60-second run

`flash.submit()` operates any command. The one convention your script owes
FlashRuntime is to write a `metrics.json` (a flat JSON object) into its working
directory; FlashRuntime collects it and records it as a trial:

```python
import flashruntime as flash

run = flash.submit(
    flash.CommandWorkload(
        command="python train.py --epochs 5",
        source=flash.Source(path="~/my-project"),
        outputs=flash.OutputSpec(collect=["metrics.json"]),
    ),
    max_restarts=2,   # a crashed attempt relaunches from the last VALID
                      # checkpoint — up to twice (see below)
    watch=True,       # opens the live run page and prints its URL
)

print(run.state.value)   # "SUCCEEDED" (or "FAILED")
print(run.trials)        # [{'accuracy': 0.91, ...}]  — parsed from metrics.json
print(run.artifacts)     # [PosixPath('.../metrics.json'), ...]
print(run.viewer_url)    # http://127.0.0.1:<port> — the live page
```

`command` is `shlex`-split (there is no shell — for a pipe, pass
`command="bash -c '...'"`), and `source` is a `flash.Source`, so `~` is expanded
for you. The same thing from the shell:

```bash
flashruntime submit "python train.py --epochs 5" \
    --source ~/my-project --max-restarts 2 --watch
```

A script that already reads its hyperparameters from `argparse` and writes a
small JSON of results needs **zero FlashRuntime imports** to be operated. The
contract at the boundary is thin on purpose: arguments in, `metrics.json` out.

## One import, fault tolerant

`max_restarts=N` is the automatic recovery budget. On a **FAILED** attempt
FlashRuntime turns the exit into failure signals, classifies them
(`recovery.classify`), and consults a *versioned, deterministic* policy
(`recovery.decide`): a deterministic application bug fails fast — a retry would
only re-hit it — while a transient failure relaunches the same spec from the
job-scoped checkpoint. Same failure + same policy version ⇒ same action, every
time. **No LLM in the loop.** Every decision is recorded on the run as
`FAILURE_CLASSIFIED` / `RECOVERY_ACTION_SELECTED` events.

Kill-and-resume is proven end to end. `tests/test_examples_e2e.py` kills a
2-process DDP run mid-training and asserts the resumed run's final loss
matches a never-crashed baseline to 1e-6 — recovery restores only a verified,
topology-compatible checkpoint (parts-first / manifest-last: a half-written
checkpoint can never look valid).

For a script you *are* willing to touch, `import flashruntime.torch as ft` adds
`ft.prepare` / `ft.checkpoint` / `ft.log_metrics`: torch's own DDP wrapped under
the same checkpoint contract, with correct per-rank device placement, so a
killed run resumes with a loss curve indistinguishable from the uninterrupted
one (1e-6 in the e2e). It is optional sugar on the launch-only
contract, never required.

**GPU-validated (2026-07-23).** The CUDA paths — `nccl` DDP, per-rank device
placement, and a GPU kill-and-resume — are validated end to end on real
hardware: **2×NVIDIA GeForce RTX 4090** (RunPod community cloud), torch
2.7.1+cu128, CUDA 12.8. Two runs, **$0.0725 total**. See `tests/test_gpu_e2e.py`
and the harness `scripts/runpod_gpu_e2e.py`.

## Watch it run

Pass `watch=True` (the default at an interactive terminal, off in pipes/CI) and
`flash.submit()` opens a live run page in your browser and prints its URL. It
draws the run's topology, its loss curve, its verified checkpoints, and every
recovery decision, refreshing every couple of seconds — served entirely from a
loopback server with **no external assets**, so it renders with the network cut.
These docs are served from that same viewer at `/docs`.

> _screenshot: run any `flashruntime submit … --watch` to see the live page._

## Documentation

A PyTorch-style docs site is built from [`docs/site/`](docs/site/) by
`scripts/build_docs.py` (every byte inline — no CDN, web font, or remote image).
It is served offline by the viewer at `/docs`, and deploys to GitHub Pages on
release:

- **<https://zolli-labs.github.io/flashruntime/>** — _live after the first
  Pages deploy._

Start with [Get started](docs/site/get-started.md) (install → first job → first
DDP run on CPU). The strategy planner's code walkthrough lives in
[`docs/planner/README.md`](docs/planner/README.md).

## Benchmarks

An honest benchmark suite (`python -m benchmarks run`) records every result with
the host that produced it and reports missing comparators as skipped rows, never
faked. A measured baseline is committed under
[`benchmarks/results/`](benchmarks/results/). On **Apple M4 (CPU)**:

| Scenario | Measured | Read it honestly |
|---|---|---|
| Launch overhead | **+0.04 s** | `flash.submit` vs bare `torchrun`, paired. A one-time fixed cost, not per-step. |
| Checkpoint cost | **−1.2 ms / checkpoint** (p90 6.2 ms) | A few-KB state dict; the median delta is *below* the run-to-run noise floor at this size (it goes slightly negative), so checkpoints are effectively free here — the p90 bounds it. A larger model surfaces a real positive cost. |
| Auto-resume | **40 steps not recomputed** | The size-*independent* guarantee: resume never re-does work past the last valid checkpoint. |
| Adoption | **7 lines** | To make a vanilla script framework-ready (difflib vs each project's own docs). |
| Resilience | **5/5 faults handled · 20/20 integrity under `kill -9`** | Every fault type routed to the right typed recovery; a checkpoint survives a mid-write `SIGKILL` on 20/20 kills (naive `torch.save`: 20/20 corrupted). 16-trial storm, half crash-armed: 16/16 completed, all 8 crashed trials auto-resumed, 0 manual. Dead-worker MTTD **3.0 s**, MTTR **~3.5 ms**. |

The *timing* rows are smoke-scale on one host: the models are tiny, so the
*wall-clock* saved by recovery is at the noise floor here. The figure that
scales is `steps_not_recomputed`, not seconds — the seconds-saved grows with the
compute between the last checkpoint and the crash (negligible at smoke size,
hours on a real job). The resilience counts are exact, not smoke-diluted: they
are *counted* from real fault injection (`kill -9` in the checkpoint-write
window, a storm of crash-armed trials, a killed lease worker), and the full
Resilience table with per-scenario method lives in
[the benchmarks page](docs/site/benchmarks.md). Full provenance and caveats live
in the result file.

---

## Also: plan a job (no cluster required)

Before you launch, `flash.plan()` turns model + hardware + budget + deadline
into a ranked, explained execution plan — deterministic, framework-import-free:

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
(torchrun, DDP/FSDP2, transformers+peft, bitsandbytes, PyTorch DCP) and their
roles, the knobs, and *why* — including why the alternatives lost. Details:
[`docs/planner/README.md`](docs/planner/README.md).

## Also: leases, checkpoints, recovery

The reliability core is an embeddable library, not a service you must run:

```python
from flashruntime.leases import LeaseManager
from flashruntime.recovery import FailureSignals, classify, decide
from flashruntime.protocol.v1alpha1 import TaskSpec

mgr = LeaseManager()
mgr.add_task(TaskSpec(task_id="trial-01", job_id="sweep", commit_key="sweep/trial-01"))
lease = mgr.claim(node_id="laptop-1")            # pull, never push
mgr.heartbeat(lease.lease_id)                    # prove liveness
mgr.complete(lease.lease_id, output_sha256="…")  # first valid commit wins

decision = decide(classify(FailureSignals(heartbeat_lost=True)),
                  mode="independent_tasks")      # → RETRY_TASK, cordon node
```

The FlashRuntime service exposes these same objects over HTTP; FlashNode's
device executor is their remote client. The lease coordinator's durable
`SqliteLeaseStore` means in-flight leases survive a coordinator restart.

## Testing

```bash
pytest                  # unit tests — pure Python, no infrastructure
pytest -m integration   # opt-in: Docker / Kubernetes / MinIO environments
```

Integration tests live in [`tests/integration/`](tests/integration/README.md)
and skip themselves (with instructions) when their environment is absent. The
one GPU test (`tests/test_gpu_e2e.py`) is CUDA-gated and skips without a GPU. The
local kind + KubeRay + MinIO stack is owned by the workspace Makefile, not this
repo — the library stays `pip install`-clean.

## Package layout

**Pure-Python core** — `pip install flashruntime` brings pydantic and nothing
else; every module below works with zero infrastructure:

```
flashruntime/
├── protocol/    # versioned public schemas (v1alpha1: JobSpec, Event, Lease,
│                #   TaskAttempt, CheckpointManifest, RecoveryDecision, plans)
├── sdk.py       # flash.submit(CommandWorkload) → Run: compile→launch→wait→
│                #   collect, with the automatic max_restarts recovery loop
├── workloads/   # CommandWorkload: framework-neutral "run this command"
├── integrations/# one-file adapters (sklearn / pytorch DDP / HF) — no framework
│                #   imports at module level; convention is CLI-flags-in/JSON-out
├── torch/       # optional in-script helper: ft.prepare / checkpoint / log_metrics
├── planner/     # strategy planner: estimators, curated menus, explained selector
├── leases/      # Mode A core: claim / heartbeat / expiry sweep / idempotent
│                #   commit; InMemory + SQLite stores (restart-durable)
├── checkpoint/  # manifest catalog: parts-first/manifest-last, validation ladder
├── recovery/    # failure-signal classifier + versioned deterministic policy
└── viewer/      # stdlib live run page + the built docs site (zero external assets)
```

**Infrastructure integrations** (opt-in extras, never core imports):

```
├── service/     # [service] the coordinator: job submission + task expansion,
│                #   lease/node/checkpoint HTTP endpoints, local artifact hosting,
│                #   join codes, SQLite ledger, dashboard at GET /, the CLI
├── launchers/   # LocalProcessLauncher (the concrete local launcher today)
├── strategies/  # CommandWorkload → LaunchSpec compilation
├── recipes/     # coordinator-side WorkloadRecipe registry (command jobs)
├── backends/    # [k8s] ExecutionBackend contract + KubeRay backend (Mode B)
└── artifacts/   # [artifacts]/[oss] MinIO/S3-compatible + Alibaba OSS stores

flashml_workloads/   # runnable task modules (pure stdlib/sklearn):
│                    #   sklearn_trial, kmeans_shard + kmeans_driver,
│                    #   sgd_trainer (checkpointable, bit-identical resume)
```

Still to come (each lands with its vertical slice, no empty scaffolds):
multi-node DDP rendezvous (`nnodes > 1` raises `NotImplementedError` today);
FlashNode's argv runner so service-side command jobs execute remotely;
checkpoint-manifest persistence (catalog is in-memory; checkpoint *files* are
durable); Stage-8 ledger metrics (MTTD/MTTR/goodput); `flash.run(plan)` wiring;
the cloud stage (Postgres, SSE, HF Trainer + PEFT LoRA recipes).

Four orthogonal axes keep the architecture honest: **providers** get machines,
**launchers** start processes, **strategies** configure execution, **recipes**
integrate user code. Hugging Face lives in recipes — it is the workload layer,
not an execution backend. The planner never imports framework code; it emits a
backend-neutral `StrategyPlan` that strategy compilers translate.

## What FlashRuntime builds on (and what it owns)

| Layer | Libraries reused | What FlashRuntime adds |
|---|---|---|
| ML math & models | PyTorch, HF Transformers + PEFT, bitsandbytes, scikit-learn | Nothing — untouched, operated by recipes |
| Distribution strategies | DDP, FSDP2 (`fully_shard`); DeepSpeed ZeRO later | The *choice* and its explanation, compiled from a StrategyPlan |
| Launching | torchrun / Torch Elastic; Ray Core on clusters | Checkpoint-aware restart around torchrun; the **lease/heartbeat protocol** for machines outside any cluster — the layer with no existing library |
| Checkpointing | PyTorch Distributed Checkpoint (parallel save/load, resharding) | The manifest **catalog**: written-last commit, validation status, compatibility, recovery-time selection |
| Infrastructure | Kubernetes + KubeRay today; SkyPilot / Slurm / provider APIs later | Provider adapters + one cross-environment job model |

Deliberately not built on: HF Accelerate (overlaps torchrun and strategy
config — user scripts that use it still work via the launch-only contract) and
Ray Train (mid V1→V2 migration).

## The dependency rule

`flashruntime` owns the public protocol and imports **neither** application
repo. `flashnode` and `flashml-cloud` both depend on `flashruntime`; they never
import each other.

## Contributing, security, changelog

- [`CONTRIBUTING.md`](CONTRIBUTING.md) — dev setup, the test/doc/audit gates, and
  the Developer Certificate of Origin (`git commit -s`).
- [`SECURITY.md`](SECURITY.md) — how to report a vulnerability.
- [`CHANGELOG.md`](CHANGELOG.md) — release notes ([Keep a Changelog](https://keepachangelog.com/)).

## License

[Apache-2.0](LICENSE). Contributions via Developer Certificate of Origin
(`git commit -s`).
