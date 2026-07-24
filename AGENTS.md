# AGENTS.md — flashruntime

Context for AI coding agents (Claude Code, Codex) working in this repository.

## What this repo is

The **open-source fault-tolerant distributed ML runtime** of the FlashML
system (Zolli Labs). It owns the public protocol: job specs, task graphs,
leases, heartbeats, checkpoint manifests, failure taxonomy, recovery state
machine, adapters, CLI/SDK. Full product context: `docs/SYSTEM_OVERVIEW.md`.

Sibling repos (cloned side-by-side under `~/Work/Zolli-Labs/`):
- `../flashnode` — public host agent; depends on this repo.
- `../flashml-cloud` — private managed control plane; depends on this repo.

## Hard rules

1. **Dependency direction:** this repo imports NOTHING from `flashnode` or
   `flashml-cloud`. Ever. They import us.
2. **This repo goes public at launch** (Apache-2.0). No secrets, no keys, no
   private business logic, no references to private-repo internals in code or
   commit history. Secrets only via `.env` (gitignored).
3. **Schemas are versioned.** Any wire-visible message or spec carries a
   schema version. Security-relevant fields fail closed.
4. **The runtime must stay useful without the cloud** — self-hosted local
   coordinator is a first-class mode, not a demo shim.
5. Recovery actions are typed, deterministic, and logged. No LLM-driven
   recovery decisions.
6. **Four orthogonal axes** (ADR-0003): providers get machines, launchers
   start processes, strategies configure execution, recipes integrate user
   code. Hugging Face code lives in `recipes/` — it is workload layer,
   never a backend. The planner emits a backend-neutral `StrategyPlan` and
   **never imports framework code** (no `import transformers`/`torch
   .distributed`/`ray` inside `planner/`).
7. **No empty scaffolding.** Create a module only in the vertical slice
   that makes it real (workspace-root `PLAN_2WEEKS.md`). Mode A (leases)
   before Mode B extensions; never reimplement distributed ML — plan,
   launch, observe, recover. *Amendment (2026-07-19, by user request):*
   future-work packages carry **complete designed interfaces** (ABCs with
   full input/output contracts + contract tests in
   `tests/test_interfaces.py`) instead of docstring stubs — implement
   against them; change a contract only with a red test + a note.
8. Ledger is append-only; job/task status is derived from events, never a
   hand-mutated field. Checkpoint manifests are written last, after part
   hashes verify — no manifest, no checkpoint.

## Current state (July 2026)

- `protocol/v1alpha1.py` = the versioned public protocol (JobSpec, JobState,
  Event vocabulary, ArtifactRecord, node registration/heartbeat). flashnode
  and flashml-cloud import it.
- `backends/` = `ExecutionBackend` contract + working `KubeRayExecutionBackend`
  (JobSpec → RayJob CR, status mapping, event normalization). Pins: KubeRay
  chart 1.6.2, Ray 2.46.0.
- `artifacts/` = ArtifactStore protocol; MinIO (S3-compatible) + native OSS.
- `service/` = FastAPI runtime API (:8100) + SQLite ledger + `flashruntime` CLI.
- `flashml_workloads/sharded_kmeans.py` = the POC's Ray workload (deterministic
  shards, retriable tasks, honest recovery evidence).
- `deploy/docker/` = kmeans + service images. `docs/adr/` = ACK Edge + PAI-DLC ADRs.
- `scheduler/` = `PlacementPolicy` ABC + `FifoPlacement` + `IsolationAwarePlacement`
  (fail-closed sandbox placement for service-side command tasks). No scaffold
  packages remain; the numpy prototype engine has been removed.
- `planner/` + `protocol/plan_v1alpha1.py` = the strategy planner (July
  2026): `flash.plan(PlanRequest)` → explained `PlanReport`/`StrategyPlan`;
  CLI `flashruntime plan spec.yaml`. Deterministic, framework-import-free,
  curated menu (single_gpu/ddp/fsdp2/zero3_cpu_offload + QLoRA variants,
  lease_tasks for Mode A). Walkthrough: `docs/planner/README.md`. Estimator
  constants live in `planner/catalog.py`; the arithmetic is pinned by
  `tests/test_planner.py` — change formulas only with justification against
  workspace-root `FLASHRUNTIME_EVALUATION.md`.
- `leases/` = the Mode A state machine (July 2026): `LeaseManager` with
  claim/heartbeat/expiry-sweep/idempotent-commit, injectable clock, typed
  Event emission, `LeaseStore` seam (InMemory + restart-durable `SqliteLeaseStore`).
- `checkpoint/` = `CheckpointCatalog` (July 2026): parts-first /
  manifest-last commit, validation ladder (hash→restore-verified→invalid),
  topology-compatible `latest_valid()`, `lost_work()` economics.
- `recovery/` = `classify(FailureSignals)` precedence taxonomy +
  `decide(failure, mode)` versioned policy table (POLICY_VERSION); total
  over FailureClass × mode, correlated incidents freeze automation.
- **Core is pydantic-only**: `import flashruntime` + planner/leases/
  checkpoint/recovery must never require numpy/k8s/minio/fastapi (verified
  by a clean-venv smoke; the bring-your-own-code SDK helpers are lazy via
  PEP 562 `__getattr__` so the core import stays minimal). Keep it that way.
- `service/` Mode A surface (July 2026): LeaseManager over HTTP
  (`/v1alpha1/leases/claim`, `/attempts/{id}/{heartbeat,complete,fail}`),
  minimal node registry (`/nodes/*` — self-hosted profile; cloud fronts it
  with join codes later), **local artifact hosting** (`PUT|GET
  /v1alpha1/artifacts/{key}` under FLASHML_LOCAL_ARTIFACTS_DIR),
  `hyperparameter_search` job→task expansion (`service/modea.py`,
  `execution.backend: leases`), derived job states + 2 s sweeper, and a
  self-contained dashboard at `GET /` (`service/dashboard.py`). KubeRay is
  optional: `FLASHML_ENABLE_KUBERAY=0` runs the coordinator cloud-free.
- `workloads/` + `flash.submit` = bring-your-own-code (July 2026):
  `CommandWorkload` (framework-neutral "run this command"; `argv()`,
  `to_jobspec()`) + `flash.submit(workload, output_dir=None) → Run`
  (synchronous local compile→launch→wait→collect). First concrete
  launcher (`launchers/local.py`), recipe (`recipes/command.py`), and
  strategy compiler (`strategies/command.py` — a function, not a
  `StrategyCompiler`, since a StrategyPlan carries no argv). Thin
  framework adapters in `integrations/` (sklearn sweeps, pytorch DDP, HF
  Trainer callback — no framework imports at module level) + the optional
  in-script `flashruntime.torch` helper (3 verbs + read-only accessors incl.
  `device()`/`backend()` — capability guardrail, not a count: prepare/
  checkpoint/log_metrics/rank/world_size/is_main/start_step; wraps torch's
  own DDP and stops — ADR-0003 guardrail). Service-side command jobs expand
  + lease with fail-closed sandbox placement (`scheduler.IsolationAwarePlacement`);
  **executing** the `argv` payload waits on flashnode's argv runner tier
  (cross-repo). User-facing guide: `docs/guides/bring-your-code.md`.
- **Automatic recovery in the SDK** (July 2026): `flash.submit(...,
  max_restarts=N)` turns a FAILED attempt into `FailureSignals`, calls
  `recovery.classify()`/`decide()`, and — unless the failure is a
  deterministic app error (FAIL_JOB, fast-stop) — relaunches from the
  job-scoped checkpoint. Decisions surface as `FAILURE_CLASSIFIED` /
  `RECOVERY_ACTION_SELECTED` events on the run (`sdk.py`). This is the
  SDK-local path; the *service coordinator* still recovers via implicit
  lease-expiry (classify/decide uncalled there).
- **Live run viewer** (`viewer/`, July 2026): a stdlib, zero-CDN HTML page
  served on loopback during a watched run (KPI dashboard strip, machine →
  worker → rank process flow map with click-in detail panel, loss +
  resource charts, verified checkpoints, recovery feed; telemetry from
  `monitor/` ResourceSampler — optional psutil via the `[monitor]` extra —
  and per-rank heartbeat files from `flashruntime.torch`), reading only
  the versioned `run.json` (`viewer_v1`) contract. `flash.submit(watch=…)`
  opens it (auto: on at a TTY, off in pipes/CI); the built docs site is
  served from it at `/docs`.
- **Docs site + benchmarks** (July 2026): PyTorch-style site under
  `docs/site/`, built by `scripts/build_docs.py` (every byte inline;
  `--check` link-verifies), deployed to GitHub Pages on release. Honest
  benchmark suite (`python -m benchmarks run`): each result records its
  host, missing comparators are skipped rows not fakes; measured Apple-M4
  baseline committed under `benchmarks/results/`.
- **Real-GPU validation (2026-07-23):** the CUDA paths (nccl DDP, per-rank
  device placement, GPU kill-and-resume) validated on **2×RTX 4090**
  (RunPod community cloud, torch 2.7.1+cu128, CUDA 12.8), $0.0725 total.
  `tests/test_gpu_e2e.py` (CUDA-gated, skips without a GPU); harness
  `scripts/runpod_gpu_e2e.py` (`--plan-only` dry-run). It found a real
  GPU-only bug the CPU/gloo suite could not: batches must move to
  `ft.device()`.
- **Release engineering** (July 2026): CI matrix (Python 3.10–3.13 ×
  ubuntu/macos) with a pydantic-only core-smoke job, a docs link-check, and
  a bench-smoke; tag-triggered release (PyPI Trusted Publishing, test-gated,
  + GitHub Pages); `scripts/audit_secrets.sh` (worktree + history scan,
  CLEAN); `CONTRIBUTING.md` / `SECURITY.md` / `CHANGELOG.md`; PEP 561
  `py.typed`, built docs bundled in the wheel.
- Tests: `pytest` → **264 passed, 1 skipped** (the CUDA-gated GPU test),
  9 deselected (integration + bench_smoke, opt-in via `-m`; live in
  `tests/integration/` with env auto-skip). Images: `deploy/docker/`.
  Full-loop proof: workspace-root `e2e/` (`make e2e`, `make e2e-demo`) +
  in-repo `tests/test_examples_e2e.py` (4 real bring-your-code e2e tests,
  incl. kill-at-40 → auto-resume, final loss matching baseline to 1e-6).
  POC runbook: workspace-root
  `archive/POC_PLAN.md` + Makefile.

## Status vs. plan (what's done, what's missing)

Architecture decisions: `docs/adr/0003-…` + workspace-root
`FLASHRUNTIME_EVALUATION.md`; execution log: workspace-root `PROGRESS.md`.

**Done (July 2026, deploy-ready milestone):** the 0.1.0 feature set.
Bring-your-own-code SDK (`flash.submit(CommandWorkload)`) + `flashruntime
submit`/`submit-spec` CLI; sklearn/pytorch/HF integration adapters +
optional `flashruntime.torch` helper; automatic SDK recovery (`max_restarts`
→ classify/decide, one-call kill-and-resume proven in
`tests/test_examples_e2e.py`); live run viewer opening on `--watch`;
PyTorch-style docs site (built + served at `/docs` + Pages-ready); honest
benchmark suite + measured M4 baseline; CI matrix + test-gated trusted
publishing + secrets audit CLEAN; real-GPU validation (2×RTX 4090, nccl,
$0.0725). Earlier local-milestone core still stands: Mode A lease
coordinator over HTTP (claim/heartbeat/complete/fail; commit-time sha256
validation; join-code registration; artifact size caps); durable
`SqliteLeaseStore` (in-flight leases survive restarts); job→task expansion
for `hyperparameter_search`/`sharded_kmeans`; checkpoint HTTP surface;
`flashml_workloads/` task modules; dashboard at GET /; the planner. Proven
by workspace-root `e2e/` (kill-a-machine sweep, K-means convergence,
cross-machine resume). Suite: **237 passed, 1 skipped, 9 deselected**.

**Missing (in rough priority order):**
1. **Multi-node DDP rendezvous** — `nnodes > 1` in the pytorch adapter
   raises `NotImplementedError` today; single-node (`--standalone`) is the
   proven path. GPU is validated single-node; multi-node GPU rendezvous is
   a later launcher slice.
2. **FlashNode argv runner** (cross-repo): service-side command jobs expand
   + lease + placement-check today, but *executing* the `argv` payload
   waits on flashnode's argv runner tier.
3. **Checkpoint-manifest persistence** — catalog is in-memory; checkpoint
   *files* are durable, so manifests die with the coordinator.
4. **Stage-8 ledger metrics** (MTTD/MTTR/goodput/lost-work) + case study —
   every needed event already exists.
5. Service-side recovery wiring — the SDK path fires classify/decide +
   FAILURE_CLASSIFIED / RECOVERY_ACTION_SELECTED, but the *coordinator*
   still recovers via implicit lease-expiry.
6. `flash.run(plan)` — planner and executor exist but aren't linked (raises
   `NotImplementedError`); e2e plans then builds the JobSpec by hand.
7. Cloud stage: Postgres (same append-only schema), SSE instead of
   polling, ACK/KubeRay hybrid pool, HF Trainer + PEFT LoRA recipes on the
   sgd_trainer checkpoint contract.

## Dev workflow

```bash
uv venv && uv pip install -e ".[sklearn,dev]"
pytest                    # 264 passed, 1 skipped (CUDA-gated), 9 deselected
                          #   (+ opt-in: pytest -m integration / -m bench_smoke)
```

Python ≥3.10, Pydantic for new schemas, pytest for everything. Match existing
code style; keep modules small and boundaries explicit.
