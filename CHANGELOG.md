# Changelog

All notable changes to FlashRuntime are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The public wire protocol carries its own schema version (`v1alpha1`)
independent of this package version.

## [Unreleased]

_No unreleased changes yet._

## [0.2.0] - Unreleased

### Added

- **`run_fedavg(build_round=...)`**: a round no longer has to be the built-in
  `federated_averaging` body. A caller supplies a `RoundPlan`
  (`{"body", "task_ids"}`) per round, which is what lets *user code* be the
  round worker — a repo entrypoint compiled to a `command` job per round —
  instead of `flashml_workloads.fedavg_worker`. Everything downstream
  (quorum, reduce, weights artifact, `resume_state`) is unchanged: none of
  it depends on what ran inside the round, only on the task ids it produced
  and the `metrics.json`/`delta.json` pair each one committed. `task_ids` is
  carried rather than inferred because the committed-artifact filter is a
  participant-count security boundary, and each recipe names its tasks
  differently (`shard-000` vs `CommandRecipe`'s `task-000`).
- **`run_fedavg(initial_weights={})`** is now defined rather than a
  `WeightShapeMismatch`: it means the driver holds no weights yet, so round
  0's reduced contribution *is* the first set of weights. That is the only
  reading consistent with the worker contract ("`delta.json` is the change
  from the weights you were given") when a worker was given none, and it is
  what makes a driver that has never seen the user's model — one running
  inside a service with no torch — able to start a run at all. Passing real
  `initial_weights` behaves exactly as before.
- **`NodeRegistration.module_capable`** (default `True`): advertises whether
  a node can run "module" (`python -m <allowlisted module>`) tasks,
  independent of `argv_capable`. `IsolationAwarePlacement` now refuses to
  place a module-shaped task on a node that explicitly advertises
  `module_capable: false` (an argv-only volunteer) — closing a gap where one
  such node joining a pool would burn every attempt on an unrelated
  `hyperparameter_search`/`sharded_kmeans` job and fail it. Unlike
  `argv_capable`, this gate is fail-**open**: a registration that omits the
  field (every node on the 0.1.0 wire format) keeps receiving module work.

### Changed

- Package version bumped to 0.2.0 because this release adds a new
  wire-visible field (`module_capable`) to `NodeRegistration`. Consumers
  pinned to `flashruntime>=0.1` may run against a 0.1.0 wheel that lacks
  the field entirely; `flashnode`'s dependency pin moves to `>=0.2` in
  lockstep.

## [0.1.0] - Unreleased

First public release: the self-hostable, cloud-free fault-tolerant runtime.
FlashRuntime plans, launches, observes, and recovers ML jobs — the distributed
computation itself is always done by established libraries (PyTorch
DDP/FSDP2/torchrun, Ray, Hugging Face).

### Added

- **Bring-your-own-code SDK and CLI.** `flash.submit(workload, …)` runs a
  user's unmodified training/eval script; `flashruntime submit CMD [--source
  DIR] [--task-params JSON] [--max-restarts N] [--output-dir DIR]` is the CLI
  equivalent. `submit(wait=False)` returns immediately with a live `run`
  handle (`run.wait()`, `run.state`, append-only `run.events`).
- **One-file framework integrations** (`flashruntime.integrations`): `sklearn`
  (independent-task sweeps + grid HPO), `pytorch` (DDP/torchrun coordinated
  training), and `huggingface`. Adding a framework stays one small adapter file
  that imports no framework code — the contract with the user's script is
  convention only (CLI flags in, `metrics.json`/checkpoints out).
- **`flashruntime.torch`** — a tiny helper (`ft.prepare` / `ft.checkpoint`) that
  gives a plain PyTorch script topology-agnostic, resumable checkpointing and
  correct device/DDP setup with a few lines. Resumes from the newest valid
  manifest (e2e asserts the resumed final loss matches an uninterrupted run
  to 1e-6); save writes CPU tensors so manifests stay topology-compatible.
- **Automatic fault tolerance** via `flash.submit(..., max_restarts=N)`: a
  failed launch is classified (`recovery.classify`) and a typed action chosen
  (`recovery.decide`); transient failures relaunch from the last valid
  checkpoint with no human in the loop, deterministic app errors fast-stop
  without burning restarts. Decisions are recorded on the run
  (`FAILURE_CLASSIFIED`, `RECOVERY_ACTION_SELECTED`) and rendered by the viewer.
- **Live run viewer** — a self-contained, zero-CDN HTML page served during a
  watched run: job/task state, events, recovery decisions, checkpoints. Reads
  only the versioned `run.json` contract, so any launcher that writes it
  renders.
- **Strategy planner** (`flash.plan` / `flashruntime plan spec.yaml`):
  deterministic, framework-import-free, explainable strategy selection over a
  curated menu (single_gpu / ddp / fsdp2 / zero3_cpu_offload + QLoRA variants;
  `lease_tasks` for Mode A). Emits a backend-neutral, explained
  `PlanReport` / `StrategyPlan`. No cluster required.
- **Mode A lease coordinator** over HTTP: claim / heartbeat / expiry-sweep /
  idempotent *validated* commit (commit-time sha256 against the task's
  commit_key), a minimal node registry with optional join codes,
  coordinator-hosted local artifacts, and per-task checkpoint manifests over
  the wire (parts-first / manifest-last). Durable `SqliteLeaseStore`: in-flight
  leases survive a coordinator restart. Job→task expansion for
  `hyperparameter_search` and `sharded_kmeans`. Built-in dashboard at `GET /`.
- **Honest benchmark suite** (`python -m benchmarks run`): scenarios add via a
  one-file registry; every result file records the host that produced it;
  missing comparators are reported as skipped rows, never faked. A measured
  baseline is committed and rendered into the docs.
- **Docs site** built from `docs/site/` by `scripts/build_docs.py` — the same
  visual tokens as the viewer, every byte inline (no CDN/web-font/remote
  image). Served offline at `/docs` by the packaged viewer and deployed to
  GitHub Pages on release.
- **Packaging**: PEP 561 `py.typed` (the package ships inline types), the built
  docs bundled in the wheel, classifiers, keywords, and project URLs.
- **Release + CI engineering**: a CI matrix (Python 3.10–3.13 × ubuntu/macos)
  with a pydantic-only core-smoke job, a docs link-check, and a benchmark
  smoke; a tag-triggered release pipeline (PyPI Trusted Publishing + GitHub
  Pages); and `scripts/audit_secrets.sh` (history + worktree credential scan).

### Notes

- **GPU validation: done (2026-07-23).** The CUDA code paths — nccl DDP,
  per-rank device placement, and GPU kill-and-resume — are validated end-to-end
  on real hardware: **2×NVIDIA GeForce RTX 4090** (RunPod community cloud),
  torch 2.7.1+cu128, CUDA 12.8. `tests/test_gpu_e2e.py` asserts a 2-process
  nccl run completes (backend reported `nccl`, model on `cuda:0`) and that a
  crash at step 40 resumes from the last valid checkpoint onto the uninterrupted
  loss. The run surfaced and fixed a real GPU-only bug the CPU/gloo suite could
  not: batches must be moved to `ft.device()` (model was placed on CUDA, data
  was not). Harness: `scripts/runpod_gpu_e2e.py` (dry-runnable via `--plan-only`).
- **Cloud is out of scope here.** FlashRuntime is useful and complete without
  the cloud; the managed control plane is a separate, private component.

[Unreleased]: https://github.com/Zolli-Labs/flashruntime/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/Zolli-Labs/flashruntime/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/Zolli-Labs/flashruntime/releases/tag/v0.1.0
