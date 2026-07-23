# Changelog

All notable changes to FlashRuntime are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The public wire protocol carries its own schema version (`v1alpha1`)
independent of this package version.

## [Unreleased]

_No unreleased changes yet._

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
  correct device/DDP setup with a few lines. Bit-identical resume from the
  newest valid manifest; save writes CPU tensors so manifests stay
  topology-compatible.
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

- **GPU validation is pending.** The CUDA code paths (nccl DDP, device
  placement, GPU kill-and-resume) are implemented and unit-tested on CPU via a
  pure device-selection helper, but end-to-end validation on real GPUs is not
  yet complete. This note will be updated with the validated hardware, torch/
  CUDA versions, and date once that run lands.
- **Cloud is out of scope here.** FlashRuntime is useful and complete without
  the cloud; the managed control plane is a separate, private component.

[Unreleased]: https://github.com/Zolli-Labs/flashruntime/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/Zolli-Labs/flashruntime/releases/tag/v0.1.0
