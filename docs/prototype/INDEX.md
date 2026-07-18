# FlashML Documentation Map

Use this page to choose the smallest trustworthy reading path.

## Start here

1. [`status/README.md`](status/README.md) — what stage is real, what is missing, and the next task.
2. [`ARCHITECTURE.md`](ARCHITECTURE.md) — the current code and execution flow.
3. [`guides/CODE_WALKTHROUGH.md`](guides/CODE_WALKTHROUGH.md) — trace one working local job through real functions.
4. [`phases/`](phases/) — requirements, evidence, missing work, and exit gate for every phase.

## Documents by role

| Document | Authority |
|---|---|
| [`../README.md`](../../README.md) | Product idea, working quickstart, high-level status |
| [`status/README.md`](status/README.md) | Current verified snapshot and immediate gap list |
| [`ARCHITECTURE.md`](ARCHITECTURE.md) | As-built behavior only |
| [`ROADMAP.md`](ROADMAP.md) | Compact phase index and advancement rules |
| [`phases/`](phases/) | Detailed delivery requirements and exit gates |
| [`PROVIDERS.md`](PROVIDERS.md) | Compute/provider extension contract |
| [`MODELS.md`](MODELS.md) | Algorithm/model extension contract |
| [`TOOLS.md`](TOOLS.md) | RunPod Flash mechanics used by Phase 1 |
| [`IMPLEMENTATION.md`](IMPLEMENTATION.md) | Historical MVP design; not current truth |

## Task-specific paths

### Understand the current implementation

`status/README.md` → `guides/CODE_WALKTHROUGH.md` →
`examples/local_kmeans_and_linear_regression.py` → `tests/test_local_provider.py`.

### Add a provider

The relevant phase document → `PROVIDERS.md` →
`flashml/providers/base.py` → `flashml/providers/local/provider.py`.

For RunPod, then read `TOOLS.md` and use `legacy/coordinator/flash_worker/`
only as a migration source.

### Add an algorithm or framework

The relevant phase document → `MODELS.md` →
`flashml/algorithms/base.py` → either `kmeans.py` or
`sklearn_partial_fit.py`, depending on the distributed shape.

### Work on the dashboard

`phases/03-serve-dashboard/README.md` → `apps/dashboard/lib/api.ts` →
`legacy/coordinator/server.py` for the contract being replaced.

## Maintenance rules

1. As-built behavior belongs in `ARCHITECTURE.md`; plans belong in a phase folder.
2. Check a requirement in the same change that adds its implementation and verification.
3. Record direct evidence before marking a phase complete.
4. When a phase changes state, update its document, `status/README.md`, `ROADMAP.md`, and the root status table together.
5. When an abstract contract changes, update `PROVIDERS.md` or `MODELS.md` before dependent implementation docs.
6. Historical and legacy code must remain visibly labeled; it must not be presented as the current package.

## Glossary

| Term | Meaning |
|---|---|
| Connector | A named `Provider` implementation such as `local` or future `runpod`. |
| Worker pool | Provisioned execution capacity implementing submit/gather. |
| Storage | Blob store reachable by a provider's workers. |
| Entrypoint | Dotted `module:function` resolved inside a worker. |
| Distributed algorithm | Plan/initialize/make-tasks/reduce/converge/finalize contract. |
| Conformance suite | Shared behavioral tests every connector must pass. |
