# Phase 0 — Foundation

**State:** Implemented; baseline commit pending.

## Outcome

A user can install `flashml`, select `provider="local"`, train the built-in
algorithms end to end, stream iteration events, and receive a usable fitted
result without depending on the legacy coordinator.

## Requirements and evidence

| Requirement | Status | Evidence |
|---|---|---|
| Public `Cluster`, `Job`, `JobEvent`, and `algorithms` API | Done | `flashml/__init__.py` |
| Provider, worker pool, task, storage, and algorithm contracts | Done | `flashml/providers/base.py`, `flashml/algorithms/base.py` |
| Zero-setup local provider | Done | `flashml/providers/local/` |
| MapReduce implementation | Done | `flashml/algorithms/kmeans.py` |
| Parameter-server/FedAvg-style sklearn adapter | Done | `flashml/algorithms/sklearn_partial_fit.py` |
| Large data stored once; task payloads remain small | Done | `flashml/engine/loop.py` |
| Numerical end-to-end tests and runnable example | Done | `tests/`, `examples/` |
| Clean version-control baseline | Missing | Phase 0 files are still untracked in the current worktree |

## Verification

```bash
.venv/bin/python -m pytest tests/ -q
.venv/bin/python examples/local_kmeans_and_linear_regression.py
```

## Exit gate

All verification passes, the package can be installed from `pyproject.toml`,
and the implementation plus docs are committed as one reviewable baseline.

## Deferred

- Remote compute belongs to Phase 1.
- New distributed model shapes belong to Phase 2.
- HTTP serving and live background jobs belong to Phase 3.
