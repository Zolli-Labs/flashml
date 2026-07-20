# Phase 3 — Serve and Dashboard

**State:** Waiting on Phases 1–2.

## Outcome

`flashml serve` exposes the API consumed by `apps/dashboard/`, and the
dashboard displays live jobs from the provider-agnostic engine. The retained
coordinator is no longer a separate implementation of training behavior.

## Required deliverables

- [ ] `flashml/server/app.py` with the existing frontend API contract.
- [ ] Background job execution; `Cluster.train()` must not finish before HTTP clients can observe events.
- [ ] Thread-safe job store and event/status serialization.
- [ ] Dataset upload and result endpoints required by `apps/dashboard/lib/api.ts`.
- [ ] `flashml serve` command in `pyproject.toml`.
- [ ] Dashboard points to the new server with no model-specific backend fork.
- [ ] `legacy/coordinator/server.py` becomes a temporary import/deprecation shim.
- [ ] Browser verification for local and RunPod jobs, classical and PyTorch.
- [ ] API contract tests covering every frontend request/response type.

## Exit gate

The production dashboard build passes and a browser can launch, observe, and
finish jobs through `flashml serve`; no training loop remains in the legacy shim.
