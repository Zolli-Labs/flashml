# Phase 5 — v1 Hardening

**State:** Planned.

## Outcome

The public API and operational behavior are dependable enough for a v1.0
release rather than only controlled demonstrations.

## Required deliverables

- [ ] Retry policy and straggler redispatch with bounded attempts.
- [ ] Checkpoint/resume semantics for supported algorithms.
- [ ] Per-job cost reporting where providers expose billing data.
- [ ] Cross-provider offers API with normalized availability/pricing.
- [ ] Exportable result artifacts and metadata.
- [ ] Public API compatibility review and versioning policy.
- [ ] Opt-in cloudpickle convenience path with an explicit trust boundary.
- [ ] Packaging, release automation, supported Python matrix, and PyPI decision.
- [ ] Operational documentation for timeouts, cleanup, credentials, and failure recovery.

## Exit gate

The frozen v1 API passes conformance and end-to-end tests across supported
providers, has documented failure/recovery behavior, and can be installed from
the chosen distribution channel.
