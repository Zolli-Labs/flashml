# Phase 4 — Conformance Suite and SSH Connector

**State:** Waiting on Phase 1.

## Outcome

The provider contract is independently testable, and a contributor can build
an SSH connector using the contract rather than reading engine internals.

## Required deliverables

- [ ] Package `flashml.testing.provider_conformance`.
- [ ] Cover lifecycle, round-trip, parallelism, storage visibility, typed failures, partial provisioning cleanup, and payload limits.
- [ ] Run the suite against `local` and `runpod`.
- [ ] Implement `flashml/providers/ssh/` with host staging and remote execution.
- [ ] Support credentials/keys without placing secrets in task payloads.
- [ ] Register `ssh` and update the provider matrix.
- [ ] Record any contract ambiguity discovered while implementing SSH and fix the contract first.

## Exit gate

Local, RunPod, and SSH pass the same suite, and SSH was implementable from
`docs/PROVIDERS.md` plus its own transport documentation.
