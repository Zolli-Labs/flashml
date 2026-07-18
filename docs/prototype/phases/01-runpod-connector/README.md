# Phase 1 — RunPod Connector

**State:** Current. Not implemented.

## Outcome

The Phase 0 K-Means example runs on real RunPod Flash workers by changing the
provider selection, without modifying the engine or algorithm.

## Prerequisites

- Phase 0 baseline and provider contract.
- RunPod mechanics in [`../../TOOLS.md`](../../TOOLS.md).
- Migration source in `legacy/coordinator/flash_worker/`.

## Required deliverables

- [ ] `flashml/providers/runpod/tiers.py`: normalized tier mapping; unsupported tiers raise `ProvisionError`.
- [ ] `flashml/providers/runpod/storage.py`: `Storage` backed by NetworkVolume/S3-compatible operations.
- [ ] One generic remote worker endpoint that resolves `WorkerSpec.entrypoint`; no per-algorithm endpoints.
- [ ] `RunpodWorkerPool.submit()` and `gather()` using `/runsync`.
- [ ] Task failures return `TaskResult(ok=False)` rather than escaping from `gather()`.
- [ ] `RunpodProvider.provision()` blocks until submissions are accepted and cleans partial resources on failure.
- [ ] `teardown()` scales to zero and is idempotent.
- [ ] `offers()` is best-effort and never provisions resources.
- [ ] `RUNPOD_API_KEY` plus explicit `credentials=` override; missing/rejected credentials raise `AuthError` before task dispatch.
- [ ] Provider registration under the name `runpod`.
- [ ] Unit tests for tiers, credentials, failure translation, and idempotent teardown using fakes.
- [ ] Live smoke-test record for K-Means on real RunPod workers.
- [ ] Update the provider matrix in `docs/PROVIDERS.md`.

## Implementation order

1. Tier mapping and unit tests.
2. Storage adapter and mocked contract tests.
3. Generic endpoint and entrypoint/dependency loading.
4. Worker pool transport and failure normalization.
5. Provider lifecycle and cleanup.
6. Registry, docs, then paid live smoke test.

## Verification

```bash
.venv/bin/python -m pytest tests/ -q
# Future focused command:
.venv/bin/python -m pytest tests/providers/test_runpod_provider.py -q
```

Live acceptance: run the K-Means scenario from the local example with
`provider="runpod"`; results must match local execution within documented
numerical tolerance and leave no billable worker after teardown.

## Missing now

Every deliverable above is missing. The legacy code proves RunPod mechanics,
but it is fused to one coordinator and therefore does not satisfy this phase.

## Exit gate

The same algorithm and dataset execute through both `local` and `runpod`, with
only provider configuration changed, typed failures, shared storage visibility,
and verified cleanup.
