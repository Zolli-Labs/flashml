# Current Project Status

Last verified: 2026-07-12.

## Current stage

FlashML has a working **Phase 0 local foundation** and is beginning
**Phase 1, the RunPod connector**. Phase 2 model adapters may proceed against
the local connector in parallel.

The repository contains three intentionally separated surfaces:

| Surface | Role | Runtime status |
|---|---|---|
| `flashml/` | Current provider-agnostic library | Working on `local` |
| `apps/dashboard/` | Product dashboard retained for Phase 3 | Builds; still calls legacy API |
| `legacy/coordinator/` | Migration source for RunPod and HTTP behavior | Retained, not part of the package |
| `archive/experiments/` | Pre-library experiments and local datasets | Historical only |

## Verified evidence

```bash
.venv/bin/python -m pytest tests/ -q
# 10 passed

.venv/bin/python examples/local_kmeans_and_linear_regression.py
# K-Means, linear regression, and logistic regression passed

cd apps/dashboard && npm run build
# Next.js production build passed
```

## Missing requirements

1. Commit the Phase 0 package, tests, examples, and structured docs as a clean baseline.
2. Implement `flashml.providers.runpod` and prove K-Means on real RunPod workers.
3. Add PyTorch/FedAvg and embarrassingly-parallel algorithm adapters.
4. Replace the retained coordinator with `flashml serve` before calling the dashboard integrated.
5. Package the provider conformance checklist and use it to validate connectors.

## Next task

Start Phase 1 with the normalized accelerator mapping described in
[`../phases/01-runpod-connector/README.md`](../phases/01-runpod-connector/README.md),
then implement provider-visible storage. These are deterministic pieces that
can be tested without provisioning paid compute.
