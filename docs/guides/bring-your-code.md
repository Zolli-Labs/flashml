# Bring your own code

> **This guide now lives in the documentation site**, split per framework and
> paired with hands-on tutorials. This page is a pointer so existing links keep
> working; follow the links below for the full, maintained content.

FlashRuntime **operates** your training job — it never rewrites your model. You
keep the framework, the model, the loop, and the loss you already have.
FlashRuntime wraps a reliability and reproducibility layer around them: it
launches your command, injects the environment it promises, tracks metrics,
validates checkpoints, retries on failure, and collects the artifacts.

| You own | FlashRuntime owns |
|---|---|
| model, training loop, loss, data, framework | launch, env vars, metric tracking, checkpoint validity, retry, recovery, artifact collection |

The contract at the boundary is deliberately thin: **CLI flags in,
`metrics.json` out.** A script that already reads its hyperparameters from
`argparse` and writes a small JSON file of results needs *zero* FlashRuntime
imports to be operated. This is ADR-0003's fourth axis in practice: **recipes
integrate user code**; the distributed math is always done by your framework.

## Where the content went

The documentation site (built by `scripts/build_docs.py`, served at `/docs`
from the live viewer, and published to GitHub Pages) carries the full guides
and tutorials:

**Tutorials** (`docs/site/tutorials/`)

- [`convnet.md`](../site/tutorials/convnet.md) — the flagship: an ordinary
  PyTorch ConvNet made into a fault-tolerant, crash-resuming 2-process DDP run.
- [`sklearn-sweeps.md`](../site/tutorials/sklearn-sweeps.md) — parallel
  scikit-learn hyperparameter sweeps.
- [`fault-tolerance.md`](../site/tutorials/fault-tolerance.md) — how a crash
  becomes signals → a failure class → a typed recovery action.

**Guides** (`docs/site/guides/`)

- [`pytorch.md`](../site/guides/pytorch.md) — the two launch paths (already-DDP
  scripts vs. the `flashruntime.torch` helper) and every caveat.
- [`sklearn.md`](../site/guides/sklearn.md) — the fan-out adapter and the
  "distribute across runs, never inside `.fit()`" rule.
- [`huggingface.md`](../site/guides/huggingface.md) — the Trainer callback seam
  for verified-manifest checkpoints.
- [`jobspec-and-isolation.md`](../site/guides/jobspec-and-isolation.md) —
  compiling to a coordinator `JobSpec` and the fail-closed isolation tiers.

**Concepts & reference** (`docs/site/concepts/`, `docs/site/reference/`)

- [`concepts/architecture.md`](../site/concepts/architecture.md) — the four
  axes, leases, manifests, and recovery, with diagrams.
- [`reference/sdk.md`](../site/reference/sdk.md),
  [`integrations.md`](../site/reference/integrations.md),
  [`torch-helper.md`](../site/reference/torch-helper.md), and
  [`cli.md`](../site/reference/cli.md) — exact signatures.

**See also:** the strategy planner walkthrough
([`docs/planner/README.md`](../planner/README.md)) for choosing *how* a job
should run before you submit it, and ADR-0003
([`docs/adr/0003-reliability-runtime-first-planner-second.md`](../adr/0003-reliability-runtime-first-planner-second.md))
for the four-axis architecture this contract comes from.
