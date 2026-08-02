# JobSpec & isolation guide

`flash.submit()` runs your workload on the **local** machine. To hand a command
workload to a FlashRuntime **coordinator** — so nodes pull and run it under
leases, heartbeats, and recovery — you compile it to the versioned wire form (a
`JobSpec`) and POST it.

This guide covers that compile step and the **isolation tier** that decides
which machines a task is allowed to land on. For the local path, see the
[PyTorch](pytorch.md) and [scikit-learn](sklearn.md) guides.

---

## Compile to a JobSpec

```python
from flashruntime.workloads.command import to_jobspec
from flashruntime.protocol.v1alpha1 import ImageSpec

jobspec = to_jobspec(
    workload,                                    # a CommandWorkload
    name="my-sweep",
    image=ImageSpec(repository="myrepo/trainer", tag="2026.07-a1b2c3"),
)
# POST jobspec.model_dump() to  POST /v1alpha1/jobs
# (or from the CLI:  flashruntime submit-spec spec.json)
```

`to_jobspec(workload, name, image=None)` produces a
`JobSpec{execution.backend: "leases", workload.type: "command"}`. **A pinned
image is required** — remote runs must be reproducible, and the schema already
rejects the tag `latest`. On the coordinator the `command` recipe expands the
job into one `TaskSpec` per `task_params` entry (or a single task), each
carrying an `argv` payload, its env, its `artifact://` inputs, and its
isolation requirement.

---

## Isolation tiers (fail closed)

Every command task carries an isolation tier from `workload.isolation`:

| Tier | Where it runs | Meaning |
|---|---|---|
| `standard` (default) | your own machines, RunPod, trusted pools | ordinary placement — runs anywhere |
| `sandboxed` | community / untrusted machines | may only be leased to a node advertising `sandbox_capable is True` |

The placement gate (`scheduler.IsolationAwarePlacement`) is **fail-closed** on
the security-relevant field, per the schema-security rule:

- A node counts as capable **only** when `sandbox_capable is True` — a truthy
  stand-in (the string `"false"`, `1`, `"yes"`) does **not** count.
- Any tier that is not `None` / `""` / `"standard"` (including a mistyped
  `"Sandboxed"`) is treated as requiring capability — no silent downgrade.
- A `sandboxed` task never falls back to an uncapable node unless the workload
  explicitly sets `isolation.allowFallback = True`.

So a `sandboxed` task will sit unclaimed rather than land on a node that cannot
isolate it. That is the intended behavior: unsafe placement fails closed.

---

## What runs where today

> **Local SDK path — works now.** `flash.submit()` runs sklearn sweeps,
> 2-process CPU DDP (via `gloo`), and kill-and-resume from checkpoints on this
> machine. All three are proven by the example e2e tests.
>
> **Service-side command jobs — expansion, placement, and execution all
> work.** POSTing a `to_jobspec()` workload expands it into leased tasks,
> places them fail-closed by isolation tier, and — with a FlashNode agent
> running `--runner argv` — executes the `argv` payload inside a hardened,
> network-isolated container and commits the result. `sandboxed` tasks are
> only ever placed on a node that advertises both `sandbox_capable` and
> `argv_capable`; see the repo's `docs/guides/donate-a-machine.md` for exactly
> what that container confines (and does not).
>
> **Later slices.** Multi-node DDP (`nnodes > 1` rendezvous — not available on
> volunteer nodes even later, since `--network none` rules out rendezvous),
> result verification for untrusted volunteer nodes, remote providers
> (RunPod) with source packaging (`git_revision`), and
> `flash.run(StrategyPlan)` wiring are open follow-ups.

For how a leased task recovers when a node disappears, see the
[fault-tolerance tutorial](../tutorials/fault-tolerance.md) and the
[architecture](../concepts/architecture.md) page — the same failure taxonomy
and policy table drive both the local path and the coordinator.

---

## Adding another framework

Isolation and JobSpec compilation are **framework-neutral**: `to_jobspec`
serializes any `CommandWorkload`, whatever built it (`fr_torch.ddp`,
`fr_sklearn.sweep`, `fr_hf.trainer`, or one you hand-construct). So a new
framework adapter (see the [PyTorch adapter](pytorch.md#adding-another-framework))
gets coordinator submission and isolation-aware placement for free — it only
has to return a `CommandWorkload`.

---

## Built-in task modules

Besides `command` workloads, the coordinator ships three allowlisted task
modules under `flashml_workloads/` — `sklearn_trial` (hyperparameter
trials), `kmeans_shard`/`kmeans_driver` (sharded K-means), and
`sgd_trainer` (checkpointable SGD with bit-identical resume). They are
reference workloads for the lease protocol, not a required path: they
predate `command` workloads and remain the workspace e2e's proof fixtures.
Their contract is documented in each module's docstring and in the repo's
`AGENTS.md`.
