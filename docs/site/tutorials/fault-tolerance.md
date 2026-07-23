# Tutorial: automatic recovery, explained

When a run fails, FlashRuntime does not guess what to do. It turns the failure
into typed **signals**, **classifies** them into one failure class, and looks
that class up in a **versioned, deterministic policy table**. Same failure +
same policy version ⇒ same action, every time. No LLM, no scoring, no learned
model in the loop.

This tutorial follows one crash all the way through that pipeline, using the
local `flash.submit()` path. It assumes you have done the
[ConvNet tutorial](convnet.md), whose kill-and-resume run is the worked
example here.

---

## The budget: `max_restarts`

Recovery on the local path is opt-in through one argument:

```python
import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch

run = flash.submit(
    fr_torch.ddp("train.py", source=".", nproc_per_node=1,
                 script_args="--steps 24 --checkpoint-every 8 --kill-at-step 8"),
    output_dir="out/convnet",
    max_restarts=1,     # the automatic fault-tolerance budget (default 0 = no retry)
)
```

`max_restarts=0` (the default) means "run once, no retry" — the original
behavior. Any higher number is the recovery budget: on a FAILED attempt,
FlashRuntime consults the policy and, unless the failure is a deterministic
application error, relaunches the **same spec** from the job-scoped checkpoint,
up to that many times.

---

## Step 1 — signals: what a local process can actually tell you

The first thing FlashRuntime does with a dead attempt is translate it into
`FailureSignals`. On the local path the translator is
`recovery.signals.from_local_launch(exit_code, log_tail)`, and it is
**deliberately narrow**: a single local process can only evidence two things —
its OS exit code and the tail of its captured stdout+stderr. It never
fabricates node, accelerator, communication, or storage signals it cannot
observe; those classes belong to the distributed coordinator, not a local
launcher.

It is a transparent lookup table, not an inference engine. The rules, checked
top to bottom, first match wins:

| # | Evidence | Signals produced |
|---|---|---|
| 1 | exit 0 / never started | neutral (nothing broke — a guard that keeps the function total) |
| 2 | a named deterministic exception on a traceback's **terminal line** (`ModuleNotFoundError:`, `NameError:`, `SyntaxError:`, `ImportError:`, `IndentationError:`, `AttributeError:`) | `exit_deterministic=True` |
| 3 | a bare `Traceback` that is **not** the `torchrun` wrapper | `exit_deterministic=True` |
| 4 | everything else (signal death / OOM / bare `SystemExit` / `torchrun` `ChildFailedError`) | plain crash (transient) |

Two subtleties worth knowing, because they are where naive implementations get
it wrong:

- **Rule 2 is anchored to the start of the traceback's terminal line, not a
  bare substring.** CPython prints the failing exception type at the *start* of
  the final traceback line, while prose only ever *mentions* a name mid-line.
  The classic false positive is the startup log `... WARNING ImportError:
  flash_attn not available, falling back to eager` — a *transient* run that a
  substring scan would wrongly fail fast. Line-anchoring ignores the name in
  prose and fires only on a real terminal.
- **`torchrun` wraps every worker death — transient ones included — in a
  `ChildFailedError` and prints its own traceback.** That would trip rule 3, so
  rule 3 is disqualified whenever `ChildFailedError` is present, letting the
  death fall through to the transient default (rule 4). This is exactly why the
  ConvNet kill-and-resume run recovers instead of failing fast.

---

## Step 2 — classify: signals to one failure class

`recovery.classify(signals)` maps the signals to exactly one
`FailureClass`, precedence-ordered so **systemic evidence beats node evidence
beats process evidence beats application evidence** — a worker crash *during* a
correlated incident is the incident, not the worker. The full taxonomy:

```
correlated_incident      systemic: many failures in the window — stop acting
control_plane_failure    coordinator unreachable
preemption               spot capacity reclaimed
accelerator_failure      GPU / driver / XID fault
node_loss                heartbeat lost / node unreachable
communication_error      NCCL / RCCL / rendezvous
storage_timeout          object-store errors
artifact_corruption      hash mismatch / failed validation
application_error        deterministic bug (exit_deterministic) ── fail fast
worker_crash             non-deterministic process death ── one fresh attempt
unknown                  nothing matched
```

On the local path only the last three are reachable, because those are the only
classes a single process can evidence. A crashed `torchrun` worker with no
deterministic terminal is a **`worker_crash`**.

---

## Step 3 — decide: class × mode to a typed action

`recovery.decide(failure_class, mode)` is a pure table lookup returning a typed
`RecoveryDecision` stamped with `POLICY_VERSION`. The **mode** matters because
the same failure has a different blast radius:

- `independent_tasks` (a fan-out sweep) — one task retries; the others are
  untouched.
- `coordinated_training` (a `torchrun` group) — a lost rank stops the whole
  group, so recovery is a whole-group restart from the latest valid checkpoint
  (NCCL collective state is not repairable in place).

`flash.submit()` picks the mode from the workload: a fan-out is
`independent_tasks`, anything else (including a `torchrun` DDP run) is
`coordinated_training`. A few rows of the table:

| Failure class | independent_tasks | coordinated_training |
|---|---|---|
| `application_error` | **fail_job** (retrying burns money on a bug) | **fail_job** |
| `worker_crash` | retry_task | **restart_group** (from latest valid checkpoint) |
| `node_loss` | retry_task (cordon + requeue) | replace_node |
| `storage_timeout` | pause_job (don't burn compute on dead storage) | pause_job |
| `correlated_incident` | **freeze_automation** (no retry storms) | freeze_automation |

The most important action in the table is `freeze_automation`: during a
correlated incident, the policy's job is to know when to **stop** acting.

---

## Putting it together: the ConvNet crash

The kill-and-resume run from the ConvNet tutorial produces exactly this trail
in `run.events` (`FAILURE_CLASSIFIED` is amber and `RECOVERY_ACTION_SELECTED`
is cyan on the live page):

```
LAUNCH_STARTED           task-000 launched (pid ...)
FAILURE_CLASSIFIED       task-000: worker_crash (exit 1)
RECOVERY_ACTION_SELECTED task-000: restart_group — a lost rank stops the group —
                         restart all workers from latest valid checkpoint
LAUNCH_STARTED           task-000-r1 launched (pid ...)
```

Reading it back through the pipeline: the `torchrun` child raised
`SystemExit(3)`; `torchrun` re-raised a `ChildFailedError` (so rules 2 and 3
did **not** fire) and exited non-zero → signals say "plain crash" → `classify`
returns `worker_crash` → `decide(worker_crash, "coordinated_training")` returns
`restart_group`. The retry (`task-000-r1`) keeps the **same** job id, so its
`ft.prepare()` finds the predecessor's newest valid manifest and resumes —
`resumed_from: 8`, not a restart from zero.

And the fail-fast case: introduce a `NameError` in the script and the terminal
traceback trips rule 2 → `application_error` → `fail_job`. FlashRuntime does
**not** spend your restart budget re-hitting a deterministic bug; it fails
immediately and tells you.

---

## Honest scope

- **This is the local signal surface.** `from_local_launch` sees an exit code
  and a log tail — nothing more. The richer classes (`node_loss`,
  `accelerator_failure`, `communication_error`, `correlated_incident`) are real
  and fully policy-covered, but they are evidenced by the distributed
  **coordinator** (leases, heartbeats, health signals), not by one local
  process. The taxonomy and policy table are the same in both places.
- **Recovery never changes the math.** A resumed run restores a verified,
  topology-compatible checkpoint and continues; determinism (fixed seeds, the
  seed-0 `DistributedSampler`) is what makes a resumed run reproduce the
  uninterrupted one.

---

## Where to go next

- **[Architecture](../concepts/architecture.md)** — recovery as one of the four
  axes, with the leases and manifests it relies on.
- **[SDK reference](../reference/sdk.md)** — `submit(..., max_restarts=N)` and
  the `Run` event log.
- **[JobSpec & isolation guide](../guides/jobspec-and-isolation.md)** — how the
  same recovery machinery runs on a coordinator, under leases.
