# CIFAR-10 fragmented-cluster demo — design

Date: 2026-07-23 · Status: approved (brainstorming) · Branch: local-milestone-2026-07

## Purpose

A local-driven showcase demo that proves FlashRuntime's value on a realistic
workload across a **genuinely heterogeneous, fragmented cluster**:

- **Your Mac (Apple M4)** runs the coordinator **and** works (MPS).
- **A rented RunPod box (1×RTX 4090)** joins as a worker (CUDA) over outbound
  HTTP through a reverse SSH tunnel.

The demo runs a real CIFAR-10 CNN hyperparameter sweep, fans it across both
machines, and injects real failures (a hard node kill; a deterministic bug) so
the operator can *watch* — in our own dashboard/viewer — the product survive:
lease expiry → requeue → cross-machine, cross-accelerator resume, and
fail-fast on an application bug.

It is a **showcase demo, not a CI test**: optimized for being watched live by
the operator, correctness over a stopwatch (~10 min is a soft target). It still
leaves behind a self-verifying `demo_report.md` with assertions, so the demo
cannot silently lie.

Non-goals (explicitly dropped during brainstorming):
- Not a laptop-CPU-portable demo (real GPUs are the point).
- Not multi-node data-parallel training of a single model over the internet
  (the product does not claim it; it would be a dishonest demo).
- Not a benchmark row and not a pytest acceptance test.
- Does **not** stage `ACCELERATOR_FAILURE` / `NODE_LOSS` signals the SDK-local
  path cannot honestly evidence — only failure classes actually reachable
  through the coordinator + local launcher are demonstrated.

## Architecture

```
   YOUR MAC (M4)                                  RUNPOD POD (1×4090)
   ┌────────────────────────────┐                 ┌──────────────────────┐
   │ coordinator :8100          │                 │ flashnode work       │
   │  ├─ lease manager          │◀───outbound─────┤  --coordinator       │
   │  ├─ node registry          │     HTTP        │   http://127.0.0.1:  │
   │  ├─ artifact hosting       │                 │        8100          │
   │  └─ dashboard  GET /       │                 │                      │
   │                            │                 │ claims → trains on   │
   │ flashnode work  (worker A) │                 │ cuda:0   (worker B)  │
   │  claims → trains on MPS    │                 └──────────────────────┘
   └────────────────────────────┘                            ▲
                │                                            │
                └────── ssh -R 8100:127.0.0.1:8100 ──────────┘
                        (pod's localhost IS your Mac's coordinator)
```

### Why this topology

- **Pull model, capability-blind.** Both workers claim from the same queue. The
  coordinator knows nothing about GPUs (node GPU reporting is a documented
  flashnode follow-up, `inventory/capabilities.py:91`, and capability-aware
  placement is research item R9 — both deliberately out of scope here). The
  4090 finishes trials faster, comes back, and claims more, so an uneven,
  honest split *emerges* from work-stealing. That emergent split — visible via
  each trial's reported `device` — is a core thing to watch.
- **Reverse SSH tunnel** (`ssh -R 8100:127.0.0.1:8100`), not ngrok / public
  bind:
  - Nothing about the Mac is exposed to the internet (no public bind, no router
    change).
  - `flashnode` runs **completely unmodified**, over its real outbound-HTTP
    code path, dialing a localhost address.
  - The join code (`FLASHNODE_JOIN_CODE`) still gates registration, so the auth
    path is exercised too.
- **Dataset distribution:** each worker downloads CIFAR-10 via `torchvision`
  into its **own local cache** — NOT through the coordinator's artifact
  endpoint (pushing 170 MB through the tunnel proves nothing and is slow). Two
  internet-connected machines each fetching a public dataset is what a real
  user does. The coordinator's artifact endpoint still carries what it is for:
  per-trial `metrics.json` outputs and relayed checkpoint files coming back.

### What already exists (reused unmodified)

Verified during brainstorming — this is assembly, not new construction:

- Coordinator on `0.0.0.0:8100` + join code + dashboard (`make local-coordinator`).
- Mac as coordinator **and** worker (`make local-agent`).
- `flashnode work --coordinator …` pull loop: claim → download → run
  (heartbeating) → upload → commit (`flashnode/executor/loop.py`).
- Lease expiry → requeue; late/duplicate commits rejected
  (`leases/manager.py`), proven in `e2e/test_local_loop.py`.
- **Cross-machine resume already works** — `e2e/test_training_resume.py`
  ("Stage 7's money demo"): a worker dies at step 35, another downloads the
  relayed step-30 checkpoint from the coordinator and finishes. Fetch logic in
  `executor/loop.py:167-175`.
- `hyperparameter_search` expansion already accepts `module:` and `grid:`
  parameters (`service/modea.py:89-99`), so the sweep needs **no coordinator
  expansion changes**.
- RunPod rent/teardown harness (`scripts/runpod_gpu_e2e.py`): one 2-GPU box (we
  need only 1 GPU here), 30-min hard cap, SSH via ephemeral ed25519 keypair,
  `finally:` teardown, `<$1` target.

## Components

### New: `flashml_workloads/cifar_cnn.py`

Same executor contract as its three siblings
(`sklearn_trial` / `kmeans_shard` / `sgd_trainer`):

```
python -m flashml_workloads.cifar_cnn --spec spec.json --out OUTDIR
```

`spec.json` (written by the executor from the lease payload):

```jsonc
{
  "task_id": "trial-007",
  "params": {
    "lr": 0.05, "arch": "resnet8", "epochs": 3, "batch_size": 128,
    "seed": 0, "checkpoint_every": 100,
    "kill_at_step": null          // deterministic crash hook; fires only on a
                                  // FRESH start so a resumed retry never re-crashes
  },
  "inputs": { "resume": "/abs/path/resume.pt" }  // present ONLY on a retry
}
```

Outputs:
- `OUTDIR/metrics.json` — the commit artifact:
  `{accuracy, final_loss, device, hostname, epochs, steps, resumed_from,
    wall_s, params}`.
- `OUTDIR/ckpt/step-NNNNNN.pt` — **one file per checkpoint**
  (`{model, opt, step}`), picked up by the executor's relay as it appears.
  Single-file (not a parts directory) because the relay resumes from
  `manifest["parts"][0]` (`executor/loop.py`).

Behavior:
- Device selection: `cuda` → `mps` → `cpu`, reported in `metrics.json.device`
  (the field that makes the emergent split legible: `cuda:0@pod` vs
  `mps@your-mac`).
- On a retry, if `inputs.resume` is present, load it with
  `torch.load(..., map_location=<this device>)` and set `resumed_from` — the
  cross-accelerator round-trip (MPS-saved ↔ CUDA-loaded) is handled here.
- Deterministic-bug path: an unknown `arch` value raises `ValueError` with a
  real traceback → classified `APPLICATION_ERROR` by `recovery/signals.py`.
- `torch` / `torchvision` imported **inside functions only** — the module
  import stays light and the core stays pydantic-only. Heavy deps live in a new
  optional extra, never a core dependency.

Supported `arch`: `resnet8`, `cnn3`, `resnet14` (small, CIFAR-appropriate).

### Changed files — the entire product-code footprint

| File | Change |
|---|---|
| `flashml_workloads/cifar_cnn.py` | **new** — the task module |
| `flashruntime/service/modea.py:51` | +1 line: allowlist `flashml_workloads.cifar_cnn` in `ALLOWED_TASK_MODULES` |
| `flashnode/flashnode/executor/runner.py:26` | +1 line: same, in `DEFAULT_ALLOWED_MODULES` (sibling repo — allowed; it imports flashruntime) |
| `pyproject.toml` | new optional extra (e.g. `[vision]`) for `torch` + `torchvision` |

Two one-line allowlist edits; everything else (expansion, leasing,
heartbeating, relay, resume, dashboard) runs unmodified.

### New: `demo/cifar_cluster/` (the demo, not product code)

- `run.py` — narrates each act, keeps the dashboard open, sequences the acts,
  supports running a single act, and writes `demo_report.md`.
- `report.py` — turns collected act results into `demo_report.md` with hard
  numbers **and assertions that fail loudly** when a claim does not hold.
- Harness integration: a reverse-tunnel step added to (or alongside)
  `scripts/runpod_gpu_e2e.py` so the pod's `flashnode work` reaches the Mac's
  coordinator.

### The sweep job

`hyperparameter_search`, `module: flashml_workloads.cifar_cnn`:

```
grid: { lr: [0.2, 0.1, 0.05, 0.02], arch: ["resnet8", "cnn3", "resnet14"] }
```

→ 12 trials, `lease_seconds: 45`, `checkpoint: {}` (non-None switches the relay
on). One trial in act 3 is given a bad `arch` to force the deterministic-bug
path.

## Demo acts (data flow)

| Act | Injection | Real code path | What the operator watches |
|---|---|---|---|
| **0 · Baseline** | none | 12 trials fan across Mac(MPS)+pod(CUDA) | uneven split emerges — pod claims more; each trial's `device` proves who ran what |
| **1 · Node death → requeue** | real `SIGKILL` to the pod's `flashnode` mid-trial (over SSH, no goodbye) | lease stops renewing → sweep expiry → `LEASE_EXPIRED` → `TASK_REQUEUED` → resume on Mac from relayed checkpoint | in-flight trial goes red, requeues, finishes on the other machine; ledger reports steps-lost, not all |
| **2 · Cross-accelerator resume** | (same trial as act 1) | CUDA-saved `step-NNNNNN.pt` → `map_location` → MPS load | `resumed_from > 0`, trial still completes — the heterogeneous round-trip verified |
| **3 · Deterministic bug → fail-fast** | one trial gets `arch: "nonexistent"` | module raises → `APPLICATION_ERROR` → task exhausts `maxTaskAttempts` → `TASK_EXHAUSTED` | one trial fails cleanly without a retry storm; other 11 finish |

Acts 1 and 2 are the same event observed two ways (a kill that also happens to
cross accelerators). Act 3 is the "knows when to stop" contrast.

## Error handling / honesty guardrails

- Only failure classes reachable through the real coordinator + local launcher
  are shown. `NODE_LOSS`/`ACCELERATOR_FAILURE`/`PREEMPTION` are NOT staged.
- Node death is a real OS `SIGKILL`; the deterministic bug is a real raised
  exception — no synthetic `FailureSignals`.
- The relay is best-effort by contract; a resumed run picks the latest *valid*
  relayed checkpoint (parts-first / manifest-last), so a half-shipped
  checkpoint can never be selected.
- `demo_report.md` assertions fail loudly if: cross-machine resume did not
  occur (`resumed_from == 0`), lost work is unbounded, the sweep did not reach
  11/12, or the bad trial retried instead of failing fast.

## Testing / validation strategy (cheap-first, subagent-built)

Each unit is small, independently testable, handed to a cheap subagent
(Haiku/Sonnet) with a tight contract; the diff is reviewed and the check run
before moving on.

| Unit | Agent tier | Cheap validation |
|---|---|---|
| `cifar_cnn.py` task module | Haiku | `python -m flashml_workloads.cifar_cnn --spec … --out …` on a tiny CIFAR subset, CPU — assert `metrics.json` shape + a `ckpt/step-*.pt` appears; a resume run sets `resumed_from > 0` |
| 2 allowlist edits + `[vision]` extra | Haiku | existing `pytest` stays green (264 passed, 1 skipped, 9 deselected); core-import smoke still pydantic-only |
| reverse-tunnel harness step | Sonnet | `--plan-only` dry run + a localhost loopback rehearsal (no pod) |
| demo driver + report | Sonnet | run the full acts 0/1/3 against **two local `flashnode` procs on the Mac** (no pod, no money) before any 4090 is rented |

The local two-worker rehearsal validates everything except real-GPU and
real-tunnel for **$0**. The 4090 run is the final confirmation, not the debug
loop.

## Open follow-ups (out of scope, noted)

- Capability-aware placement (R9): would let the coordinator route GPU-heavy
  trials to the 4090 by policy instead of by emergent work-stealing. This demo
  motivates it by showing the one thing it can't do (a slow MPS node grabbing a
  trial a GPU should have taken).
- Node GPU reporting in flashnode `inventory/capabilities.py` (prerequisite for
  R9).
