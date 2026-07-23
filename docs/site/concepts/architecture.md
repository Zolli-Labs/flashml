# Architecture

FlashRuntime plans, launches, observes, and recovers a distributed ML job. It
**never** reimplements the distributed math — that always belongs to your
framework (PyTorch DDP/FSDP, `torchrun`, Ray, Hugging Face). This page explains
the shape of the system: the four axes it is built from, and the three
mechanisms — leases, manifests, recovery — that make "runs to *verifiably*
completed on unreliable machines" a real guarantee rather than a slogan.

The design decisions here are recorded in ADR-0003 (*Reliability runtime first;
planner as an explainable feasibility filter*).

---

## The four orthogonal axes

The central idea: **getting machines, starting processes, configuring
execution, and integrating user code are four independent concerns.** Keeping
them orthogonal is what lets the same job run on your laptop, on RunPod, or on a
community pool without rewriting anything.

```
   ┌──────────────────────────────────────────────────────────────┐
   │                     your training job                          │
   └──────────────────────────────────────────────────────────────┘
        │             │              │                  │
        ▼             ▼              ▼                  ▼
   ┌─────────┐   ┌──────────┐   ┌────────────┐   ┌──────────────┐
   │providers│   │launchers │   │ strategies │   │   recipes    │
   │─────────│   │──────────│   │────────────│   │──────────────│
   │  get    │   │  start   │   │ configure  │   │  integrate   │
   │machines │   │processes │   │ execution  │   │  user code   │
   └─────────┘   └──────────┘   └────────────┘   └──────────────┘
   RunPod,       torchrun,      DDP / FSDP2 /    PyTorch, sklearn,
   local,        local proc,    zero3 offload,   Hugging Face
   K8s pools     leased node    single-GPU       (this is the
                                                  workload layer)
```

The rule that keeps them honest: **Hugging Face / PyTorch code lives in the
recipe (workload) layer — it is never a backend.** The planner emits a
backend-neutral `StrategyPlan` and never imports framework code (no `import
transformers` / `torch.distributed` / `ray` inside the planner). Launching is
orthogonal to the strategy your code chooses, which is why FlashRuntime can
launch an FSDP or DeepSpeed script correctly without knowing anything about
FSDP or DeepSpeed.

The `integrations.*` adapters you use from the SDK (`fr_torch.ddp`,
`fr_sklearn.sweep`, `fr_hf.trainer`) are the recipe axis in practice: each
returns a `CommandWorkload` describing *what to run*, and the other three axes
handle the rest.

---

## Leases — the Mode A reliability core

A **lease** is a time-bounded right to run one task. It is the layer no
existing distributed-ML library provides, so FlashRuntime builds it first
(Mode A) before coordinated training (Mode B). A node *claims* a task, sends
*heartbeats* to keep the lease alive, and *commits* the result idempotently; if
the heartbeats stop, the lease *expires* and the task *requeues* — automatic
recovery with no central decision required.

```
   task: PENDING
      │  claim  (a node takes a time-bounded lease)
      ▼
   LEASED ──heartbeat──► LEASED ──heartbeat──► LEASED
      │                                           │
      │ no heartbeat within TTL                   │ commit (validated, idempotent)
      ▼                                           ▼
   EXPIRED ──requeue──► PENDING              COMPLETED
```

Key properties:

- **Status is derived from an append-only ledger of events**, never a
  hand-mutated field. "What is this job's state?" is always answered by
  replaying events, so the answer is reproducible and auditable.
- **Commit is idempotent and validated.** A result is accepted only if it
  matches the task's expected commit key (a sha256), so a duplicate or corrupt
  commit cannot poison the job.
- **Lease state is durable.** In-flight leases survive a coordinator restart
  (a SQLite-backed store); agents re-register on their own.

---

## Manifests — checkpoint validity by construction

A checkpoint is only useful if you can trust it after a crash. FlashRuntime
makes validity **structural** with a parts-first / manifest-last commit: the
checkpoint's parts are written first, then — only after their hashes verify —
the **manifest** is written last. No manifest, no checkpoint.

```
   write step-000123/
     ├─ model.pt         (part)   ─┐
     ├─ optimizer.pt     (part)    │  written FIRST
     └─ ...                        ─┘
                                    │  hashes verified
     └─ manifest         ──────────┘  written LAST  ✔ now "latest_valid"

   crash between parts and manifest  ⇒  no manifest  ⇒  never selected
```

So a half-written checkpoint — the exact thing a crash tends to produce — can
never look valid. Recovery restores only a **verified, topology-compatible**
manifest (the newest whose world size and framework match), which is why a
resumed run continues correctly instead of loading garbage. `ft.checkpoint(...)`
in the [torch helper](../reference/torch-helper.md) writes under this contract;
the Hugging Face callback commits Trainer checkpoints the same way.

---

## Recovery — typed, deterministic, logged

When something fails, recovery is a **pure function of the evidence**, not a
judgment call. Raw signals are classified into one failure class, and the class
(with the execution mode) is looked up in a versioned policy table that returns
a typed action. There is no LLM, no scoring, no learned model — same failure +
same policy version ⇒ same action, always.

```
   failure evidence            classify()              decide(class, mode)
   ────────────────      ─────────────────────      ──────────────────────────
   exit code,            precedence-ordered:         table lookup, versioned:
   log tail,       ───►  systemic > node >     ───►  worker_crash + coordinated
   heartbeat loss,       process > app               → restart_group (from ckpt)
   health signals        → one FailureClass          app_error → fail_job (fast)
                                                      correlated → freeze_automation
```

The design commitments:

- **Deterministic application errors are never retried** — fail fast and tell
  the user. Burning capacity re-hitting a bug is the most expensive kind of
  "recovery".
- **Blast radius depends on mode.** A `worker_crash` costs one task retry in
  `independent_tasks` mode but a whole-group restart in `coordinated_training`
  (NCCL collective state is not repairable in place).
- **Correlated incidents freeze automation.** The policy's most important
  action is knowing when to *stop* acting — retry storms during a systemic
  incident are how orchestrators destroy trust.
- **Every decision is logged** with its failure class and human-readable
  reason, and emitted as `FAILURE_CLASSIFIED` / `RECOVERY_ACTION_SELECTED`
  events the live page and ledger both show.

The [fault-tolerance tutorial](../tutorials/fault-tolerance.md) walks one real
crash through this pipeline end to end.

---

## How it fits together

```
   plan  ──►  launch  ──►  observe  ──►  recover
   (StrategyPlan,    (providers +   (leases +        (classify + decide,
    explained)        launchers)     heartbeats +     typed actions from
                                     manifests)       the policy table)
```

The runtime is the spine; the planner is an explainable feasibility filter that
sits in front of it (`flash.plan()`), and the runtime's ledger is the planner's
dataset. Everything above is usable **without the cloud** — a self-hosted local
coordinator is a first-class mode, not a demo shim.

See also: the [SDK reference](../reference/sdk.md) for the entry points, and the
[JobSpec & isolation guide](../guides/jobspec-and-isolation.md) for the
coordinator wire form.
