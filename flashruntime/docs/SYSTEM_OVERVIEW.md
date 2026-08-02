# FlashML System Overview

> **This document is the shared source of truth for the whole FlashML product
> system.** The canonical copy lives in `flashruntime/docs/SYSTEM_OVERVIEW.md`;
> the copies in `flashnode` and `flashml-cloud` are synced from it — edit the
> canonical copy and run `make sync-docs` from the workspace root, never edit
> a synced copy directly. It distills the *FlashML Master Product Architecture
> and Strategy Report* (July 2026).

---

## 1. What FlashML is

FlashML turns fragmented computers — idle desktops, lab workstations, private
clusters, and cloud provider capacity — into one fault-tolerant
machine-learning platform.

- Hosts install an **open-source node agent** and contribute compute.
- Developers submit **fault-tolerant distributed ML jobs** through an
  **open-source runtime**.
- A **private managed cloud** schedules the work, recovers when machines
  disappear, validates outputs, and tracks both the value delivered to users
  and the contribution of every host.

The product does **not** sell "access to a machine." It sells **completed
useful work**: a finished sweep, a processed dataset, a trained adapter, a
validated result — despite node failures. The core success metric is
**completed useful work per dollar and per host-hour**.

**Identity note:** FlashML here means Zolli Labs' fragmented-compute platform.
It is unrelated to Apache Spark, Hadoop, the old 247.ai "FlashML" artifact, or
FlashLib. Public materials lead with the component brands **FlashNode**,
**FlashRuntime**, and **FlashML Cloud** under the **Zolli Labs** umbrella.

## 2. The three-product system

| Product | Repo | Visibility | Role |
|---|---|---|---|
| **FlashNode** | `Zolli-Labs/flashnode` | Public (Apache-2.0) | Host agent installed by resource contributors: identity, inventory, benchmarking, sandboxed execution, telemetry, artifact staging, contribution tracking |
| **FlashRuntime** | `Zolli-Labs/flashruntime` | Public (Apache-2.0) | The **reliability runtime** (job specs, task graphs, leases, heartbeats, checkpoint manifests, failure taxonomy, recovery state machine, adapters, CLI/SDK, self-hostable coordinator) plus a **strategy planner** — an explainable feasibility filter that turns model + hardware + budget + deadline into a ranked, explained execution plan. It plans, launches, observes, and recovers; it never reimplements distributed ML |
| **FlashML Cloud** | `Zolli-Labs/flashml-cloud` | Private (proprietary) | Managed control plane: identity, matching, scheduling, trust and economic policy, billing/credits/payouts, reliability graph, enterprise controls, web dashboard |

Why open/open/closed: the agent runs on other people's machines, so it must be
inspectable (trust requirement + distribution strategy). The runtime must be
credible and self-hostable so the protocol is adoptable. The cloud is where
compounding advantages live — marketplace density, cross-node reliability
history, fraud signals, billing operations.

**Boundary principle:** the open runtime must be genuinely useful without the
cloud. The cloud wins by operating the network better, not by crippling the
public repositories.

## 3. Repository boundaries and the dependency rule

```
flashml-cloud ──┐
                ├──> imports flashruntime (versioned protocol package)
flashnode ──────┘

flashruntime imports NEITHER application repository.
flashnode and flashml-cloud never import each other.
```

- `flashruntime` owns the public protocol package and versioned schemas.
- `flashnode` imports it to register, receive leases, and report events.
- `flashml-cloud` imports the same package to validate API requests and
  control messages. It may add internal fields/services, but public-node
  compatibility must never depend on private code.
- Anything FlashNode needs to talk to the control plane lives in
  `flashruntime`'s protocol package — never copied into `flashml-cloud`.
- Every job spec, node message, and checkpoint manifest carries a schema
  version (`flashml/v1alpha1` style). Security-relevant fields fail closed.

Local development uses editable installs across sibling clones
(`uv pip install -e ../flashruntime`).

## 4. Architecture: logical planes

| Plane | Components | Responsibility |
|---|---|---|
| Interface | CLI, Python SDK, web dashboard, REST API | Submit jobs and policies; inspect progress, recovery, cost, contribution |
| Managed control | Identity, registry, queue, scheduler, policy engine, event ledger | Validate, match, observe, decide, meter, audit |
| Runtime control | Job graph, leases, heartbeats, checkpoint catalog, recovery state | Consistent execution semantics across environments |
| Adapter | Community pool, RunPod, Vast.ai, SkyPilot, Kubernetes/Kueue, Slurm, Ray | Provision/select capacity; translate provider events |
| Execution | FlashNode + runtime integration inside workers | Execute signed tasks, stage artifacts, emit telemetry |
| Storage | S3-compatible stores, managed artifact store, local cache | Preserve inputs/outputs/checkpoints independently of node lifetime |

**Design invariants:** any node is replaceable without losing the control
plane's understanding of the job; any adapter is replaceable without changing
the public job model. Nodes are disposable; state is not.

Durable state objects: User, Organization, Node, NodeCapabilitySnapshot, Job,
JobAttempt, Task, TaskAttempt, Lease, Heartbeat, CheckpointManifest, Artifact,
FailureEvent, RecoveryAction, UsageRecord, ContributionRecord, Settlement.
An event ledger records every state transition and automated decision.

## 5. Workload model

**Mode 0 — local single-process** (first-class, not a degenerate case): the
planner's fallback answer ("your job fits on one GPU; distribution would
cost money for nothing"), the profiling vehicle, and the debugging story. A
planner that cannot recommend *not distributing* will over-distribute.

**Mode A — fragmented independent tasks** (first and primary distributed
mode): hyperparameter search, cross-validation, preprocessing shards,
offline batch inference, synthetic-data generation, sharded K-means,
single-node LoRA trials. A slow node affects only its task; a lost node
causes lease expiry and reassignment. This is the lease/heartbeat protocol
FlashRuntime owns outright — no existing library provides it for machines
that don't share a network.

**Mode B — coordinated training** (controlled environments only): DDP/FSDP
jobs inside one provider, data center, or low-latency pool. PyTorch Elastic
restarts *all* workers on membership change, so the promise is **controlled
restart with bounded lost work**, never "zero-downtime." Cross-provider
synchronous training stays a research track. Recovery here is *borrowed and
recorded*: Kubernetes replaces machines, the training stack restarts,
FlashRuntime owns checkpoint selection and the audited timeline.

**Mode C — elastic/semi-synchronous training** (reserved in the schema only):
per-step fault tolerance without whole-group restart (torchft-class HSDP,
LocalSGD/DiLoCo). Not built; the StrategyPlan's `recovery_model` enum
reserves it so it becomes an adapter later, not a schema break.

CPU-first is a validation and distribution wedge, not the endgame. Nothing in
the public job or node model may hard-code CPU assumptions.

**Library reuse stance** (decided July 2026; rationale in the workspace-root
`FLASHRUNTIME_EVALUATION.md` and ADR-0003): build **on** torchrun/Elastic
(Mode B launcher), DDP + FSDP2 (strategy families; FSDP1 skipped —
deprecated upstream), PyTorch Distributed Checkpoint (checkpoint backend;
resharding on load), Ray Core (cluster Mode A backend), Hugging Face
Transformers + PEFT (recipes layer), DeepSpeed later (only for what FSDP2
lacks: NVMe offload, MoE). Deliberately **not** built on: HF Accelerate
(overlaps torchrun/strategy config — two owners of one decision) and Ray
Train (mid V1→V2 migration; too unstable for a public abstraction).

## 6. Fault tolerance

Leases + heartbeats for independent tasks: a lease grants one node the right
to execute a task attempt for a bounded period, renewed by heartbeat. Expiry
⇒ new attempt on another node. Only one attempt commits; late duplicates are
rejected.

Failure taxonomy → typed default actions: application error (fail/policy
retry), worker crash (restart attempt), node loss (expire leases, cordon,
reassign), accelerator failure (cordon, replace), network degradation (retry /
move / restart group), artifact corruption (reject commit, retry elsewhere,
lower score), preemption (acquire replacement, restore), correlated incident
(stop automation, escalate).

Checkpoints are **manifests**, not paths: job/attempt IDs, step, world size,
rank objects, dataset cursor, hashes, validation status. Recovery restores
only from complete, topology-compatible manifests. Recovery actions are
deterministic, typed, and logged — no ML-driven recovery until enough labeled
events exist, and never an unauditable LLM agent.

Fault tolerance that returns wrong results is worse than failure: outputs
commit through temporary keys with content hashes and workload validators;
training recovery validates checkpoint completeness, step monotonicity, and
metric continuity. Escalate when safety can't be proven.

## 7. Scheduling and economics

Three independent node assessments, never merged into one number:

- **Capability score** — can this node run the workload?
- **Reliability score** — how likely is it to finish correctly?
- **Trust tier** — what data and actions may be placed here?

Objective: minimize **expected cost per completed job** = compute + cold
start + transfer + checkpoint overhead + expected failure loss + retry cost +
platform fee. User presets: Cheapest / Balanced / Fastest / Trusted.

Contribution accounting distinguishes reserved time, attempted compute, and
**accepted useful work**; hosts are paid only for verified results. The moat
flywheel: more jobs → more failure/recovery observations → better placement
and recovery → better completed-work economics → more jobs.

## 8. Security and trust (staged)

Threat model: protect host from workload, workload/data from host, control
plane from malicious nodes/users, marketplace from fraud. No single sandbox
solves all four.

Isolation tiers: (1) baseline Docker/rootless OCI + cgroups + allowlisted
images for trusted participants and public data; (2) hardened community
(+gVisor); (3) professional pools (Kata/microVM); (4) confidential enterprise
(attestation). Community nodes initially process only public/synthetic data.

FlashNode security requirements: outbound-only control connection (no inbound
ports), signed node identity (Ed25519), short-lived session credentials,
allowlisted images, non-root execution with explicit resource limits, no
host Docker socket or privileged mode, complete event logging.

## 9. Technical stack (first version)

| Area | Choice |
|---|---|
| FlashNode | Python 3.12, asyncio, psutil, Docker SDK, WebSocket client, Ed25519 identity |
| FlashRuntime | Python, Pydantic schemas, FastAPI local coordinator, Redis leases, SQLAlchemy/Postgres state |
| Cloud web | Next.js, TypeScript, Tailwind, React Query, Recharts/React Flow |
| Managed API | FastAPI, Postgres, Redis, background workers, WebSocket/SSE |
| Storage | S3-compatible object storage + content hashes |
| Observability | Structured JSON logs, OpenTelemetry, Prometheus-compatible metrics |
| Identity | Supabase/Clerk auth; internal non-cash credits first |

## 10. Current milestone: the local loop is DONE (July 2026)

Two proof points now exist, both recorded in the workspace-root
`PROGRESS.md`:

1. The original POC (2026-07-17/18, `archive/POC_REPORT.md`): the Mode B loop on
   simulated devices — submit → KubeRay → kill a worker → automatic
   replacement and retry → durable artifacts → dashboard timeline.
2. **The local milestone (2026-07-19): the complete Mode A system, cloud-
   free**, implemented in the repos and proven by the workspace `e2e/`
   suite across real process/network boundaries:
   - Lease coordinator over HTTP with idempotent, **commit-time-validated**
     results (uploaded output must hash to the claim; bad output = failed
     attempt, requeued — never accepted).
   - Durable SQLite lease state: **in-flight leases survive coordinator
     restarts**; agents re-register automatically.
   - FlashNode device executor, two tiers (allowlisted subprocess with
     scrubbed env; allowlisted Docker with `--network none`, cpu/mem
     limits, read-only rootfs), join codes, coordinator-hosted local
     artifacts, built-in dashboard.
   - Three workloads: hyperparameter search (kill-a-machine sweep, 12/12
     exactly-once with per-node credit), sharded K-means (Lloyd iterations
     as consecutive lease jobs, converges across two agents), and a
     **checkpointable trainer with bit-identical resume** — machine B
     crashes at step 35, machine A resumes from B's relayed step-30
     checkpoint; the ledger reports 5 steps lost, not 35.

Remaining from the local plan: Stage-8 ledger metrics (MTTD/MTTR/goodput)
+ case study, and a second *physical* machine run (runbook in
`e2e/README.md`). Next horizon: Stage 5 — the Alibaba deployment (ECS →
ACK), where the coordinator's address changes and the code does not.

Deferred deliberately: global marketplace, arbitrary GPU/OS support,
internet-wide synchronous training, learned scheduling, real-money payouts,
gVisor/Kata isolation, zero-downtime claims.

## 11. Roadmap after the milestone

- **Weeks 1–4 (pilot foundation):** Postgres state + event ledger, cleaner
  CLI/spec, artifact store with idempotent commit, capability snapshots +
  basic reliability score, 3 external workloads, 5 active hosts, case study.
- **Weeks 5–12 (usable platform):** orgs/auth/API keys, first provider
  adapter (RunPod / Vast.ai / Kubernetes), trust tiers + placement presets,
  metering, one controlled PyTorch LoRA recovery path, first paying pilot.
  Plus the standalone planner wedge: `flash plan` for single-node
  fine-tuning — model + GPUs + budget in, ranked explained plans (with
  rejections and the arithmetic) out, useful with zero cluster attached.
- **Months 3–6:** PyTorch Distributed Checkpoint manifests, certified
  FSDP/DeepSpeed or Ray Train envelope, checkpoint scoring, SkyPilot/Kueue/
  Slurm adapter by demand, BYOC deployment mode.
- **Months 6–12:** cross-provider reliability graph, professional operator
  pools, public failure benchmark suite, enterprise policy/SSO/audit.

## 12. Guardrails (what FlashML must not become)

- Not a generic GPU marketplace; not "OpenRouter for GPUs"; not a universal
  any-device promise; not a multi-cloud launcher clone; not a blockchain.
- No proprietary or regulated data to unknown community hosts.
- No "zero-downtime" or in-place rank-repair claims before demonstrated.
- No LLM agent at the center of recovery — typed, authorized, logged actions.
- Success is measured by completed useful work, repeat usage, and willingness
  to pay — not registered nodes or GitHub stars.

## 13. Terminology

Use: *fragmented compute network, open host agent, fault-tolerant distributed
runtime, completed useful work / goodput, controlled restart with bounded
lost work, managed control plane, recovery policy engine, reliability graph.*

Avoid: *any-device cloud, mining software, retry script, raw uptime,
zero-downtime recovery, new cloud provider, autonomous infrastructure agent,
provider leaderboard.*

Key terms: **accepted work** (validated + committed output), **lease**
(time-bounded right to execute a task attempt), **goodput** (share of paid
wall-clock producing valid progress), **trust tier** (policy class governing
data placement), **BYOC** (customer-controlled infrastructure).
