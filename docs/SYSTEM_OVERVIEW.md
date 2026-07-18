# FlashML System Overview

> **This document is the shared source of truth for the whole FlashML product
> system.** An identical copy lives in `docs/` of all three repositories
> (`flashnode`, `flashruntime`, `flashml-cloud`). If you change it, change it
> everywhere. It distills the *FlashML Master Product Architecture and
> Strategy Report* (July 2026).

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
| **FlashRuntime** | `Zolli-Labs/flashruntime` | Public (Apache-2.0) | The workload protocol and execution layer: job specs, task graphs, leases, heartbeats, checkpoint manifests, failure taxonomy, recovery state machine, adapters, CLI/SDK, self-hostable local coordinator |
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

**Mode A — fragmented independent tasks** (first and primary mode):
hyperparameter search, cross-validation, preprocessing shards, offline batch
inference, synthetic-data generation, sharded K-means, single-node LoRA
trials. A slow node affects only its task; a lost node causes lease expiry
and reassignment.

**Mode B — coordinated training** (controlled environments only): DDP/FSDP
jobs inside one provider, data center, or low-latency pool. PyTorch Elastic
restarts *all* workers on membership change, so the promise is **controlled
restart with bounded lost work**, never "zero-downtime." Cross-provider
synchronous training stays a research track.

CPU-first is a validation and distribution wedge, not the endgame. Nothing in
the public job or node model may hard-code CPU assumptions.

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

## 10. Current milestone (3-week build, July 2026)

The smallest complete loop, built as a real product rather than a staged demo:

1. Hosts install FlashNode (`pip install flashnode`, `flashnode join --code …`)
   and connect outbound; capability + benchmark registered.
2. A developer submits a distributed job (hyperparameter search or sharded
   K-means, 12 trials) via CLI/web.
3. FlashRuntime expands trials, leases them across nodes; live progress in
   the dashboard.
4. One host is killed mid-run. Heartbeat expires, lease invalidates, the task
   is reassigned, the job completes with every task exactly once.
5. The result page shows the recovery timeline and credits each host for
   **accepted** work only.

Acceptance criteria: onboarding without inbound ports; remote containerized
execution; ≥2 hosts concurrent; detection within configured interval;
automatic requeue and acceptance; exactly-once aggregation; visible
attempt/failure/reassignment timeline; usage + credits computed from accepted
work.

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
