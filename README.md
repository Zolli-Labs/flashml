# FlashNode

> **The open host agent of the FlashML system.** Install FlashNode on a
> machine you own, and it can safely execute distributed ML tasks for the
> FlashML network — earning contribution credits for verified useful work.

FlashNode is one of three components in the FlashML system by
[Zolli Labs](https://github.com/Zolli-Labs):

- **flashnode** (this repo) — open host agent installed by resource
  contributors. Because it runs on someone else's machine and executes
  third-party workloads, it must be inspectable, minimal, and explicit about
  permissions — which is why it is open source.
- **[flashruntime](https://github.com/Zolli-Labs/flashruntime)** — the open
  workload protocol and execution layer.
- **flashml-cloud** (private) — the managed control plane and dashboard.

Read [`docs/SYSTEM_OVERVIEW.md`](docs/SYSTEM_OVERVIEW.md) for the full
product architecture, and [`AGENTS.md`](AGENTS.md) if you are an AI coding
agent working in this repo.

## Status

**Pre-release; the device executor works today** (July 2026). A machine
with this agent can join a FlashRuntime coordinator over outbound HTTP,
pull leased tasks, execute them, relay training checkpoints, and commit
verified results. Two profiles:

- **Device profile** (`flashnode work`) — the pull-based executor for
  laptops/workstations. Implemented.
- **Kubernetes profile** (`flashnode agent`) — per-node telemetry reporter
  inside managed pools (DaemonSet); KubeRay owns workload pods there.
  Implemented.

## What it does today

```bash
pip install -e .                      # plus: pip install -e ../flashruntime
flashnode work --coordinator http://<coordinator>:8100
# optional hardening / pool config:
#   FLASHNODE_JOIN_CODE=...          join-code-gated pools
#   --runner docker + FLASHNODE_ALLOWED_IMAGES=img:tag,...   container tier
#   FLASHNODE_WORKDIR=$HOME/.cache/flashnode   (macOS + colima: VM-visible workdirs)
```

- Stable node identity; registers with capabilities (CPU, RAM, arch, GPU)
  and **re-registers automatically** if the coordinator restarts.
- **Outbound-only** HTTP — no inbound ports, no router configuration.
- Claims task leases, renews them with attempt heartbeats, and stops work
  the moment a lease is refused (the coordinator's idempotent commit
  rejects late duplicates regardless — defense in depth).
- Two execution tiers behind one interface: `SubprocessRunner`
  (allowlisted Python modules, wall-clock timeout, **scrubbed
  environment** — agent secrets never reach task code) and `DockerRunner`
  (allowlisted images, `--network none`, cpu/memory limits, read-only
  rootfs, uid mapping).
- Downloads shared input artifacts; uploads outputs with sha256 for the
  coordinator's commit-time validation.
- **Checkpoint courier**: tasks stay network-isolated, so the agent
  fetches the task's latest valid checkpoint before a run (resume) and
  ships each new checkpoint file during it — a task killed on this
  machine resumes from its checkpoint on another.

Still to come: Ed25519-signed identity, admission benchmarks (`benchmark/`),
richer telemetry (`telemetry/`), gVisor/Kata isolation tiers, and the
`join`/`status`/`leave` UX.

## Security contract

- Outbound-only control connection; no inbound SSH or public ports.
- Signed node identity; short-lived session credentials.
- Allowlisted or signed workload images only.
- Non-root execution; no host Docker socket, device passthrough, or
  privileged mode.
- The agent shows exactly which limits and permissions apply to a workload
  before executing it.
- Complete event logging of task assignment, image digest, permissions, and
  artifact commits.

Supported host class (initial): x86-64 Linux, Python 3.10+, Docker or
Podman, ≥4 CPU cores, ≥8 GB RAM, stable outbound internet.

## Package layout

Working today:

```
flashnode/
├── agent/       # CLI (`work`, `agent`), K8s-profile daemon, kube helper
├── identity/    # stable node ID (Ed25519 signing: planned)
├── inventory/   # capability discovery (psutil + K8s allocatable)
└── executor/    # the device work cycle:
    ├── client.py         # stdlib outbound HTTP: leases, artifacts, checkpoints
    ├── runner.py         # Tier 1: allowlisted subprocess, scrubbed env
    ├── docker_runner.py  # Tier 2: allowlisted containers, network-none
    └── loop.py           # claim → run (heartbeating) → relay ckpts → commit
```

Scaffolds awaiting their vertical slice: `benchmark/` (admission probes),
`telemetry/` (rich metrics), `artifacts/` (local caching), `config/`
(host-owner policy).

## License

[Apache-2.0](LICENSE). Contributions via Developer Certificate of Origin
(`git commit -s`).
