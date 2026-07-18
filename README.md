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

**Pre-release scaffold.** The package structure and security contract are
defined; the agent implementation is being built (July 2026).

## What it will do

```bash
pip install flashnode
flashnode join --code HACKATHON-2026
flashnode status
```

- Generate an Ed25519 node identity and register with the control plane.
- Report capabilities (CPU, RAM, disk, OS, optional GPU) and run short
  admission benchmarks.
- Maintain an **outbound-only** authenticated WebSocket — no inbound ports,
  no router configuration.
- Execute allowlisted container images non-root with explicit CPU, memory,
  disk, time, and network limits.
- Stream heartbeats, task progress, and telemetry; stage artifacts with
  content hashes and idempotent commits.
- Track accepted work units and contribution credits.

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

```
flashnode/
├── agent/       # control connection, lifecycle, CLI
├── identity/    # Ed25519 keys and registration
├── inventory/   # hardware/software discovery
├── benchmark/   # admission and workload probes
├── executor/    # sandboxed container execution
├── telemetry/   # heartbeats, metrics, logs
├── artifacts/   # input/output/checkpoint staging
└── config/      # host-owner policy and limits
```

## License

[Apache-2.0](LICENSE). Contributions via Developer Certificate of Origin
(`git commit -s`).
