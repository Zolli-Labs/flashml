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

## Check your machine first

```bash
flashnode doctor
```

Six checks: the `docker` CLI, the engine behind it, an anonymous pull of a
curated image, whether a container can see your work directory, whether your
Docker accepts the sandbox flags, and whether any directories you lend via
`FLASHNODE_LOCAL_DATA` are readable. Every failure names the fix.

Run it once before `flashnode work`. `work` repeats all of it **except** the
image pull — a registry blip should not stop an agent whose images are
already cached — and refuses to start if anything fails, because a host that
cannot run tasks should not be claiming them.

## While it runs

On a terminal you get a live status block:

```
flashnode 0.3.2 · flashml-api.onrender.com · up 2h14m
  running    fed-2e2d4d6ab57f  ·  attempt 1  ·  38s
  session    12 accepted   0 failed
  heartbeat  2s ago
```

`waiting · no work queued — this is normal` means exactly that: the pool has
nothing for you right now, and your machine is fine. Pipe the output
anywhere, or pass `--log-json`, and you get the machine-readable log instead.

If three tasks in a row fail **on your machine**, the agent re-runs its own
checks. Pass, and the jobs were broken rather than your host, so it carries
on. Fail, and it stops claiming and tells you what to fix — instead of
burning a job's retries on a machine that cannot run anything.
`--max-consecutive-failures 0` turns that off.

## What it does today

```bash
pip install -e .                      # plus: pip install -e ../flashruntime
flashnode doctor                      # check this machine can run tasks
flashnode work --coordinator http://<coordinator>:8100
# optional hardening / pool config:
#   FLASHNODE_JOIN_CODE=...          join-code-gated pools
#   --runner docker + FLASHNODE_ALLOWED_IMAGES=img:tag,...   container tier
#   FLASHNODE_WORKDIR=$HOME/.cache/flashnode   (macOS + colima: VM-visible workdirs)
#   FLASHNODE_WORKDIR=C:\Users\<you>\.flashnode  (Windows: must be under a
#                                                 directory Docker Desktop shares)
#   FLASHNODE_LOCAL_DATA=patients=/srv/data/patients-2026,labs=/srv/labs
#                                    lend local directories to tasks by LABEL
```

`FLASHNODE_LOCAL_DATA` lets you offer data **without uploading it**. Only the
label names (`patients`, `labs`) are advertised to the coordinator — never the
paths — and a task that names a label in its `local_inputs` gets that directory
bind-mounted **read-only** at `/work/inputs/<label>`. A task asking for a label
this machine does not lend is refused, not run half-fed; a task that asks for
nothing sees nothing, exactly as before.

If the coordinator enforces per-machine authentication
(`FLASHML_NODE_TOKENS` set server-side), save the bearer token you were
given before running `work`:

```bash
flashnode login --coordinator http://<coordinator>:8100 --token <token>
flashnode work --coordinator http://<coordinator>:8100   # reads the saved token automatically
flashnode logout --coordinator http://<coordinator>:8100 # forget it locally (does not revoke server-side)
```

`login`/`logout` write to a per-coordinator credential store at
`~/.flashnode/credentials.json` (override with `FLASHNODE_CREDENTIALS`),
keyed by coordinator URL so one machine can hold separate tokens for
separate pools. The file is written with mode `0600` on every save. A
missing or unparseable file is treated as "no saved token" rather than a
crash. `CoordinatorClient` sends the saved token as a bearer header on
every request to that coordinator once it's saved — there is nothing else
to configure. Token issuance is still manual and out-of-band today (the
coordinator operator hands you the token; there is no self-service signup
or browser device flow yet), and `flashnode logout` only removes the local
copy — the operator revokes access by removing your token from the
coordinator's configuration.

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

Supported host class (initial): x86-64 Linux, macOS (Docker Desktop or
Colima), or Windows (Docker Desktop with the **WSL2 backend**), Python
3.10+, ≥4 CPU cores, ≥8 GB RAM, stable outbound internet.

**Windows note:** `flashnode work` used to crash immediately on Windows
(`os.getuid`/`os.getgid` don't exist there). It now omits `--user` on
Windows instead, relying on the curated images' own non-root `USER`
declaration for non-root execution — see
[`docs/guides/donate-a-machine.md`](https://github.com/Zolli-Labs/flashruntime/blob/main/docs/guides/donate-a-machine.md#platform-support)
in flashruntime for the full picture, including honest caveats: **Windows
support is constructed-argv-verified (tests fake the platform), not yet
execution-verified against a real Windows machine.**

## Package layout

Working today:

```
flashnode/
├── agent/       # CLI (`work`, `agent`), K8s-profile daemon, kube helper
├── identity/    # stable node ID (Ed25519 signing: planned); credentials.py
│                #   is the per-coordinator bearer-token store behind
│                #   `flashnode login`/`logout`
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
