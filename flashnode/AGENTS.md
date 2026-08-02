# AGENTS.md — flashnode

Context for AI coding agents (Claude Code, Codex) working in this repository.

## What this repo is

The **open-source host agent** of the FlashML system (Zolli Labs). It runs on
contributors' machines and executes third-party ML tasks, so it must be
inspectable, minimal, and explicit about permissions. Full product context:
`docs/SYSTEM_OVERVIEW.md`.

Sibling repos (cloned side-by-side under `~/Work/Zolli-Labs/`):
- `../flashruntime` — public protocol + runtime. **This repo will depend on
  it** for all wire schemas (add the dependency when the protocol package
  lands; install editable: `uv pip install -e ../flashruntime -e .`).
- `../flashml-cloud` — private control plane. We talk to it only through the
  versioned flashruntime protocol. **Never import it.**

## Hard rules

1. **This repo goes public at launch** (Apache-2.0). No secrets, no private
   business logic, nothing in history you wouldn't publish.
2. **Wire messages come from `flashruntime.protocol`** — never define a
   duplicate schema here, never copy one from flashml-cloud.
3. **Security contract is non-negotiable** (see README): outbound-only
   connection, non-root execution, allowlisted images, explicit resource
   limits, no host Docker socket, no privileged mode. Any change that
   loosens these needs explicit human sign-off.
4. The host owner sees exactly what a workload may do before it runs.
5. Trust through transparency: log every task assignment, image digest,
   permission grant, and artifact commit.

## Current state (July 2026)

Working Kubernetes-profile agent: `identity/store.py` (stable node ID on a
hostPath volume), `inventory/capabilities.py` (psutil + K8s allocatable +
allow-listed labels), `agent/kube.py` (stdlib-only in-cluster GET of own
Node), `agent/daemon.py` (register + heartbeat to FlashML Cloud, graceful
SIGTERM). Runs as `flashnode agent` in a DaemonSet (manifest in
flashml-cloud/infra/base/flashnode.yaml): non-root, read-only rootfs, no
runtime socket, RBAC = `get nodes` only. In the Kubernetes profile the agent
never executes workloads — KubeRay owns workload pods. Tests: `pytest` (28 — includes real-Docker smoke, auto-skipped without a daemon).

## The agent's two profiles

1. **Kubernetes profile** (works today, above): per-node telemetry reporter
   inside managed pools — register + heartbeat only.
2. **Device profile** (implemented July 2026, both tiers): `executor/` —
   `CoordinatorClient` (stdlib urllib, outbound only; sends
   X-FlashML-Join-Code from FLASHNODE_JOIN_CODE), `SubprocessRunner`
   (allowlisted modules, wall-clock timeout, **scrubbed env** — only
   PATH/HOME/PYTHONPATH/LANG/LC_ALL/TMPDIR reach task code),
   `DockerRunner` (allowlisted images from FLASHNODE_ALLOWED_IMAGES,
   `--network none`, cpu/mem limits, read-only rootfs, uid mapping;
   `--runner docker`; on macOS+colima set FLASHNODE_WORKDIR under $HOME —
   the VM only shares $HOME), `ExecutorLoop` (claim → download inputs →
   run with attempt-heartbeat thread → upload outputs + sha256 → complete;
   failures reported via fail(); lease loss discards results;
   **re-registers automatically** when a node heartbeat is refused after a
   coordinator restart), and the **checkpoint relay** (`_CheckpointRelay`:
   agent = courier because tasks are network-isolated — fetch latest
   manifest → `resume` input before the run; upload→register→commit each
   new `ckpt/step-*.json` during it, final flush on death), and
   **archive inputs** (`executor/archives.py`): an input named in the
   payload's `unpack_inputs` list is extracted to `inputs/<name>/` and
   handed to the runner as that *directory* — how a user's GitHub repo
   reaches the argv `python /work/inputs/code/<entrypoint>`, with the
   tarball's single wrapper directory stripped. Opt-in per input and never
   inferred from a filename the submitter chose; inputs not listed keep the
   plain-file behaviour byte for byte. The extractor refuses zip-slip
   (relative *and* absolute), escaping symlinks, decompression bombs
   (capped **during** extraction), member-count blowups, and
   device/fifo/hardlink members. It duplicates the cloud API's guard
   because of hard rule 2, and the drift that invites is caught by
   `e2e/test_archive_parity.py` — one attack corpus, both extractors. CLI:
   `flashnode work --coordinator URL`. Tests (116): stubbed transport +
   real-Docker (auto-skip); full loop + cross-machine training resume in
   workspace-root `e2e/`. Two heartbeats, never merged: attempt →
   coordinator (lease liveness), node → registry (online/offline). Wire
   models from `flashruntime.protocol` (hard rule 2).
   **Host doctor** (`doctor.py`, `flashnode doctor`): six checks — CLI on
   PATH, engine reachable, a curated image pulls, the workdir bind-mounts,
   the real `harden_args` flags are accepted, and every
   `FLASHNODE_LOCAL_DATA` label resolves to a readable directory. It also
   gates `flashnode work` fail-closed, replacing the old
   `shutil.which("docker")` — which BOTH hosts that stopped the 2026-08-02
   §10 run-through passed before failing every task they claimed. Checks
   take their subprocess/which call as a parameter, so the whole suite runs
   with no daemon. `work` passes `pull=False`: a registry blip must not stop
   an agent whose images are cached, so checks 4 and 5 use `--pull=never`
   and a fresh install must run `flashnode doctor` once.
   **NOT covered: mid-session breakage.** A host whose engine dies an hour
   in still claims and fails tasks exactly as before; the fix is server-side
   node quarantine in the coordinator, not here.
   **Missing/next**: Ed25519 identity, gVisor/Kata tiers,
   `join/status/leave` UX. `benchmark/`, `telemetry/`, `config/` now
   carry their complete designed interfaces (ABCs + contract tests in
   tests/test_interfaces.py; HostPolicy ships concrete with conservative
   defaults) — implement the concrete probes/collector against them.

## Dev workflow

```bash
uv venv && uv pip install -e ".[dev]"
flashnode            # usage; `flashnode work` runs the device executor
pytest               # 28 tests (real-Docker smoke auto-skips)
```

Python ≥3.10, asyncio, psutil, websockets, cryptography (Ed25519). Keep the
agent dependency-light — every dependency is attack surface on someone
else's machine.
