# Volunteer compute pool: argv runner tier + composite lease keying

**Date:** 2026-07-28
**Status:** approved direction, spec for implementation
**Surfaces:** `flashnode/executor/` (new hardening helper + argv runner, agent CLI);
`flashruntime/` protocol (`NodeCapabilities`), `scheduler/` (placement gate),
`recipes/command.py` (tier validation), `leases/` (composite key).
**Slices:** A (argv runner tier) + E (composite `(job_id, task_id)` lease key)
from the multi-machine decomposition below.

## Goal

Let **any machine on the internet join as a compute resource**, pull tasks from
the coordinator, and execute a submitting user's *arbitrary code* inside a
sandbox strong enough that donating a machine is a reasonable thing to do.

Today a joined machine can only run `python -m <module>` from a fixed
`DEFAULT_ALLOWED_MODULES` allowlist — it can run FlashML's built-in workloads
and nothing else. The coordinator half of the argv contract already exists and
is documented as waiting on this work (`recipes/command.py` module docstring).
This slice ships the flashnode half.

### Decisions made with the user

1. **Trust model: public volunteers, untrusted.** Nodes may be operated by
   strangers; they may lie, vanish, or probe the coordinator.
2. **Open workloads only.** A volunteer node can read any code and data it
   receives, and that is accepted by construction. Confidential workloads are
   out of scope forever on volunteer nodes — they route to owned nodes.
   This removes any need for confidential computing (SGX/SEV-SNP/H100 CC).
3. **Result integrity = spot-check + reputation** — re-run a sample of tasks on
   a trusted node, compare a deterministic result hash, quarantine on mismatch.
   Chosen, but **deferred to slice C**; this slice ships no verification.
4. **Isolation is tiered with a hardened-Docker floor.** The node advertises
   what it can provide, the job declares what it needs, and the existing
   fail-closed placement gate matches them. gVisor/microVM are later tiers.
5. **Structure: additive runner** (approach 1 of 3). A new `ArgvDockerRunner`
   plus a shared `harden_args()` helper; the proven module runners are left
   alone. Converges on "everything is argv" later, once argv is proven on real
   volunteer hardware.
6. **Volunteer pool = independent tasks only.** Multi-node DDP is explicitly
   *not* the target (see Non-goals).

### The multi-machine decomposition (context for this slice)

| # | Slice | Status |
|---|---|---|
| **A** | argv runner tier — execute arbitrary user code, tier-gated | **this spec** |
| **B** | per-node Ed25519 identity + revocation (replaces shared join code) | next |
| **C** | result integrity — result hash, spot-check verifier, reputation, quarantine | after B |
| **D** | GPU capability probing (`capabilities.py` has `gpus=[]` today) | independent |
| **E** | composite `(job_id, task_id)` lease key | **this spec** |
| F | multi-node DDP rendezvous | not planned — see Non-goals |

## Non-goals

- **No multi-node DDP.** Gradient-synchronised training needs low-latency,
  high-bandwidth links and all ranks alive simultaneously; one volunteer
  closing a laptop stalls every other rank. Volunteer pools get independent
  tasks (sweeps, sharded work), where a lost node costs one task, not the run.
  `mode: "coordinated"` is rejected on volunteer nodes — with `--network none`
  ranks could not rendezvous anyway.
- **No result verification in this slice.** Slice C. A node that lies is
  currently believed. This is written into the docs as a known limitation, not
  left implicit.
- **No new authentication.** The shared `FLASHML_JOIN_CODE` remains the join
  gate for now; per-node identity is slice B. This slice must not make the auth
  situation *worse* (no new unauthenticated surface).
- **No confidentiality against the host.** Out of scope by decision 2.
- **No new coordinator endpoints, and no change to the task payload contract** —
  `CommandRecipe` already emits everything the runner needs. The one
  wire-visible change is a single additive, fail-closed field on
  `NodeRegistration` (registration only); see Architecture §4.
- **No GPU passthrough.** `--network none` plus no `--device` flags; GPU work on
  volunteer nodes waits on slice D.

## Architecture

### The existing contract (unchanged, for reference)

`CommandRecipe.expand()` already emits exactly the payload the runner needs:

```python
{"argv": [...],                       # normalized, placeholders already filled
 "env": {...},                        # per-task, placeholders already filled
 "inputs": {name: "artifact://..."},
 "output_prefix": "jobs/{job_id}/{task_id}/",
 "task_id": "task-000",
 "image": "<pinned reference>",       # schema already rejects ':latest'
 "isolation": {"tier": ..., "allowFallback": ...},
 "checkpoint": {...}}                 # optional
```

`ExecutorLoop.execute_one()` (`flashnode/executor/loop.py:144`) needs **no
change**: it already downloads `artifact://` inputs into the workdir, resumes
from the task's latest valid checkpoint, runs a checkpoint relay thread,
heartbeats throughout, uploads `outdir/*`, and commits with a sha256. It calls
`self.runner.run(payload, workdir, inputs) -> outdir`. `ArgvDockerRunner`
satisfies that same three-argument interface.

### Data flow (one volunteer task)

The agent is the courier; the job never touches the network.

```
coordinator                     volunteer machine
    |  claim ------------------> ExecutorLoop
    |                              |- download_artifact()  -> workdir/inputs/
    |                              |- checkpoint_latest()  -> workdir/inputs/resume.json
    |                              |
    |                              |- ArgvDockerRunner.run(payload, workdir, inputs)
    |                              |     docker run --network none --read-only
    |                              |       -v workdir:/work  <image>  <argv...>
    |                              |     the job sees ONLY /work
    |                              |
    |  <-- attempt_heartbeat ------|  (throughout)
    |  <-- checkpoint relay -------|  (parts shipped as they appear)
    |  <-- upload_artifact() ------|  outdir/* -> artifact store
    |  <-- complete(sha256) -------|  commit-time hash validation
```

### 1. `flashnode/executor/hardening.py` (new)

Single source of truth for the container security posture. Both docker runners
call it, so they cannot drift apart — the failure mode where a new runner
quietly misses a flag the old one had.

```python
def harden_args(workdir: Path, *, cpus: float, memory_gb: float,
                pids_limit: int = 512) -> list[str]:
    """Docker flags common to every sandboxed task. Security-relevant:
    changing this function changes the guarantee for ALL runners."""
```

Emitted flags:

| Flag | Why |
|---|---|
| `--network none` | job never reaches the volunteer's LAN or the internet |
| `--read-only` | immutable rootfs |
| `--tmpfs /tmp:rw,noexec,nosuid,size=256m` | was unbounded; `noexec` stops drop-and-run |
| `--user <uid>:<gid>` | never root in the container's world |
| `--cap-drop=ALL` | no `CAP_NET_RAW`, `CAP_SYS_ADMIN`, … |
| `--security-opt=no-new-privileges` | setuid binaries cannot escalate |
| `--pids-limit=512` | fork-bomb containment |
| `--memory <N>g` + `--memory-swap <same>` | equal values; otherwise `--memory` is bypassable via swap |
| `--cpus <N>` | CPU share cap |
| `--ulimit nofile=1024:1024` | fd exhaustion |
| `-v <workdir>:/work` + `-w /work` | the one writable path |

`docker_runner.py` is refactored to call `harden_args()`. That refactor must be
**behaviour-identical for the flags it already had** and purely additive for the
new ones; `tests/test_docker_runner.py` pins the existing argv construction.

### 2. `flashnode/executor/argv_runner.py` (new)

```python
class ArgvDockerRunner:
    def __init__(self, allowed_images: frozenset[str], cpus: float = 2.0,
                 memory_gb: float = 2.0, timeout_seconds: float = 3600.0,
                 max_output_bytes: int = 2 * 1024**3): ...
    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path: ...
```

Order of operations (fail-closed checks happen **before** any subprocess):

1. `argv = payload["argv"]` must be a non-empty `list[str]` — else
   `TaskExecutionError`.
2. `image = payload["image"]` must be present **and in `allowed_images`** — else
   `TaskExecutionError`. This is the volunteer's consent mechanism.
3. `env` keys must match `^[A-Za-z_][A-Za-z0-9_]*$`; values must be `str`.
4. Build `docker run --rm` + `harden_args(...)` + `--env K=V`… + `image` +
   `argv`.
5. `subprocess.run(argv_list, capture_output=True, timeout=…)` — never
   `shell=True`.
6. On non-zero exit: `TaskExecutionError` with the last 800 bytes of stderr.
7. Enforce `max_output_bytes` over `outdir` before returning (disk-fill
   mitigation, see Known gaps).
8. **`metrics.json` is mandatory**, exactly as for the module runner: absent →
   `TaskExecutionError`. This is forced by the commit protocol, not a runner
   preference — `CommandRecipe` sets `commit_key` to
   `jobs/{job_id}/{task_id}/metrics.json`, and `_output_valid()`
   (`service/modea.py:244`) reads the artifact at `commit_key` and compares its
   sha256 against the claimed value. No file at that key ⇒ the commit is
   rejected. Failing early in the runner turns a confusing commit rejection
   into a clear task error. Collecting *additional* files via
   `OutputSpec.collect` works as today; only `metrics.json` is load-bearing.

**No `--spec`/`--out` injection.** The module contract passes
`--spec spec.json --out out/`; argv workloads receive the user's argv verbatim.
Inputs are discoverable at `/work/inputs/`, outputs are written to `/work/out/`,
communicated by convention and documented — the runner does not rewrite argv.

### 3. Security contract: argv is container-only

`IsolationSpec.tier` defaults to `"standard"`, and `IsolationAwarePlacement`
lets standard tasks run anywhere. So an argv runner added with no other change
would let a command job default to standard, land on any node, and execute a
stranger's argv unsandboxed. The governing rule:

> **An argv payload is only ever executed inside a container. There is no
> unsandboxed argv path — not by default, not by flag, not by fallback.**

Enforced in three independent places; any one alone would suffice.

1. **`CommandRecipe.validate_params()`** requires `tier == "sandboxed"` and
   rejects `"standard"`. The only escape is a **coordinator-side** env opt-in,
   `FLASHML_ALLOW_UNSANDBOXED_ARGV=1`, for an operator running their own
   trusted fleet. Deliberately server-side: a *submitter* must never be able to
   downgrade the isolation their code runs under.

   **`allowFallback` is rejected for argv payloads.** The existing placement
   gate treats `allowFallback is True` as a waiver of the capability
   requirement (`scheduler/__init__.py:124`) — left alone, a submitter could
   set it and land arbitrary argv on a non-sandboxed node, bypassing the whole
   contract. `validate_params()` therefore rejects a command job that sets
   `allowFallback: true`, and the argv placement gate is checked *before* the
   waiver can apply.
2. **`SubprocessRunner.run()`** raises `TaskExecutionError` on sight of an
   `argv` key. This protects the volunteer who runs `flashnode work` with
   default flags; Tier 1 must never gain argv capability by accident.
3. **`ArgvDockerRunner`** is container-only by construction — no non-docker
   branch exists to fall into.

**Injection surfaces and why they are closed:**

- `payload["image"]` is matched against `allowed_images` *before* any subprocess
  call, so a hostile value such as `--privileged` never reaches docker's flag
  parser.
- `payload["argv"]` is placed *after* the image reference, where docker treats
  it as the container command; leading-`-` tokens are inert there.
- `payload["env"]` keys are regex-validated before becoming `--env K=V`, so a
  key cannot smuggle a docker flag.

**Image allowlist vs. code allowlist.** `FLASHNODE_ALLOWED_IMAGES` stays. It is
the *operator's consent* mechanism (what a volunteer agrees to run), not a code
allowlist — the user's code inside a permitted image is unrestricted, gated by
isolation tier. This is the distinction that makes the slice consistent with the
project's isolation stance.

### 4. Protocol: one new field

```python
class NodeRegistration(BaseModel):
    ...
    capabilities: NodeCapabilities
    environment: NodeEnvironment = NodeEnvironment.LOCAL
    sandbox_capable: bool = False
    argv_capable: bool = False     # NEW - fail closed
```

The field goes on **`NodeRegistration`**, beside the existing
`sandbox_capable` — *not* on `NodeCapabilities`, which carries only hardware
facts (cpu/memory/gpus/os/arch). The claim handler builds its `NodeView` from
the registration (`service/modea.py:295`), so the new field must be added to
that dict alongside `sandbox_capable` or the placement gate will never see it.

A node running the *module* `DockerRunner` is genuinely sandbox-capable but
cannot execute argv; overloading `sandbox_capable` would let the coordinator
place a command task on a node certain to fail it. Defaulting to `False` means
every already-deployed flashnode is excluded from argv work until upgraded and
explicitly opted in (AGENTS.md rule 3: security-relevant fields fail closed).

`IsolationAwarePlacement.eligible()` gains a parallel gate, mirroring the
existing `is True` idiom (which refuses truthy stand-ins like `"false"` or `1`):

```python
if "argv" in task.payload and node.get("argv_capable") is not True:
    return False        # checked BEFORE the allowFallback waiver below
```

Ordering is load-bearing: this gate runs ahead of the `allowFallback` waiver so
the waiver can never grant argv placement on an incapable node.

**Tier vocabulary stays `standard | sandboxed`.** `gvisor`/`microvm` literals
are deliberately not added yet: the placement gate already treats any unknown
tier as requiring capability rather than silently downgrading, so adding them
later is non-breaking. Adding unimplemented literals now would be scaffolding
(AGENTS.md rule 7).

### 5. Agent CLI

`flashnode work --runner argv` selects `ArgvDockerRunner` and advertises
`argv_capable: true` at registration. `--runner subprocess` (default) and
`--runner docker` are unchanged and advertise `argv_capable: false`.

New env knobs, all with safe defaults: `FLASHNODE_ALLOWED_IMAGES` (existing),
`FLASHNODE_MAX_CPUS`, `FLASHNODE_MAX_MEMORY_GB`, `FLASHNODE_TASK_TIMEOUT_S`,
`FLASHNODE_MAX_OUTPUT_BYTES`. These are the volunteer's dial for how much of
their machine the pool may use.

### 6. Slice E: composite `(job_id, task_id)` key

`CommandRecipe.expand()` names tasks positionally (`f"task-{i:03d}"`) with no
job scope, and the lease stores key by `task_id` alone
(`InMemoryLeaseStore._tasks`, `sqlite_store` `task_id TEXT PRIMARY KEY`). Two
concurrent jobs both produce `task-000`; the second collides with a 500
(`task task-000 already exists`). A volunteer pool is multi-job by definition,
so this breaks the moment two people use it.

- `TaskRecord.spec` already carries both `job_id` and `task_id`, so `add()` and
  `save()` derive the composite key internally — **no signature change** on the
  two hot paths.
- `get(task_id)` becomes `get(job_id, task_id)`. Exactly one call site
  (`manager.py:281`), plus the `LeaseStore` Protocol and both implementations.
- `InMemoryLeaseStore._tasks` becomes `dict[tuple[str, str], TaskRecord]`;
  insertion order is preserved so `next_pending()` stays deterministic, as its
  docstring promises.
- SQLite: `PRIMARY KEY (job_id, task_id)`; the upsert's `ON CONFLICT(task_id)`
  becomes `ON CONFLICT(job_id, task_id)`.

**Migration.** SQLite cannot alter a primary key in place. `_migrate()` inspects
`PRAGMA table_info(lease_tasks)` and, on finding the old single-column PK,
creates `lease_tasks_v2` with the composite key, copies rows, and swaps.
Rebuilding rather than dropping is the entire point of the durable store:
**in-flight leases must survive the upgrade**, the property
`tests/test_leases_sqlite.py` already pins.

## Error handling

| Condition | Behaviour |
|---|---|
| `argv` missing / not `list[str]` / empty | `TaskExecutionError` before any subprocess → `fail()` → requeue |
| `image` absent or not in `allowed_images` | `TaskExecutionError`, refused before subprocess |
| `env` key fails regex | `TaskExecutionError`, refused before subprocess |
| argv payload reaches `SubprocessRunner` | `TaskExecutionError` — the tier-confusion guard |
| container exits non-zero | `TaskExecutionError` with last 800 B of stderr → attempt fails → retried per `maxTaskAttempts` |
| wall-clock timeout | container killed, `TaskExecutionError` → requeue |
| no `metrics.json` in `/work/out` | `TaskExecutionError` naming the missing file — fails early instead of surfacing as an opaque commit rejection |
| outputs exceed `max_output_bytes` | `TaskExecutionError`, nothing uploaded |
| volunteer disappears mid-task | unchanged: heartbeat stops → lease expires → requeue elsewhere |
| lease lost while running | unchanged: `hb.lost` discards the result, so a slow volunteer cannot commit stale work over a reassigned task |
| `docker` binary absent | agent refuses to start with `--runner argv` and says why (never silently degrades to Tier 1) |

Failure semantics deliberately do **not** change: everything routes through the
existing `TaskExecutionError` → `fail()` → lease-expiry → requeue path that the
e2e kill-sweep already proves.

## Testing

**Unit (flashnode), no docker required** — assert on constructed argv:

- `harden_args()` contains every flag in the table above; a test asserts
  `--memory-swap` equals `--memory` (the bypass this prevents is invisible
  otherwise).
- `ArgvDockerRunner` refuses: missing/empty/non-list argv; absent image;
  non-allowlisted image; bad env key. Each refusal asserted to happen **without
  invoking subprocess** (patched `subprocess.run` asserted not called) — a
  check that runs after launching is not a check.
- Injection: `image="--privileged"` is refused; an argv token `--privileged`
  appears only *after* the image position.
- `SubprocessRunner` raises on an `argv` payload.
- `docker_runner` argv construction is unchanged for pre-existing flags
  (regression pin on the refactor).

**Unit (flashruntime):**

- `IsolationAwarePlacement`: argv payload + `argv_capable` absent/`False`/
  `"true"`/`1` → ineligible; only genuine `True` → eligible.
- `CommandRecipe.validate_params()` rejects `tier: "standard"`, rejects
  `allowFallback: true`, and accepts `"standard"` only when
  `FLASHML_ALLOW_UNSANDBOXED_ARGV=1` is set on the coordinator.
- The waiver cannot bypass the argv gate: a task with `allowFallback: true` and
  an `argv` payload is ineligible on a node without `argv_capable` — the
  ordering test for the gate.
- Lease stores: two jobs each adding `task-000` coexist; `get()` returns the
  right record per job; `next_pending()` order unchanged.
- SQLite migration: build a DB with the **old** schema, open it with the new
  store, assert rows survive with the composite PK **and an in-flight lease is
  still leased** after the migration.

**Integration (opt-in, real docker, `-m integration`):**

- End-to-end argv task: a trivial image runs `sh -c 'echo … > /work/out/m.json'`
  and commits.
- `--network none` is real: a task attempting egress fails.
- `--read-only` is real: writing outside `/work` fails.
- Volunteer-kill: SIGKILL the agent mid-argv-task → lease expires → a second
  agent completes it. This is the existing e2e proof re-run over the argv path.

**Suite gate:** current baseline is 297 passed / 1 skipped / 20 deselected in
flashruntime. The slice must land green with the new tests added, and
`scripts/audit_secrets.sh` must stay CLEAN.

## Known gaps (documented, not solved here)

- **Disk fill.** `--read-only` protects the rootfs, but the bind-mounted workdir
  is writable and Docker cannot easily quota a bind mount. Mitigated by
  `max_output_bytes` at upload time; a per-task loopback volume is the real fix.
- **Container escape.** Hardened Docker still shares the host kernel. This is
  why the tier vocabulary stays open for gVisor/microVM.
- **Lying nodes.** No result verification until slice C.
- **Shared join code.** One static secret for all volunteers until slice B.
- **No network in-container** means jobs cannot `pip install` or pull from
  HuggingFace; everything must be baked into the pinned image or pre-staged as
  an `artifact://` input. This is a real constraint on job authorship and must
  be stated loudly in user-facing docs with a clear error, not discovered at
  runtime.

## Phasing

1. **E first** — composite lease key + migration. Independent, small, and
   unblocks multi-job pools regardless of what happens to A.
2. **`harden_args()`** + `docker_runner` refactor, with the regression pin.
3. **`ArgvDockerRunner`** + `SubprocessRunner` argv refusal.
4. **Protocol + placement gate** (`argv_capable`) + `CommandRecipe` tier
   validation.
5. **Agent CLI** wiring (`--runner argv`, capability advertisement, env knobs).
6. **Integration tests** + the volunteer-kill e2e over the argv path.
7. **Docs** — a "donate a machine" guide covering the volunteer's consent knobs,
   and a job-authoring note on the no-network constraint.
