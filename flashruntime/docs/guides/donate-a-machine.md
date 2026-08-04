# Donate a machine

FlashML's compute pool works because strangers let other strangers' code run
on their hardware. This page is the honest version of what that means: what
`flashnode work --runner argv` actually confines, what it does not, and the
knobs you control before you type the command.

**Read this before you run anything.** If the trust model below is not
acceptable for your machine, do not join — declining is a completely
reasonable outcome of reading this page.

---

## What joining does

`flashnode work --runner argv` registers your machine with a FlashRuntime
coordinator, then polls for tasks. Each task it accepts is **someone else's
command, running as an arbitrary Docker image and argv you did not choose and
cannot inspect in advance** (only the image reference and the allowlist you
set — never the code inside the image). The agent:

1. downloads the task's declared `artifact://` inputs into a private workdir,
2. runs the image + argv inside a hardened container bound to that workdir,
3. uploads whatever the container wrote to `/work/out/`, and
4. reports success or failure to the coordinator.

Your machine never talks to the job's author directly, and the job never
talks to anything but its own workdir — see [Isolation, precisely](#isolation-precisely).

### The honest trust statement

**Your machine runs other people's code.** It is confined to a hardened
container with no network, a read-only root filesystem, no Linux
capabilities, and CPU/memory/PID caps. That confinement is real and it is
enforced on every task. It is also **not a virtual machine boundary** —
hardened Docker still shares your host kernel with the container, so a
container escape is not impossible, only made harder. If you would not run
an unknown binary as a limited user on this machine, do not run it as a
volunteer node either.

---

## Platform support

| Platform | Status | Prerequisite | `FLASHNODE_WORKDIR` |
|---|---|---|---|
| Linux | Proven | Docker or Podman | Any local path; defaults to the system temp dir. |
| macOS | Proven | Docker Desktop, or Colima | Must be under `$HOME` — the VM only shares `$HOME` by default (colima), so a workdir outside it silently bind-mounts as an **empty** directory rather than failing loudly. |
| Windows | **Constructed-argv-verified, not execution-verified** (see below) | Docker Desktop with the **WSL2 backend** | Must be under a directory Docker Desktop shares (by default your user profile, e.g. `C:\Users\<you>\...`) — a workdir outside Docker Desktop's shared drives has the same silent-empty-mount failure mode as the macOS/colima case above, and has already cost real debugging time once. |

`flashnode work` used to crash immediately on Windows: `os.getuid`/
`os.getgid`, which the agent needs to build the container's `--user` flag,
do not exist there. As of this change:

- On Windows, `--user` is **omitted** rather than crashing. This is safe
  only because the curated images this pool runs (`python-slim`, `sklearn`,
  `pytorch-cpu`) each declare a fixed non-root `USER` in their Dockerfile —
  see `flashml-cloud/images/README.md`. If you point `FLASHNODE_ALLOWED_IMAGES`
  at some other image on a Windows host, and that image runs as root by
  default, your container runs the task **as root inside the container**.
  Use only images you've confirmed declare a non-root `USER`.
- Your workdir's bind-mount source (`-v <workdir>:/work`) is rewritten from
  Windows's `C:\Users\...` form into the form Docker Desktop's Windows CLI
  accepts. You should not need to do anything for this — it's automatic.

**What "constructed-argv-verified, not execution-verified" means:** every
test covering the Windows code path runs on macOS/Linux with the platform
faked (`sys.platform` monkeypatched, a synthetic `PureWindowsPath` in
place of a real one). That proves the `docker run` argv flashnode
*constructs* is correct — it does not prove Docker Desktop on a real
Windows machine accepts that argv, or that a task actually completes
there. Windows becomes **execution-verified** only once someone runs
`flashnode work` on an actual Windows machine and completes a real task —
that acceptance run belongs to a later deploy plan, not this change.

---

## Quickstart

```bash
export FLASHNODE_ALLOWED_IMAGES=ghcr.io/zolli/trainer:1.0
export FLASHNODE_MAX_CPUS=4 FLASHNODE_MAX_MEMORY_GB=8
flashnode work --runner argv --coordinator https://<coordinator>
```

`FLASHNODE_ALLOWED_IMAGES` is mandatory. `--runner argv` **refuses to start**
against an empty allowlist — there is no "run whatever image the task sends"
mode and no silent downgrade to an unsandboxed tier. You are naming, in
advance, every image you consent to run.

**The allowlist is enforced locally, by your agent — not by the
coordinator.** The coordinator does not receive your `FLASHNODE_ALLOWED_IMAGES`
and does not filter placement by image reference; it will claim you a task
whose pinned image you never allowlisted just as readily as one you did.
Your agent is what refuses to run it: `ArgvDockerRunner` checks the task's
image against your local allowlist before invoking `docker run` at all, and
the task simply fails (and gets requeued to another node) if it isn't
listed. There is no silent downgrade either way — you just find out at
your machine, not at the coordinator.

---

## Authenticating your machine

If the coordinator you're joining enforces per-machine authentication (it
does this by setting `FLASHML_NODE_TOKENS` server-side — see below), you
need a token before `flashnode work` can register, claim leases, or commit
results. The operator hands you a token out of band (chat, a secrets
manager, whatever channel they already use); you save it once with:

```bash
flashnode login --coordinator https://<coordinator> --token <the-token-they-gave-you>
```

This writes the token to a per-coordinator credential store at
`~/.flashnode/credentials.json` (mode `0600`, one entry per coordinator URL
so a single machine can join several pools without one login clobbering
another). Every subsequent `flashnode work --coordinator https://<coordinator>`
reads it automatically and sends it as a bearer token on every request — you
do not pass `--token` to `work` itself. To stop using a coordinator, run
`flashnode logout --coordinator https://<coordinator>`, which deletes the
saved entry (it does not revoke the token on the server — see below).

**What the token buys you, and what it doesn't:**

- The coordinator resolves your `node_id` **from the token**, never from
  anything your agent sends in a request body. A stolen bearer token
  impersonates your node; treat it like a password.
- Once authenticated, the coordinator only lets you write under
  `jobs/{job}/{task}/` for tasks you currently hold a **live lease** on —
  the [known limitation below](#known-limitations) about one node
  overwriting another's results is closed when enforcement is on. Reads are
  unauthenticated regardless (by design — see the note at the end of this
  section).
- **Revocation is real but manual and server-side.** The coordinator's
  operator removes your `node_id:token` pair from `FLASHML_NODE_TOKENS` and
  restarts the coordinator; your next request gets `401` immediately.
  There is no self-service token rotation and no expiry — a token is valid
  until the operator edits their configuration. `flashnode logout` only
  forgets the token locally; it has no effect on the server.
- **Tokens are configured statically on the coordinator today.** The
  operator sets `FLASHML_NODE_TOKENS=node-a:tok-a,node-b:tok-b,...` before
  starting the process; there is no self-service enrolment, no signup form,
  and no browser device-code flow. `flashnode login` just saves a token you
  were already given — it does not obtain one. Self-service issuance is
  planned for the cloud API, not this milestone.
- **If the coordinator does not set `FLASHML_NODE_TOKENS` at all**, it
  behaves exactly as before this feature existed: no authentication, no
  write scoping, `flashnode login` is unnecessary. This is still the
  default for a single trusted machine or a fully local dev loop. An
  operator who wants to *require* tokens (refuse to start without any
  configured) sets `FLASHML_REQUIRE_NODE_AUTH=1`.
- **Read access is unauthenticated either way.** Anyone who can reach the
  coordinator can `GET` an artifact today, token or no token. Read-side
  authorization is the cloud API's job in a later milestone — see the
  known limitations below.

---

## Your consent knobs

Every knob below has a safe default. Set the ones that matter for your
machine before you run `flashnode work`.

| Variable | Default | What it controls |
|---|---|---|
| `FLASHNODE_ALLOWED_IMAGES` | *(none — required)* | Comma-separated pinned image references this node will run. No default; the agent refuses to start without at least one. |
| `FLASHNODE_MAX_CPUS` | `2.0` | `--cpus` cap passed to every container. |
| `FLASHNODE_MAX_MEMORY_GB` | `2.0` | `--memory` cap, with `--memory-swap` set to the **same** value (see below — otherwise the cap is bypassable). |
| `FLASHNODE_TASK_TIMEOUT_S` | `3600` | Wall-clock seconds before the agent kills the container. |
| `FLASHNODE_MAX_OUTPUT_BYTES` | `2147483648` (2 GiB) | Enforced against the task's output directory before upload; over the cap, nothing is uploaded and the task fails. |
| `FLASHNODE_JOIN_CODE` | *(none)* | The shared secret this coordinator's operator gave you to register at all, if the coordinator still gates registration that way. |

---

## Isolation, precisely

`--runner argv` selects `ArgvDockerRunner`. Every task it runs gets the exact
same `docker run` flags — there is no per-task relaxation:

| Flag | What it stops |
|---|---|
| `--network none` | The container cannot reach your LAN or the internet. |
| `--read-only` | The container's root filesystem cannot be modified. |
| `--tmpfs /tmp:rw,noexec,nosuid,size=256m` | `/tmp` is writable but capped and non-executable — no drop-and-run. |
| `--user <uid>:<gid>` | The process runs as a non-root, unprivileged user inside the container. **Windows only:** this flag is omitted (`os.getuid`/`os.getgid` don't exist there); non-root execution instead relies entirely on the curated image's own `USER` — see [Platform support](#platform-support). |
| `--cap-drop=ALL` | No Linux capabilities at all — no raw sockets, no admin operations. |
| `--security-opt=no-new-privileges` | setuid binaries inside the image cannot escalate. |
| `--pids-limit=512` | Caps process/thread count — contains fork bombs. |
| `--cpus <N>` / `--memory <N>g` + `--memory-swap <N>g` | CPU and memory caps; swap is pinned equal to memory so the memory cap can't be bypassed by swapping. |
| `--ulimit nofile=1024:1024` | Caps open file descriptors. |
| `-v <workdir>:/work -w /work` | The **only** writable path is your task's private workdir, bound at `/work`. Nothing else on your disk is reachable. |

If the task overruns `FLASHNODE_TASK_TIMEOUT_S`, the agent issues `docker
kill` **against the running container by name** — the cap is enforced on the
container itself, not merely by the local `docker` client giving up and
walking away.

**Only sandboxed tasks land on an `--runner argv` machine.** A job submitted
for open-pool placement (no `placement.pool` set) must declare
`isolation.tier: "sandboxed"`; the coordinator rejects a command job that
tries to set `allowFallback: true` with no pool attached. Outside a pool, a
submitting user can never downgrade the isolation their own code runs
under — that decision belongs to the node operator (you) and, separately, to
whoever runs the coordinator. (A coordinator operator running a fully
trusted fleet can opt out of the tier requirement server-side with
`FLASHML_ALLOW_UNSANDBOXED_ARGV=1`; a job submitter has no equivalent
override outside a pool.) Correspondingly, `--runner argv` is the only
runner that advertises sandbox+argv capability to the coordinator:
`--runner subprocess` (the default) and `--runner docker` never accept
argv-shaped payloads, so a volunteer who starts the agent without
`--runner argv` is never handed arbitrary code in the first place.

### Team pools: `--runner trusted`

There is one coupled exception to the paragraph above, and it exists only
inside a pool. A job scoped to a pool (`placement.pool` set) *can* set
`allowFallback: true`, and a node in that same pool *can* opt in to run it
unsandboxed — no container, no `--network none`, none of the flags in
[Isolation, precisely](#isolation-precisely) below. This is `flashnode work
--runner trusted`, and it is a different trust decision from everything
above: you are not trusting an anonymous submitter behind a sandbox, you are
trusting the specific people your pool operator invited, running their code
directly on your machine.

Say it plainly: **unsandboxed means unsandboxed.** A pool job under
`--runner trusted` can read and write anything your user account can, reach
your network, and outlive its declared limits the way a sandboxed job
cannot. Only run this for a pool you would hand a shell account to. The
coordinator confines what reaches you either way — it refuses every
non-pool task to a trusted-tier worker outright (fail closed, the same
placement gate as everything else in this doc), so a machine running
`--runner trusted` never receives a stranger's job, only its own pool's. Run
`flashnode work --runner trusted --coordinator https://<coordinator>` to opt
in; there is no environment variable equivalent, and the command prints its
own unsandboxed warning every time it starts.

This tier is a reasonable default for hosts that structurally cannot run
`--runner argv` at all — a Colab notebook or a rented pod, neither of which
can nest a Docker daemon of their own to sandbox into. If your pool operator
sent you a link to a hosted console, its own docs page and guides cover the
exact commands for those hosts; nothing further to say about it here beyond
what a pool operator's invite already explains.

---

## Known limitations

These are not edge cases buried in fine print — they are the current state of
the system, and you should weigh them before joining.

1. **No result verification.** A volunteer node that returns fabricated
   `metrics.json` output is currently **believed**. Per-machine tokens (see
   [Authenticating your machine](#authenticating-your-machine)) confine
   *where* a node may write — they say nothing about whether what it writes
   is true. Nothing today re-runs a sample of tasks elsewhere and compares
   results.

   This paragraph used to end "Spot-check verification and a reputation
   system are designed but not built." That was wrong and is corrected here:
   **no such design exists** in either repo. (`flashnode/benchmark/` is
   sometimes cited as the design; it is not — it is admission *capability*
   probing, `cpu_hash_mbps` / `mem_bandwidth_mbps` / `disk_write_mbps`, which
   measures how fast a host is, never whether its answers are correct.)

   What DOES exist, as of 2026-08-03, is narrower and applies to **federated
   jobs only**: each contribution's weight delta is capped in magnitude
   before averaging, at 3× the median norm of that round. That stops one
   participant steering the shared model arbitrarily. It does **not** verify
   anything:

   - it bounds *magnitude*, not *direction* — a small, consistently biased
     delta every round passes untouched;
   - it does not detect a lazy node, because returning zeros or replaying
     last round's delta has a perfectly ordinary norm and still earns credit;
   - with only two participants it is weak by construction: the median of two
     values is their mean, which an attacker influences directly. Robust
     statistics need a majority, and two contributors have no majority;
   - a colluding majority defeats it entirely;
   - sweeps and `command` jobs get nothing from it at all.
2. **Static, out-of-band token issuance.** Per-machine tokens replaced the
   old shared join code — each volunteer now authenticates with its own
   `flashnode login`-saved bearer token, confined to the leases it holds,
   and revocable by removing that one node's entry from the coordinator's
   `FLASHML_NODE_TOKENS` (see [Authenticating your machine](#authenticating-your-machine)).
   What's still missing: there is no self-service enrolment — an operator
   configures tokens statically before you can join — and no browser
   device-code flow; that arrives with the cloud API. `POST
   /v1alpha1/jobs` is also still unauthenticated regardless of node tokens,
   so anyone who can reach the coordinator can submit a job today; job
   ownership is the cloud API's job in a later milestone.
3. **Container escape is not ruled out.** The flags above are real and
   meaningfully raise the bar, but hardened Docker still shares your host
   kernel with the job. This is not the isolation guarantee of a virtual
   machine. Stronger tiers (gVisor, microVM) are a possible future addition;
   they do not exist yet.
4. **Disk fill is capped only at upload time.** `--read-only` protects the
   rest of your filesystem, but the bind-mounted workdir itself is writable
   for the duration of the task, and nothing enforces a quota on it *while
   the task runs* — `FLASHNODE_MAX_OUTPUT_BYTES` is checked only after the
   container exits, before upload. A task can fill your disk during the run
   even though the result is later rejected.
5. **No GPU work.** The agent does not probe for or advertise GPUs today.
   Volunteer nodes run CPU tasks only.
6. **No network inside the container, at all.** A job cannot `pip install`,
   clone a repo, or download from HuggingFace once it starts. Everything the
   job needs must already be baked into the pinned image, or staged ahead of
   time as an `artifact://` input — the agent downloads those into
   `/work/inputs/` before the container starts. The job writes its results to
   `/work/out/`, and **`metrics.json` is required there**: the coordinator
   validates the artifact at the task's commit key by sha256, so a task that
   doesn't produce it cannot commit, full stop. See the job-author's side of
   this constraint in [bring-your-code.md](bring-your-code.md#volunteer-nodes-run-with-no-network).
7. **No coordinated multi-process training.** `mode: "coordinated"`
   (torchrun/DDP-style rendezvous) is not available on volunteer nodes — with
   `--network none`, ranks have no way to find each other even if the
   coordinator tried to place them together. Volunteer pools run
   **independent tasks only** (sweeps, sharded work): losing one volunteer
   costs one task, never a whole distributed run.
8. **The sandbox guarantees are asserted, not yet kernel-verified.**
   Integration tests exist that are meant to prove `--network none` actually
   blocks egress and `--read-only` actually blocks writes outside `/work` —
   but they are currently **skipped**, because no Docker daemon was available
   in the environment that built this feature. What has been verified so far
   is that the runner *constructs* the correct `docker run` flags for every
   case (missing/empty argv, non-allowlisted image, bad env keys, the
   `--memory`/`--memory-swap` pairing, and more) — not that a real Docker
   daemon enforces them as expected. Run the integration suite yourself
   (`pytest -m integration`, real Docker required) before trusting this on
   hardware you care about.
9. **Windows support is constructed-argv-verified, not execution-verified.**
   See [Platform support](#platform-support) above. The `--user` omission
   and the bind-mount path rewrite are covered by tests that fake
   `sys.platform`/`os.getuid` and use a synthetic `PureWindowsPath` — none
   of it has run against a real Docker Desktop instance on real Windows
   yet. Non-root execution on Windows also depends on the curated images'
   `USER` declaration staying correct (see item above on `--user` in
   [Isolation, precisely](#isolation-precisely)) — that dependency doesn't
   exist on Linux/macOS, where `--user` is passed explicitly regardless of
   what the image declares.

---

## What happens when your machine goes away

If your agent stops heartbeating mid-task — network drop, laptop closed,
`Ctrl-C`, `SIGKILL` — the coordinator's lease on that task simply expires and
another node picks it up. This is the same lease-expiry recovery path proven
end-to-end for the built-in task modules; nothing about it changes for argv
tasks. You can stop volunteering at any time without coordinating with
anyone; the worst case is that whatever task you were running gets retried
elsewhere.

---

## See also

- [bring-your-code.md](bring-your-code.md) — the job author's side: what a
  command workload can and cannot assume about the machine it lands on.
- [jobspec-and-isolation.md](../site/guides/jobspec-and-isolation.md) — the
  `isolation.tier` contract and the placement gate that keeps sandboxed work
  off unsandboxed nodes.
