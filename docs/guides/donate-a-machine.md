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
| `FLASHNODE_JOIN_CODE` | *(none)* | The shared secret this coordinator's operator gave you to register at all. See [One shared join code](#one-shared-join-code) below. |

---

## Isolation, precisely

`--runner argv` selects `ArgvDockerRunner`. Every task it runs gets the exact
same `docker run` flags — there is no per-task relaxation:

| Flag | What it stops |
|---|---|
| `--network none` | The container cannot reach your LAN or the internet. |
| `--read-only` | The container's root filesystem cannot be modified. |
| `--tmpfs /tmp:rw,noexec,nosuid,size=256m` | `/tmp` is writable but capped and non-executable — no drop-and-run. |
| `--user <uid>:<gid>` | The process runs as a non-root, unprivileged user inside the container. |
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

**Only sandboxed tasks land here at all.** A job submitted for volunteer
placement must declare `isolation.tier: "sandboxed"`; the coordinator rejects
a command job that tries to set `allowFallback: true` to waive that
requirement. A submitting user can never downgrade the isolation their own
code runs under — that decision belongs to the node operator (you) and,
separately, to whoever runs the coordinator. (A coordinator operator running
a fully trusted fleet can opt out of the tier requirement server-side with
`FLASHML_ALLOW_UNSANDBOXED_ARGV=1`; a job submitter has no equivalent
override.) Correspondingly, `--runner argv` is the only runner that
advertises sandbox+argv capability to the coordinator: `--runner subprocess`
(the default) and `--runner docker` never accept argv-shaped payloads, so a
volunteer who starts the agent without `--runner argv` is never handed
arbitrary code in the first place.

---

## Known limitations

These are not edge cases buried in fine print — they are the current state of
the system, and you should weigh them before joining.

1. **No result verification.** A volunteer node that returns fabricated
   `metrics.json` output is currently **believed**. Nothing today re-runs a
   sample of tasks elsewhere and compares results. Spot-check verification
   and a reputation system are designed but not built.
2. **One shared join code.** All volunteers on a coordinator authenticate
   with the same `FLASHNODE_JOIN_CODE`. There is no per-node identity and no
   way to revoke a single misbehaving node without rotating the code for
   everyone.
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
