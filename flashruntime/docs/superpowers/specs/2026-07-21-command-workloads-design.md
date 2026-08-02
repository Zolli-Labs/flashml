# Design: Framework-neutral command workloads ("bring your own code")

Date: 2026-07-21 · Status: approved (design) · Author: pairing session

## 1. Problem & goal

Today FlashRuntime has two execution paths, and neither lets a user run an
existing training repo unmodified:

- The prototype engine (`Cluster.train`) **requires** a `DistributedAlgorithm`
  subclass (`engine/loop.py` calls `plan→initialize→make_tasks→reduce→
  converged→finalize`).
- The Mode A service runs only **allowlisted** task modules
  (`service/modea.py:ALLOWED_TASK_MODULES`) via a hand-coded expansion.

**Goal:** let a user bring their own sklearn / PyTorch / Hugging Face code and
have FlashRuntime *operate* the job — provision, launch the right command, set
distributed env vars, track workers, collect events/metrics/artifacts, manage
checkpoints, and (Mode A) retry/recover — **without FlashRuntime understanding
or rewriting the user's model**. FlashRuntime is the operator; the framework
does the distributed training; the user owns their code.

This is an **additive extension** of the existing four-axes architecture
(ADR-0003) and implements the already-designed `Launcher` / `StrategyCompiler`
/ `WorkloadRecipe` interfaces for the first time. It is not a rewrite. The
existing `Cluster.train()` + `algorithms/` paths keep working unchanged and
are reframed in docs as batteries-included examples.

## 2. Non-goals

- No auto-inspection / auto-parallelization of user code (ADR-0003:
  auto-parallelization is research-grade). Strategy is declared or handled by
  the framework, never inferred from source.
- No re-implementation of any estimator/model/training loop.
- `flashruntime.torch` (§4.6) must NOT grow into an Accelerate clone: no
  FSDP policies, no autocast/mixed-precision management, no DeepSpeed
  configs. Its complete surface is prepare/checkpoint/log_metrics/rank
  helpers.
- The **sandbox/VM mechanism** (firecracker/gVisor/docker `--network none`,
  resource caps) is **flashnode's executor tier**, out of scope here. This
  repo owns only the *contract* (isolation requirement + capability match).
- flashnode's executor accepting an `argv` payload is a **cross-repo,
  versioned change**; we define the additive payload contract here and flag
  that real service-side execution of arbitrary commands waits on it.
- Concrete remote providers (RunPod/ECS) and actual multi-GPU runs are later
  slices. (Multi-process DDP is proven here on CPU via `gloo`.)

## 3. Security & isolation model

The module allowlist was a stopgap; the durable control is **isolation tier,
not a code allowlist** (see memory `trust-and-isolation-model`).

- Reuse existing schema: `IsolationSpec{tier: standard|sandboxed, allowFallback}`
  (`protocol/v1alpha1.py`), `NodeRegistration.sandbox_capable`,
  `PlacementSpec.pool` (incl. `secure-cloud`).
- **Trusted nodes** (user's own machine, RunPod with the user's creds) may run
  `standard` — command runs directly.
- **Community / unauthorized nodes** must run `sandboxed` — the executor runs
  each task inside a sandbox/VM.
- **Fail-closed placement (AGENTS.md rule 3):** a task requiring `sandboxed`
  is leased **only** to a `sandbox_capable` node. If none is available and
  `allowFallback=False`, the task stays PENDING and the job surfaces the
  reason — it never silently runs unsandboxed.
- Transitional: the existing module-allowlist path stays working for built-in
  workloads. The SDK-local path is trusted-by-definition (the user's own
  process on their own machine) and needs no allowlist.

## 4. Architecture

Every new component implements an interface that already exists in the repo
(rule 7 — implement against the designed contracts, do not invent new ones).

```
  USER-FACING (core, pydantic-only)        COORDINATOR (service)         EXECUTION (extras)
  ┌───────────────────────────┐   submit   ┌──────────────────────┐      ┌─────────────────────┐
  │ workloads/CommandWorkload │──────────▶ │ recipes/CommandRecipe │────▶ │ launchers/           │
  │ integrations/{sklearn,    │            │  JobSpec→TaskSpec[]    │      │  LocalProcessLauncher│
  │   pytorch, huggingface}   │            │ + isolation fail-close │      │ strategies/          │
  │  build a CommandWorkload  │            │   placement gate       │      │  CommandCompiler     │
  └───────────────────────────┘            └──────────────────────┘      └─────────────────────┘
       WHAT to run                            HOW it's shaped                WHERE/HOW it starts
```

### 4.1 Core: `flashruntime/workloads/command.py` (new, no framework imports)

```python
class Source(BaseModel):
    path: str = "."               # local working dir → process cwd (v1)
    git_revision: str | None = None   # reserved for remote packaging (later)

class OutputSpec(BaseModel):
    prefix: str = "artifact://jobs/{job_id}/"     # reuses artifact:// scheme
    collect: list[str] = Field(default_factory=lambda: ["metrics.json"])  # globs, cwd-relative

class CommandWorkload(BaseModel):
    command: str | list[str]          # "python train.py --lr 0.1" OR ["python","train.py",...]
    source: Source = Source()
    image: ImageSpec | None = None    # None ⇒ current/local env (Mode 0 / dev). 'latest' already rejected.
    env: dict[str, str] = {}          # merged over base env; NEVER secrets (launcher injects those)
    inputs: dict[str, str] = {}       # name → artifact:// URI
    outputs: OutputSpec = OutputSpec()
    resources: Requirements = Requirements()   # reuse providers.Requirements
    isolation: IsolationSpec = IsolationSpec() # default standard
    mode: Literal["auto","local","independent_tasks","coordinated"] = "auto"
    checkpoint: CheckpointPolicy | None = None
    task_params: list[dict] | None = None   # Mode A fan-out: one command instance per param set

    def argv(self) -> list[str]: ...  # shlex.split(str) or list as-is; no shell=True
```

- `command` string is `shlex.split` — no `shell=True` (injection-safe). Shell
  features (`|`, `&&`) require explicit `bash -c "..."`; documented.
- `{param}` placeholders in `command`/`env` are substituted per `task_params`
  entry (Mode A), e.g. `"python train.py --lr {lr}"`.
- **`mode="auto"` resolution** (deterministic, done at submit/expand time):
  `task_params` present ⇒ `independent_tasks`; else an adapter that set a
  multi-process launcher (torchrun) ⇒ `coordinated`; else `local` (Mode 0).
  A user may always pin `mode` explicitly to override.
- No protocol schema change: on the wire this is
  `WorkloadSpec{type:"command", parameters:{...}}` (`parameters` is already
  `dict[str,Any]`).

### 4.2 Execution: `launchers/local.py` — first concrete `Launcher`

`LocalProcessLauncher(Launcher)`, `name="local"`:
- `launch(spec: LaunchSpec, job_id, attempt_id)` → `LaunchHandle`.
- Runs `subprocess.Popen(spec.argv, cwd=workdir, env=base|spec.env)`; a fresh
  `OUTDIR` is created and exported as `FLASHML_OUTPUT_DIR` (opt-in for user
  code; unmodified code just writes to its own paths).
- Non-blocking `poll()` maps exit code → `LaunchState` (0→SUCCEEDED, else
  FAILED); caches terminal state; `logs()` returns captured stdout/stderr tail;
  `execution_id` = pid.
- After terminal, collects `outputs.collect` globs (cwd-relative) into the
  artifact store under the resolved prefix.
- Honors the launcher contract: raises `LaunchError` only before anything
  starts; once running, failures are reported via `poll`, never raised.

### 4.3 Execution: `strategies/command.py` — first concrete `StrategyCompiler`

`CommandCompiler(StrategyCompiler)`, `family="local_process"`:
`CommandWorkload`/`StrategyPlan` → `LaunchSpec(argv, env, world_size, files)`.
Pure/deterministic. For coordinated mode it emits the `torchrun` argv (see
4.5). Minimal; no environment inspection.

### 4.4 Service: `recipes/command.py` — first concrete `WorkloadRecipe`

`CommandRecipe(WorkloadRecipe)`, `kind="command"`:
- `expand(job_id, spec)` → `TaskSpec[]`. One task per `task_params` entry (or a
  single task if none). Payload carries **additively**: `argv`, `env`,
  `inputs`, `output_prefix`, `task_id`, `image`, `isolation`
  (tier+allowFallback), and `checkpoint` when set. Deterministic `trial-NNN`
  ids; `commit_key = <output_prefix>/metrics.json`; honors
  `retryPolicy.maxTaskAttempts` and `lease_seconds`.
- `validate_output(metrics)` — default (recipe-agnostic); adapters can tighten.
- `service/modea.py` `expand_tasks` dispatches to `recipe_for(workload.type)`;
  existing `hyperparameter_search`/`sharded_kmeans` remain (migrated onto the
  registry, no wire change).
- **Isolation gate:** command tasks stamp their required tier. The claim path
  refuses a `sandboxed` task to a non-`sandbox_capable` node (minimal
  fail-closed check now; full `scheduler.PlacementPolicy` capability filter is
  a flagged follow-up). New event surfaced when a task can't be placed.

### 4.5 Framework adapters: `flashruntime/integrations/` (thin, no framework imports at module level)

Each builds a `CommandWorkload` (SDK) and can emit a `JobSpec` (service). They
encode **launch/checkpoint conventions only**, never model code (rule 6: no
`import torch/sklearn/transformers`).

- `integrations/sklearn.py` — `hpo(script=..., grid={...})` /
  `sweep(task_params=[...])` → **Mode A** `CommandWorkload`
  (`mode="independent_tasks"`). Runs locally.
- `integrations/pytorch.py` — `ddp(script, nproc_per_node, nnodes=1,
  script_args=..., backend="auto")` → **Mode B** `CommandWorkload` whose argv
  is `torchrun --nproc-per-node=N --nnodes=M --rdzv-backend=c10d <script> ...`,
  `mode="coordinated"`. `backend="auto"` lets the user's script pick
  gloo(CPU)/nccl(GPU). Rank/world-size come from torchrun.
- `integrations/huggingface.py` — `trainer(script, nproc_per_node=...,
  accelerate=False)` → **Mode B** via torchrun (or `accelerate launch`) +
  checkpoint-manifest wiring on the existing checkpoint contract.

Multi-GPU is spec-built + unit-tested here; single-node multi-process DDP is
proven on CPU via gloo (see §6).

### 4.6 In-script PyTorch helper: `flashruntime/torch/` (optional sugar)

A thin **in-training-script** API so a user can `import flashruntime.torch`
inside their own `train.py` and get DDP wiring + FlashRuntime fault
tolerance without boilerplate. Two halves of one story: the **launcher**
(4.5's `pytorch.ddp` adapter) starts N processes with `RANK`/`WORLD_SIZE`;
this helper, *inside* each process, reads them and wires torch's own DDP.

```python
# user's train.py
import flashruntime.torch as ft

model, optimizer, loader = ft.prepare(model, optimizer, loader)
#  launched distributed → init_process_group + torch DDP wrap + DistributedSampler
#  launched as plain `python train.py` → NO-OP; runs single-process unchanged
for step, batch in enumerate(loader):
    ...
    ft.checkpoint(model, optimizer, step=step)   # parts-first, manifest-last
    ft.log_metrics({"loss": float(loss)})        # → ledger/dashboard (best-effort)
```

Surface (complete, deliberately small):
- `ft.prepare(model, optimizer=None, dataloader=None)` — if `WORLD_SIZE>1` in
  env: `init_process_group` (gloo on CPU / nccl on CUDA), wrap model in
  `DistributedDataParallel`, swap the loader's sampler for
  `DistributedSampler`. Else: return inputs unchanged. Idempotent.
- `ft.checkpoint(model, optimizer, step, every=None)` — rank-0 saves
  state_dicts through the **checkpoint manifest contract** (parts uploaded
  first, manifest committed last, hash-verified) under
  `FLASHML_OUTPUT_DIR/ckpt/`; other ranks no-op + barrier. On restart,
  `ft.prepare` (or `ft.latest_checkpoint()`) restores from the newest *valid*
  manifest — this is what makes restart-from-safe-checkpoint automatic.
- `ft.log_metrics(dict)` — appends to `FLASHML_OUTPUT_DIR/metrics.jsonl` and
  (when a coordinator URL is present) POSTs to the ledger; never raises.
- `ft.rank()`, `ft.world_size()`, `ft.is_main()` — trivial env readers.

**Guardrail (ADR-0003: do not rebuild Accelerate):** this module wraps
torch's own primitives and STOPS at the list above. No FSDP sharding
policies, no mixed-precision management, no DeepSpeed config — users wanting
those use the real framework features, which our launcher still launches
correctly. The helper is optional: vanilla DDP scripts run unmodified
without it. `torch` is imported lazily inside `flashruntime/torch/`; the
pydantic-only core never depends on it.

### 4.7 SDK: `flash.submit()`

```python
run = flash.submit(workload: CommandWorkload, provider="local", wait=True)
run.state        # LaunchState
run.wait()       # → terminal LaunchState
run.logs()       # combined tail
run.artifacts    # collected ArtifactRecords
```

v1 local execution is synchronous (mirrors the current engine's replay model):
`compile → launch → wait → collect`. Exported lazily via PEP 562 so the
pydantic-only core stays clean. `flash.CommandWorkload` and
`flash.integrations` also exported lazily.

## 5. Data flow

- **SDK local:** `CommandWorkload → CommandCompiler → LaunchSpec →
  LocalProcessLauncher → subprocess → collect outputs → RunHandle`.
- **Service:** `POST JobSpec(type=command) → CommandRecipe.expand → TaskSpec[]
  → LeaseManager → [flashnode sandboxed runner executes argv] → sha256 commit
  validation`, with fail-closed isolation placement.

## 6. Example user code + end-to-end demo (the acceptance test)

Ships real "user-authored" repos that FlashRuntime runs **unmodified**, plus a
runnable demo. This is how the user verifies real ML tasks/datasets run.

- `examples/user_sklearn/` — `train.py` + a tiny CSV dataset (or sklearn
  `load_*`): fits an estimator, cross-validates, writes `metrics.json`. Run via
  `integrations.sklearn.hpo(...)` as a Mode A sweep.
- `examples/user_pytorch/` — `train.py`: a small MLP/CNN on a synthetic or
  bundled dataset, written with `flashruntime.torch` (`ft.prepare` +
  `ft.checkpoint` + `ft.log_metrics`), so the SAME script runs single-process
  (Mode 0) and multi-process DDP on CPU (gloo) via the `pytorch.ddp` adapter
  + `LocalProcessLauncher`. Doubles as the helper's integration test.
- `examples/user_pytorch_vanilla/` — the same model as a **plain torch DDP
  script with no flashruntime import**, proving unmodified code needs only
  the launcher (the helper is optional sugar, not a requirement).
- `examples/bring_your_code_demo.py` — submits both through `flash.submit`,
  streams events, prints metrics + collected artifacts. A `make demo-workloads`
  target (or documented `python -m` invocation).

Requires the user to `uv pip install -e ".[sklearn,dev]"` and (for the torch
example) `pip install torch` (CPU wheel is fine).

## 7. Documentation

`docs/guides/bring-your-code.md`:
- The operator/user split (who owns what).
- Run an existing repo locally (SDK) — sklearn, PyTorch, HF, each with a code
  block.
- Submit as a JobSpec (service) + the isolation tiers (standard vs sandboxed;
  community machines → sandbox).
- How built-in `algorithms/` examples relate (batteries-included, not required).
- Honest "what runs where today" box (local runs now; sandboxed/remote/multi-GPU
  need flashnode/hardware).

## 8. Testing (all pytest, no-infra tier)

- `test_workloads_command.py` — validation, `argv()` from string/list,
  `{param}` substitution, `bash -c` passthrough.
- `test_launcher_local.py` — runs a real subprocess: success collects
  artifacts; non-zero exit → FAILED; logs captured; terminal state cached.
- `test_strategy_command.py` — compiler determinism; coordinated → torchrun
  argv/env.
- `test_integrations.py` — sklearn/pytorch/hf each build the expected
  argv/env/JobSpec (no framework import needed to assert the spec).
- `test_service_command_recipe.py` — `expand` produces correct TaskSpecs incl.
  isolation; **isolation fail-closed** (sandboxed task not leased to a
  non-sandbox node); existing expansions still pass.
- `test_torch_helper.py` — `ft.prepare` no-ops without `WORLD_SIZE`; wires
  DDP+sampler under a 2-process gloo group; `ft.checkpoint` writes
  parts-first/manifest-last and restores the newest valid manifest;
  `ft.log_metrics` never raises. Skips cleanly when torch is absent.
- `examples/user_pytorch` DDP-on-CPU smoke is opt-in (skips if torch absent).

## 9. Acceptance criteria

1. `flash.submit(CommandWorkload(command="python train.py", source="…"))` runs
   an **unmodified** user repo locally, collects `metrics.json`, and surfaces a
   non-zero exit as FAILED — no algorithm class, no Docker.
2. `integrations.sklearn.hpo(...)` runs a real sklearn sweep and returns the
   best trial's metrics.
3. `integrations.pytorch.ddp(script=…, nproc_per_node=2)` launches real
   **2-process DDP on CPU (gloo)** via `torchrun` and trains to completion —
   both the `flashruntime.torch`-based script and the vanilla-DDP script.
3b. Kill the `ft.prepare`-based run mid-training; resubmit; it resumes from
   the newest **valid** checkpoint manifest and finishes (the automatic
   fault-tolerance story, demonstrated locally).
4. A `command`-type JobSpec expands to lease tasks; a `sandboxed` task is
   **never** leased to a non-`sandbox_capable` node (fail-closed test green).
5. `Cluster.train()` + `algorithms/` unchanged; full existing test suite green;
   core stays pydantic-only (clean-venv import smoke passes).
6. `docs/guides/bring-your-code.md` shows all three frameworks + isolation
   tiers; the demo script runs end to end.

## 10. Open follow-ups (flagged, not in this slice)

- flashnode executor: accept `argv` payloads + implement the sandbox tiers.
- Full `scheduler.PlacementPolicy` capability filter (beyond the minimal gate).
- Remote provider adapter (RunPod) + source packaging for remote (`git_revision`).
- `flash.run(StrategyPlan)` wiring (planner → this path).
- Async/background `flash.submit` (v1 is synchronous local).
```
