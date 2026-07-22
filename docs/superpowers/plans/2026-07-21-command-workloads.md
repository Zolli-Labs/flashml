# Command Workloads ("bring your own code") Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users run their own sklearn / PyTorch / Hugging Face code through FlashRuntime — `flash.submit(CommandWorkload(...))` locally, and `JobSpec{workload.type: "command"}` on the lease coordinator — with framework adapters and an in-script `flashruntime.torch` helper for automatic DDP wiring + checkpoint/resume.

**Architecture:** Additive extension of the four-axes design (ADR-0003). A new user-facing `CommandWorkload` compiles into the existing `LaunchSpec`; the first concrete `Launcher` (subprocess) executes it locally; the first concrete `WorkloadRecipe` expands it into lease tasks with a fail-closed isolation gate (sandboxed tasks only to `sandbox_capable` nodes). Framework adapters build workloads; they never import frameworks. Spec: `docs/superpowers/specs/2026-07-21-command-workloads-design.md`.

**Tech Stack:** Python ≥3.10, pydantic v2, pytest. Optional at runtime (never in core imports): scikit-learn, torch/torchrun, transformers.

## Global Constraints

Copied from AGENTS.md / the spec — every task implicitly includes these:

- **Pydantic-only core:** `import flashruntime` and every new core module (`workloads/`, `strategies/`, `recipes/`, `scheduler/`, `checkpoint/`) must import nothing beyond stdlib + pydantic. `torch`/`sklearn`/`transformers` are imported ONLY lazily inside functions that run in the user's process (`flashruntime/torch/`, the HF callback factory).
- **No framework imports in coordinator-side code** (`service/`, `recipes/`, `integrations/` module level).
- **Schemas versioned, additive only** within v1alpha1. No changes to `protocol/` in this plan — none are needed.
- **Fail closed on security-relevant fields:** a task requiring `isolation.tier="sandboxed"` must never be leased to a node not advertising `sandbox_capable` (unknown ⇒ not capable), unless the task's `allowFallback` is true.
- **Ledger is append-only; status derived from events** — do not add hand-mutated status fields.
- **Launchers never retry or recover** — they report (`launchers/__init__.py` contract).
- **`flashruntime.torch` guardrail:** surface is exactly prepare / checkpoint / log_metrics / rank / world_size / is_main / start_step. No FSDP policies, no autocast, no DeepSpeed config.
- **Branch:** all work on `local-milestone-2026-07` (current branch). Commit after every task.
- **Run the full suite** (`pytest`) before each commit; existing 109+ tests must stay green.
- Match existing code style: module docstrings that explain *why*, small modules, `from __future__ import annotations`.

## File Map

| File | Status | Responsibility |
|---|---|---|
| `flashruntime/workloads/__init__.py` | create | re-exports |
| `flashruntime/workloads/command.py` | create | `CommandWorkload`, `Source`, `OutputSpec`, `to_jobspec()` |
| `flashruntime/strategies/command.py` | create | `compile_workload()` → `LaunchSpec` |
| `flashruntime/launchers/local.py` | create | `LocalProcessLauncher`, `LocalLaunchHandle` |
| `flashruntime/checkpoint/local.py` | create | local manifest IO: `write_manifest`, `latest_valid_manifest` |
| `flashruntime/sdk.py` | create | `submit()`, `Run` |
| `flashruntime/__init__.py` | modify | lazy SDK exports |
| `flashruntime/integrations/{__init__,sklearn,pytorch,huggingface}.py` | create | framework adapters |
| `flashruntime/torch/__init__.py` | create | `ft.prepare/checkpoint/log_metrics/...` |
| `flashruntime/scheduler/__init__.py` | modify | add `IsolationAwarePlacement` |
| `flashruntime/leases/manager.py` | modify | optional `policy`/`node` kwargs on `claim()` |
| `flashruntime/recipes/command.py` | create | `CommandRecipe` (registered) |
| `flashruntime/service/modea.py` | modify | recipe dispatch + isolation-aware claim |
| `examples/user_sklearn/train.py` | create | ordinary sklearn script (no flashruntime import) |
| `examples/user_pytorch/train.py` | create | PyTorch script using `flashruntime.torch` |
| `examples/user_pytorch_vanilla/train.py` | create | plain DDP script (no flashruntime import) |
| `examples/bring_your_code_demo.py` | create | end-to-end demo |
| `docs/guides/bring-your-code.md` | create | user guide |
| `tests/test_workloads_command.py` … (per task) | create | see tasks |

Note (conscious simplification vs the spec): `strategies/command.py` ships `compile_workload()` as a pure module function rather than a `StrategyCompiler` subclass — a `StrategyPlan` carries no argv, so registering a compiler for it is meaningless until `flash.run()` wiring lands (flagged follow-up, spec §10).

---

### Task 1: `CommandWorkload` model

**Files:**
- Create: `flashruntime/workloads/__init__.py`, `flashruntime/workloads/command.py`
- Test: `tests/test_workloads_command.py`

**Interfaces:**
- Consumes: `IsolationSpec`, `ImageSpec`, `JobSpec`, `JobSpecInner`, `JobMetadata`, `ExecutionSpec`, `WorkloadSpec` from `flashruntime.protocol.v1alpha1`; `CheckpointPolicy` from `flashruntime.protocol.plan_v1alpha1`; `Requirements` from `flashruntime.providers`.
- Produces: `CommandWorkload` (fields as coded below; methods `argv(params: dict | None) -> list[str]`, `resolved_mode() -> str`), `Source(path, git_revision)`, `OutputSpec(prefix, collect, primary_metric, maximize)`, `to_jobspec(workload, name, image=None) -> JobSpec`. Later tasks import all of these from `flashruntime.workloads.command`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_workloads_command.py
"""CommandWorkload: the user-facing 'bring your own code' description."""

from __future__ import annotations

import pytest


def _wl(**over):
    from flashruntime.workloads.command import CommandWorkload

    base = dict(command="python train.py")
    base.update(over)
    return CommandWorkload(**base)


def test_argv_from_string_and_list():
    assert _wl(command="python train.py --lr 0.1").argv() == [
        "python", "train.py", "--lr", "0.1"
    ]
    assert _wl(command=["python", "train.py"]).argv() == ["python", "train.py"]


def test_argv_substitutes_task_params():
    wl = _wl(command="python train.py --lr {lr}")
    assert wl.argv({"lr": 0.01}) == ["python", "train.py", "--lr", "0.01"]


def test_argv_missing_placeholder_raises():
    with pytest.raises(KeyError):
        _wl(command="python train.py --lr {lr}").argv({"seed": 1})


def test_inputs_must_be_artifact_uris():
    with pytest.raises(ValueError, match="artifact://"):
        _wl(inputs={"dataset": "/tmp/data.csv"})
    wl = _wl(inputs={"dataset": "artifact://jobs/x/data.csv"})
    assert wl.inputs["dataset"].startswith("artifact://")


def test_resolved_mode_auto_rules():
    assert _wl().resolved_mode() == "local"
    assert _wl(task_params=[{"lr": 0.1}]).resolved_mode() == "independent_tasks"
    assert _wl(command="torchrun --nproc-per-node=2 train.py").resolved_mode() == "coordinated"
    # explicit pin always wins
    assert _wl(task_params=[{"lr": 0.1}], mode="local").resolved_mode() == "local"


def test_to_jobspec_requires_image_and_carries_isolation():
    from flashruntime.protocol.v1alpha1 import ImageSpec, IsolationSpec
    from flashruntime.workloads.command import to_jobspec

    wl = _wl(
        command="python train.py --lr {lr}",
        task_params=[{"lr": 0.1}, {"lr": 0.01}],
        isolation=IsolationSpec(tier="sandboxed"),
    )
    with pytest.raises(ValueError, match="image"):
        to_jobspec(wl, name="sweep")

    spec = to_jobspec(wl, name="sweep", image=ImageSpec(repository="ghcr.io/me/x", tag="1.0"))
    assert spec.spec.execution.backend == "leases"
    assert spec.spec.workload.type == "command"
    assert spec.spec.workload.parameters["command"] == ["python", "train.py", "--lr", "{lr}"]
    assert spec.spec.workload.parameters["task_params"] == [{"lr": 0.1}, {"lr": 0.01}]
    assert spec.spec.isolation.tier == "sandboxed"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_workloads_command.py -v`
Expected: FAIL/ERROR with `ModuleNotFoundError: No module named 'flashruntime.workloads'`

- [ ] **Step 3: Implement**

```python
# flashruntime/workloads/command.py
"""User-facing description of a "bring your own code" workload.

A CommandWorkload names WHAT to run (a command in a source directory,
optionally in a pinned image) and what FlashRuntime should do around it
(inputs, outputs, isolation, Mode A fan-out). It never describes HOW
distributed math happens — that belongs to the user's code and its
framework (ADR-0003: FlashRuntime operates jobs, it does not train).

Pydantic-only: importing this module must never require torch, sklearn,
kubernetes, or fastapi — it is part of the clean core.
"""

from __future__ import annotations

import shlex
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from flashruntime.protocol.plan_v1alpha1 import CheckpointPolicy
from flashruntime.protocol.v1alpha1 import (
    ExecutionSpec,
    ImageSpec,
    IsolationSpec,
    JobMetadata,
    JobSpec,
    JobSpecInner,
    WorkloadSpec,
)
from flashruntime.providers import Requirements


class Source(BaseModel):
    """Where the user's code lives. v1 executes from a local directory;
    `git_revision` is reserved for remote packaging (spec §10 follow-up)."""

    path: str = "."
    git_revision: str | None = None


class OutputSpec(BaseModel):
    """What to keep after a run. `collect` globs are resolved against the
    script's working directory; `primary_metric` names the metrics.json key
    Run.best_trial() ranks by."""

    prefix: str = "artifact://jobs/{job_id}/"
    collect: list[str] = Field(default_factory=lambda: ["metrics.json"])
    primary_metric: str | None = None
    maximize: bool = True


class CommandWorkload(BaseModel):
    """One command, operated by FlashRuntime.

    `command` may be a shell-style string (shlex-split, never shell=True —
    pipes need an explicit `bash -c "..."`) or an argv list. `{name}`
    placeholders are filled per `task_params` entry for Mode A fan-out.
    """

    command: str | list[str]
    source: Source = Field(default_factory=Source)
    image: ImageSpec | None = None
    env: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: OutputSpec = Field(default_factory=OutputSpec)
    resources: Requirements = Field(default_factory=Requirements)
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    mode: Literal["auto", "local", "independent_tasks", "coordinated"] = "auto"
    checkpoint: CheckpointPolicy | None = None
    task_params: list[dict] | None = None

    @field_validator("inputs")
    @classmethod
    def _artifact_scheme(cls, v: dict[str, str]) -> dict[str, str]:
        for name, uri in v.items():
            if not str(uri).startswith("artifact://"):
                raise ValueError(f"input '{name}' must be an artifact:// URI, got {uri!r}")
        return v

    def argv(self, params: dict | None = None) -> list[str]:
        """Exec-ready argv. `params` fills `{name}` placeholders; a
        placeholder with no matching param raises KeyError (a silent empty
        substitution would corrupt the command)."""
        tokens = shlex.split(self.command) if isinstance(self.command, str) else list(self.command)
        if params:
            tokens = [t.format(**params) for t in tokens]
        return tokens

    def resolved_mode(self) -> str:
        """Deterministic `auto` resolution (spec §4.1): fan-out params ⇒
        independent_tasks; a multi-process launcher command ⇒ coordinated;
        else local. An explicit `mode` always wins."""
        if self.mode != "auto":
            return self.mode
        if self.task_params:
            return "independent_tasks"
        tokens = self.argv()
        if tokens and tokens[0] in ("torchrun", "accelerate"):
            return "coordinated"
        return "local"


def to_jobspec(workload: CommandWorkload, name: str, image: ImageSpec | None = None) -> JobSpec:
    """Wire form for the coordinator: JobSpec{execution.backend: leases,
    workload.type: "command"}. A pinned image is required — remote runs
    must be reproducible (the schema already rejects 'latest')."""
    img = image or workload.image
    if img is None:
        raise ValueError("a pinned image is required to submit a command workload to the service")
    parameters: dict = {
        "command": workload.argv(),  # normalized argv, placeholders intact
        "env": dict(workload.env),
        "inputs": dict(workload.inputs),
    }
    if workload.task_params is not None:
        parameters["task_params"] = workload.task_params
    if workload.checkpoint is not None:
        parameters["checkpoint"] = workload.checkpoint.model_dump()
    return JobSpec(
        metadata=JobMetadata(name=name),
        spec=JobSpecInner(
            execution=ExecutionSpec(backend="leases"),
            image=img,
            workload=WorkloadSpec(type="command", parameters=parameters),
            isolation=workload.isolation,
        ),
    )
```

```python
# flashruntime/workloads/__init__.py
"""Workload descriptions: WHAT the user wants run (four-axes rule: this is
the axis-zero input the recipes/strategies/launchers axes consume)."""

from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source, to_jobspec

__all__ = ["CommandWorkload", "OutputSpec", "Source", "to_jobspec"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_workloads_command.py -v`
Expected: all PASS

- [ ] **Step 5: Full suite + commit (include the spec + this plan)**

Run: `pytest`
Expected: no failures.

```bash
git add flashruntime/workloads tests/test_workloads_command.py docs/superpowers/
git commit -m "feat(workloads): CommandWorkload — the bring-your-own-code description"
```

---

### Task 2: `compile_workload()` → LaunchSpec

**Files:**
- Create: `flashruntime/strategies/command.py`
- Test: `tests/test_strategy_command.py`

**Interfaces:**
- Consumes: `LaunchSpec` from `flashruntime.strategies`; `CommandWorkload` from Task 1.
- Produces: `compile_workload(workload: CommandWorkload, params: dict | None = None) -> LaunchSpec` with `argv` filled, `env` substituted, `world_size` parsed from torchrun args, `workdir_hint = workload.source.path`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_strategy_command.py
from __future__ import annotations


def _wl(**over):
    from flashruntime.workloads.command import CommandWorkload

    base = dict(command="python train.py")
    base.update(over)
    return CommandWorkload(**base)


def test_compile_is_pure_and_deterministic():
    from flashruntime.strategies.command import compile_workload

    wl = _wl(command="python train.py --lr {lr}", env={"TAG": "run-{lr}"})
    a = compile_workload(wl, {"lr": 0.1})
    b = compile_workload(wl, {"lr": 0.1})
    assert a == b
    assert a.argv == ["python", "train.py", "--lr", "0.1"]
    assert a.env == {"TAG": "run-0.1"}


def test_workdir_hint_carries_source_path():
    from flashruntime.strategies.command import compile_workload
    from flashruntime.workloads.command import Source

    spec = compile_workload(_wl(source=Source(path="/home/me/proj")))
    assert spec.workdir_hint == "/home/me/proj"


def test_torchrun_world_size_extracted():
    from flashruntime.strategies.command import compile_workload

    spec = compile_workload(_wl(command="torchrun --nproc-per-node=4 --standalone train.py"))
    assert spec.world_size == 4
    assert spec.argv[0] == "torchrun"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_strategy_command.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flashruntime.strategies.command'`

- [ ] **Step 3: Implement**

```python
# flashruntime/strategies/command.py
"""Compile a CommandWorkload into the backend-neutral LaunchSpec.

Pure and deterministic (same rules as StrategyCompiler): no environment
inspection, no filesystem access — resolving the workdir and creating
output directories is the launcher's job.

Note: this is a module function, not a StrategyCompiler subclass — a
StrategyPlan carries no argv, so a plan-driven compiler for commands is
meaningless until flash.run() wiring lands (spec §10 follow-up).
"""

from __future__ import annotations

from flashruntime.strategies import LaunchSpec
from flashruntime.workloads.command import CommandWorkload


def compile_workload(workload: CommandWorkload, params: dict | None = None) -> LaunchSpec:
    argv = workload.argv(params)
    env = {k: (v.format(**params) if params else v) for k, v in workload.env.items()}
    world_size = 1
    notes = [f"mode={workload.resolved_mode()}"]
    if argv and argv[0] == "torchrun":
        for token in argv:
            if token.startswith("--nproc-per-node="):
                world_size = int(token.split("=", 1)[1])
                notes.append(f"world_size from torchrun: {world_size}")
    return LaunchSpec(
        argv=argv,
        env=env,
        world_size=world_size,
        workdir_hint=workload.source.path,
        notes=notes,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_strategy_command.py -v` → all PASS

- [ ] **Step 5: Full suite + commit**

```bash
pytest && git add flashruntime/strategies/command.py tests/test_strategy_command.py \
  && git commit -m "feat(strategies): compile CommandWorkload into LaunchSpec"
```

---

### Task 3: `LocalProcessLauncher` — first concrete Launcher

**Files:**
- Create: `flashruntime/launchers/local.py`
- Test: `tests/test_launcher_local.py`

**Interfaces:**
- Consumes: `Launcher`, `LaunchHandle`, `LaunchState`, `LaunchError` from `flashruntime.launchers`; `LaunchSpec` from `flashruntime.strategies`.
- Produces: `LocalProcessLauncher(output_root: str | Path)` with `launch(spec, job_id, attempt_id) -> LocalLaunchHandle`; the handle adds a public `output_dir: Path` attribute. Contract env vars set for the child: `FLASHML_OUTPUT_DIR` (= `<output_root>/<job_id>/<attempt_id>`), `FLASHML_CKPT_DIR` (= `<output_root>/<job_id>/ckpt` — **job-scoped**, so a restarted attempt finds its predecessor's checkpoints), `FLASHML_JOB_ID`, `FLASHML_ATTEMPT_ID`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_launcher_local.py
"""LocalProcessLauncher runs real subprocesses — these tests execute
`python -c` children and assert the Launcher contract."""

from __future__ import annotations

import sys

import pytest


def _launcher(tmp_path):
    from flashruntime.launchers.local import LocalProcessLauncher

    return LocalProcessLauncher(output_root=tmp_path / "runs")


def _spec(code: str, workdir: str):
    from flashruntime.strategies import LaunchSpec

    return LaunchSpec(argv=[sys.executable, "-c", code], workdir_hint=workdir)


def test_success_sets_contract_env_and_succeeds(tmp_path):
    from flashruntime.launchers import LaunchState

    code = (
        "import os, pathlib;"
        "out = pathlib.Path(os.environ['FLASHML_OUTPUT_DIR']);"
        "(out / 'proof.txt').write_text(os.environ['FLASHML_JOB_ID'] + ':' + os.environ['FLASHML_ATTEMPT_ID']);"
        "print('hello from child')"
    )
    handle = _launcher(tmp_path).launch(_spec(code, str(tmp_path)), "j1", "a1")
    assert handle.wait(timeout_seconds=30) is LaunchState.SUCCEEDED
    assert (handle.output_dir / "proof.txt").read_text() == "j1:a1"
    assert "hello from child" in handle.logs()
    assert handle.execution_id  # pid string


def test_ckpt_dir_is_job_scoped_not_attempt_scoped(tmp_path):
    code = "import os; print(os.environ['FLASHML_CKPT_DIR'])"
    launcher = _launcher(tmp_path)
    h1 = launcher.launch(_spec(code, str(tmp_path)), "j1", "a1")
    h2 = launcher.launch(_spec(code, str(tmp_path)), "j1", "a2")
    h1.wait(30), h2.wait(30)
    assert h1.logs().strip() == h2.logs().strip()  # same ckpt tree across attempts


def test_nonzero_exit_reports_failed_and_caches(tmp_path):
    from flashruntime.launchers import LaunchState

    handle = _launcher(tmp_path).launch(_spec("raise SystemExit(3)", str(tmp_path)), "j1", "a1")
    assert handle.wait(30) is LaunchState.FAILED
    assert handle.poll() is LaunchState.FAILED  # terminal state cached


def test_preflight_failures_raise_launch_error(tmp_path):
    from flashruntime.launchers import LaunchError
    from flashruntime.strategies import LaunchSpec

    with pytest.raises(LaunchError, match="workdir"):
        _launcher(tmp_path).launch(
            LaunchSpec(argv=["true"], workdir_hint=str(tmp_path / "nope")), "j", "a"
        )
    with pytest.raises(LaunchError, match="argv"):
        _launcher(tmp_path).launch(LaunchSpec(argv=[], workdir_hint=str(tmp_path)), "j", "a")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_launcher_local.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flashruntime.launchers.local'`

- [ ] **Step 3: Implement**

```python
# flashruntime/launchers/local.py
"""LocalProcessLauncher — the first concrete Launcher.

Runs a LaunchSpec as one OS process on this machine: cwd from
`workdir_hint`, the caller's environment merged UNDER `spec.env` and the
FlashRuntime contract variables, stdout+stderr captured to a log file in
the attempt's output directory. This is Mode 0 execution and the substrate
under `flash.submit(...)`'s local path.

Contract variables exported to the child (opt-in for user code):
  FLASHML_OUTPUT_DIR  — per-attempt scratch/output directory
  FLASHML_CKPT_DIR    — per-JOB checkpoint tree (attempts share it, so a
                        restarted attempt can restore its predecessor's
                        manifests — the resume path depends on this)
  FLASHML_JOB_ID / FLASHML_ATTEMPT_ID

Honors the Launcher contract: LaunchError only before a process exists;
after that, every failure is reported through poll(), never raised — and
this launcher never retries (recovery belongs to the coordinator).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from flashruntime.launchers import Launcher, LaunchError, LaunchHandle, LaunchState
from flashruntime.strategies import LaunchSpec


class LocalLaunchHandle(LaunchHandle):
    def __init__(self, proc: subprocess.Popen, log_path: Path, output_dir: Path):
        self._proc = proc
        self._log_path = log_path
        self.output_dir = output_dir
        self._final: LaunchState | None = None
        self._cancelled = False

    def poll(self) -> LaunchState:
        if self._final is not None:
            return self._final
        code = self._proc.poll()
        if code is None:
            return LaunchState.RUNNING
        if self._cancelled:
            self._final = LaunchState.CANCELLED
        else:
            self._final = LaunchState.SUCCEEDED if code == 0 else LaunchState.FAILED
        return self._final

    def cancel(self) -> None:
        if self.poll().terminal:
            return
        self._cancelled = True
        self._proc.terminate()

    def wait(self, timeout_seconds: float | None = None) -> LaunchState:
        # native wait beats the ABC's 1 s polling loop
        try:
            self._proc.wait(timeout=timeout_seconds)
        except subprocess.TimeoutExpired:
            pass
        return self.poll()

    def logs(self, tail_lines: int = 200) -> str:
        if not self._log_path.is_file():
            return ""
        lines = self._log_path.read_text(errors="replace").splitlines()
        return "\n".join(lines[-tail_lines:])

    @property
    def execution_id(self) -> str:
        return str(self._proc.pid)


class LocalProcessLauncher(Launcher):
    name = "local"

    def __init__(self, output_root: str | Path):
        self._output_root = Path(output_root)

    def launch(self, spec: LaunchSpec, job_id: str, attempt_id: str) -> LocalLaunchHandle:
        workdir = Path(spec.workdir_hint or ".").expanduser()
        if not spec.argv:
            raise LaunchError("empty argv")
        if not workdir.is_dir():
            raise LaunchError(f"workdir does not exist: {workdir}")
        outdir = self._output_root / job_id / attempt_id
        outdir.mkdir(parents=True, exist_ok=True)
        for name, content in spec.files.items():
            (outdir / name).write_text(content)
        env = {
            **os.environ,
            **spec.env,
            "FLASHML_OUTPUT_DIR": str(outdir),
            "FLASHML_CKPT_DIR": str(self._output_root / job_id / "ckpt"),
            "FLASHML_JOB_ID": job_id,
            "FLASHML_ATTEMPT_ID": attempt_id,
        }
        log_path = outdir / "launcher.log"
        try:
            log_file = open(log_path, "w")
            proc = subprocess.Popen(
                spec.argv,
                cwd=str(workdir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise LaunchError(f"failed to start {spec.argv[0]!r}: {exc}") from exc
        return LocalLaunchHandle(proc, log_path, outdir)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_launcher_local.py -v` → all PASS

- [ ] **Step 5: Full suite + commit**

```bash
pytest && git add flashruntime/launchers/local.py tests/test_launcher_local.py \
  && git commit -m "feat(launchers): LocalProcessLauncher, the first concrete launcher"
```

---

### Task 4: Local checkpoint manifests (`checkpoint/local.py`)

**Files:**
- Create: `flashruntime/checkpoint/local.py`
- Test: `tests/test_checkpoint_local.py`

**Interfaces:**
- Consumes: `CheckpointManifest`, `CheckpointPart`, `CheckpointValidation` from `flashruntime.protocol.v1alpha1`.
- Produces: `write_manifest(step_dir: Path, *, job_id, attempt_id, step, world_size=1, framework="") -> CheckpointManifest` and `latest_valid_manifest(ckpt_root: Path, pattern: str = "step-*") -> CheckpointManifest | None`. `MANIFEST_NAME = "manifest.json"`. Used by `flashruntime.torch` (Task 7) and the HF callback (Task 6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_checkpoint_local.py
"""Parts-first / manifest-last on a plain filesystem: a crash mid-write
leaves no manifest; a corrupted part disqualifies its manifest on read."""

from __future__ import annotations

import pytest


def _make_step(root, step: int, content: bytes = b"weights"):
    from flashruntime.checkpoint.local import write_manifest

    d = root / f"step-{step:06d}"
    d.mkdir(parents=True)
    (d / "model.pt").write_bytes(content)
    return write_manifest(d, job_id="j1", attempt_id="a1", step=step)


def test_manifest_written_last_with_verified_hashes(tmp_path):
    from flashruntime.checkpoint.local import MANIFEST_NAME

    manifest = _make_step(tmp_path, 10)
    assert (tmp_path / "step-000010" / MANIFEST_NAME).is_file()
    assert manifest.parts[0].key == "model.pt"
    assert manifest.validation.value == "hash_verified"


def test_zero_parts_refused(tmp_path):
    from flashruntime.checkpoint.local import write_manifest

    d = tmp_path / "step-000001"
    d.mkdir()
    with pytest.raises(ValueError, match="no part files"):
        write_manifest(d, job_id="j1", attempt_id="a1", step=1)


def test_latest_valid_picks_newest_step(tmp_path):
    from flashruntime.checkpoint.local import latest_valid_manifest

    _make_step(tmp_path, 10)
    _make_step(tmp_path, 20)
    assert latest_valid_manifest(tmp_path).step == 20


def test_corrupted_part_disqualifies_its_manifest(tmp_path):
    from flashruntime.checkpoint.local import latest_valid_manifest

    _make_step(tmp_path, 10)
    _make_step(tmp_path, 20)
    (tmp_path / "step-000020" / "model.pt").write_bytes(b"CORRUPTED")
    assert latest_valid_manifest(tmp_path).step == 10  # falls back, never loads bad state


def test_no_checkpoints_is_none(tmp_path):
    from flashruntime.checkpoint.local import latest_valid_manifest

    assert latest_valid_manifest(tmp_path / "missing") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_checkpoint_local.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError` for `flashruntime.checkpoint.local`

- [ ] **Step 3: Implement**

```python
# flashruntime/checkpoint/local.py
"""Local checkpoint manifests: parts-first / manifest-last for processes
that have only a filesystem (no coordinator).

`write_manifest` hashes every part file already on disk and writes
manifest.json LAST — a crash mid-checkpoint leaves part files but no
manifest, so the checkpoint does not exist. `latest_valid_manifest`
re-verifies every part hash on read: a corrupted or truncated part
disqualifies its manifest, so recovery can never restore from it.

Consumers: `flashruntime.torch.checkpoint()` and the Hugging Face Trainer
callback. Pure stdlib + pydantic (protocol models) — safe in the core.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

from flashruntime.protocol.v1alpha1 import (
    CheckpointManifest,
    CheckpointPart,
    CheckpointValidation,
)

MANIFEST_NAME = "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(
    step_dir: Path,
    *,
    job_id: str,
    attempt_id: str,
    step: int,
    world_size: int = 1,
    framework: str = "",
) -> CheckpointManifest:
    """Hash every part file in `step_dir`, then write the manifest LAST."""
    step_dir = Path(step_dir)
    parts = [
        CheckpointPart(key=p.name, sha256=_sha256(p), size_bytes=p.stat().st_size)
        for p in sorted(step_dir.iterdir())
        if p.is_file() and p.name != MANIFEST_NAME
    ]
    if not parts:
        raise ValueError(f"no part files in {step_dir}")
    manifest = CheckpointManifest(
        manifest_id=f"ck-{uuid.uuid4().hex[:12]}",
        job_id=job_id,
        attempt_id=attempt_id,
        step=step,
        framework=framework,
        world_size=world_size,
        compatible_world_sizes=[world_size],
        storage_prefix=str(step_dir),
        parts=parts,
        validation=CheckpointValidation.HASH_VERIFIED,
    )
    (step_dir / MANIFEST_NAME).write_text(manifest.model_dump_json(indent=2))  # LAST
    return manifest


def latest_valid_manifest(ckpt_root: Path, pattern: str = "step-*") -> CheckpointManifest | None:
    """Newest manifest whose parts all re-verify on disk, or None.

    `pattern` matches the per-step directory names ("step-*" for
    flashruntime.torch, "checkpoint-*" for Hugging Face Trainer output).
    """
    ckpt_root = Path(ckpt_root)
    if not ckpt_root.is_dir():
        return None
    best: CheckpointManifest | None = None
    for mf_path in ckpt_root.glob(f"{pattern}/{MANIFEST_NAME}"):
        try:
            manifest = CheckpointManifest.model_validate_json(mf_path.read_text())
        except ValueError:
            continue  # unreadable manifest: treat as nonexistent
        step_dir = mf_path.parent
        intact = all(
            (step_dir / part.key).is_file() and _sha256(step_dir / part.key) == part.sha256
            for part in manifest.parts
        )
        if intact and (best is None or manifest.step > best.step):
            best = manifest
    return best
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_checkpoint_local.py -v` → all PASS

- [ ] **Step 5: Full suite + commit**

```bash
pytest && git add flashruntime/checkpoint/local.py tests/test_checkpoint_local.py \
  && git commit -m "feat(checkpoint): local parts-first/manifest-last manifest IO"
```

---

### Task 5: `flash.submit()` + `Run` (SDK)

**Files:**
- Create: `flashruntime/sdk.py`
- Modify: `flashruntime/__init__.py` (lazy exports; see Step 3)
- Test: `tests/test_sdk_submit.py`

**Interfaces:**
- Consumes: `compile_workload` (Task 2), `LocalProcessLauncher` (Task 3), `CommandWorkload`/`OutputSpec` (Task 1), `LaunchState`.
- Produces: `submit(workload: CommandWorkload, output_dir: str | Path | None = None) -> Run`. `Run` fields: `.state: LaunchState`, `.trials: list[dict]`, `.artifacts: list[Path]`, `.output_dir: Path`; methods `.logs() -> str`, `.best_trial(metric=None, maximize=None) -> dict | None`. Lazy top-level exports: `flashruntime.submit`, `flashruntime.CommandWorkload`, `flashruntime.OutputSpec`, `flashruntime.Source`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sdk_submit.py
"""flash.submit(): compile → launch → wait → collect, on real subprocesses.
Uses stdlib-only child scripts so the test needs no ML frameworks."""

from __future__ import annotations

import json
import sys
import textwrap


def _write_script(tmp_path, body: str) -> str:
    src = tmp_path / "userproj"
    src.mkdir(exist_ok=True)
    (src / "train.py").write_text(textwrap.dedent(body))
    return str(src)


def test_single_command_collects_metrics(tmp_path):
    import flashruntime as flash

    source = _write_script(
        tmp_path,
        """
        import json
        json.dump({"accuracy": 0.9}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "SUCCEEDED"
    assert run.trials == [{"accuracy": 0.9}]
    assert any(p.name == "metrics.json" for p in run.artifacts)


def test_failure_surfaces_as_failed_with_logs(tmp_path):
    import flashruntime as flash

    source = _write_script(tmp_path, "print('boom'); raise SystemExit(3)")
    run = flash.submit(
        flash.CommandWorkload(command=f"{sys.executable} train.py", source={"path": source}),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "FAILED"
    assert "boom" in run.logs()


def test_fanout_runs_each_param_set_and_picks_best(tmp_path):
    import flashruntime as flash

    source = _write_script(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser(); ap.add_argument("--x", type=float)
        args = ap.parse_args()
        json.dump({"x": args.x, "score": args.x * 2}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(
            command=f"{sys.executable} train.py --x {{x}}",
            source={"path": source},
            task_params=[{"x": 1}, {"x": 3}, {"x": 2}],
            outputs=flash.OutputSpec(primary_metric="score"),
        ),
        output_dir=tmp_path / "out",
    )
    assert run.state.value == "SUCCEEDED"
    assert len(run.trials) == 3
    assert run.best_trial()["x"] == 3  # highest score wins
    assert run.best_trial(metric="score", maximize=False)["x"] == 1


def test_stale_outputs_from_previous_trial_not_recollected(tmp_path):
    import flashruntime as flash

    # writes metrics.json only when --x != 2: trial x=2 must NOT inherit x=1's file
    source = _write_script(
        tmp_path,
        """
        import argparse, json
        ap = argparse.ArgumentParser(); ap.add_argument("--x", type=int)
        args = ap.parse_args()
        if args.x != 2:
            json.dump({"x": args.x}, open("metrics.json", "w"))
        """,
    )
    run = flash.submit(
        flash.CommandWorkload(
            command=f"{sys.executable} train.py --x {{x}}",
            source={"path": source},
            task_params=[{"x": 1}, {"x": 2}],
        ),
        output_dir=tmp_path / "out",
    )
    assert [t["x"] for t in run.trials] == [1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sdk_submit.py -v`
Expected: FAIL — `AttributeError: module 'flashruntime' has no attribute 'submit'`

- [ ] **Step 3: Implement**

```python
# flashruntime/sdk.py
"""flash.submit(): run a CommandWorkload locally and hand back a Run.

v1 is synchronous, sequential local execution (mirroring the M0 engine's
replay model): compile → launch → wait → collect, once per Mode A param
set. Sequential-by-design keeps collection correct (a trial's outputs are
copied out of the source dir before the next trial can overwrite them).
Service submission is a different door: `workloads.command.to_jobspec()`
POSTed to the coordinator.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from flashruntime.launchers import LaunchState
from flashruntime.launchers.local import LocalProcessLauncher
from flashruntime.strategies.command import compile_workload
from flashruntime.workloads.command import CommandWorkload

_JOB_ID = "local"  # deterministic: rerunning with the same output_dir resumes checkpoints


class Run:
    """Result handle for one submit(). All fields are populated by the time
    submit() returns (synchronous v1)."""

    def __init__(self, workload: CommandWorkload, output_dir: Path):
        self.workload = workload
        self.output_dir = output_dir
        self.state: LaunchState = LaunchState.PENDING
        self.trials: list[dict] = []
        self.artifacts: list[Path] = []
        self._logs: list[str] = []

    def logs(self, tail_lines: int = 200) -> str:
        return "\n".join("\n".join(self._logs).splitlines()[-tail_lines:])

    def best_trial(self, metric: str | None = None, maximize: bool | None = None) -> dict | None:
        """Highest/lowest `metric` among trials that reported it. Defaults
        come from the workload's OutputSpec (adapters set them)."""
        metric = metric or self.workload.outputs.primary_metric
        if maximize is None:
            maximize = self.workload.outputs.maximize
        if metric is None:
            raise ValueError("no metric named: pass metric= or set outputs.primary_metric")
        scored = [t for t in self.trials if metric in t]
        if not scored:
            return None
        return max(scored, key=lambda t: t[metric]) if maximize else min(scored, key=lambda t: t[metric])


def submit(workload: CommandWorkload, output_dir: str | Path | None = None) -> Run:
    out_root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="flashruntime-run-"))
    run = Run(workload, out_root)
    launcher = LocalProcessLauncher(out_root)
    source_dir = Path(workload.source.path).expanduser()

    fanout = workload.resolved_mode() == "independent_tasks" and workload.task_params
    param_sets: list[dict | None] = list(workload.task_params) if fanout else [None]

    states: list[LaunchState] = []
    for i, params in enumerate(param_sets):
        attempt_id = f"task-{i:03d}"
        spec = compile_workload(workload, params)
        started_at = time.time()
        handle = launcher.launch(spec, _JOB_ID, attempt_id)
        state = handle.wait()
        states.append(state)
        run._logs.append(f"--- {attempt_id} ({state.value}) ---\n{handle.logs()}")

        collected = _collect(source_dir, workload.outputs.collect, handle.output_dir, since=started_at)
        run.artifacts.extend(collected)
        metrics_path = handle.output_dir / "metrics.json"
        if metrics_path.is_file():
            try:
                metrics = json.loads(metrics_path.read_text())
            except ValueError:
                metrics = None
            if isinstance(metrics, dict):
                if params:
                    metrics.setdefault("params", params)
                run.trials.append(metrics)

    run.state = (
        LaunchState.SUCCEEDED
        if states and all(s is LaunchState.SUCCEEDED for s in states)
        else LaunchState.FAILED
    )
    return run


def _collect(source_dir: Path, patterns: list[str], dest: Path, since: float) -> list[Path]:
    """Copy collect-globs from the script's cwd into the attempt's output
    dir. `since` skips files older than this launch — a stale metrics.json
    from a previous trial must never be credited to a failed one."""
    out: list[Path] = []
    for pattern in patterns:
        for src in sorted(source_dir.glob(pattern)):
            if not src.is_file() or src.stat().st_mtime < since:
                continue
            target = dest / src.relative_to(source_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, target)
            out.append(target)
    return out
```

Modify `flashruntime/__init__.py` — three edits:

1. Extend `__all__` (after `"registered_providers",`):

```python
    # bring-your-own-code SDK (lazy — stdlib+pydantic only, but kept lazy
    # so `import flashruntime` stays minimal)
    "submit",
    "CommandWorkload",
    "OutputSpec",
    "Source",
    "integrations",
```

2. Below `_PROTOTYPE_EXPORTS = {...}` add:

```python
# SDK exports resolve lazily too: name -> (module, attribute).
_SDK_EXPORTS = {
    "submit": ("flashruntime.sdk", "submit"),
    "CommandWorkload": ("flashruntime.workloads.command", "CommandWorkload"),
    "OutputSpec": ("flashruntime.workloads.command", "OutputSpec"),
    "Source": ("flashruntime.workloads.command", "Source"),
    "integrations": ("flashruntime.integrations", None),
}
```

3. In `__getattr__`, before the `raise AttributeError(...)` line, add:

```python
    if name in _SDK_EXPORTS:
        import importlib

        module_name, attr = _SDK_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = module if attr is None else getattr(module, attr)
        globals()[name] = value
        return value
```

(`flashruntime.integrations` does not exist until Task 6 — that is fine: the lazy lookup only imports it when accessed, and Task 6 lands before anything accesses it.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sdk_submit.py -v` → all PASS

- [ ] **Step 5: Verify the core-import smoke still holds**

Run: `python -c "import flashruntime; print(flashruntime.PLANNER_VERSION)"`
Expected: prints the version; no torch/numpy import errors.

- [ ] **Step 6: Full suite + commit**

```bash
pytest && git add flashruntime/sdk.py flashruntime/__init__.py tests/test_sdk_submit.py \
  && git commit -m "feat(sdk): flash.submit() runs CommandWorkloads locally"
```

---

### Task 6: Framework adapters (`integrations/`)

**Files:**
- Create: `flashruntime/integrations/__init__.py`, `flashruntime/integrations/sklearn.py`, `flashruntime/integrations/pytorch.py`, `flashruntime/integrations/huggingface.py`
- Test: `tests/test_integrations.py`

**Interfaces:**
- Consumes: `CommandWorkload`, `OutputSpec`, `Source` (Task 1); `CheckpointPolicy` from plan protocol; `write_manifest`/`latest_valid_manifest` (Task 4, inside the HF callback only).
- Produces:
  - `integrations.sklearn.sweep(script, task_params, *, source=".", metric="accuracy_mean", maximize=True, python="python") -> CommandWorkload`
  - `integrations.sklearn.hpo(script, grid: dict[str, list], **kwargs) -> CommandWorkload`
  - `integrations.pytorch.ddp(script, *, source=".", nproc_per_node=2, nnodes=1, script_args="", env=None) -> CommandWorkload` (raises `NotImplementedError` for `nnodes > 1`)
  - `integrations.huggingface.trainer(script, *, source=".", nproc_per_node=1, script_args="") -> CommandWorkload`
  - `integrations.huggingface.flashruntime_callback()` (lazy transformers import) and `integrations.huggingface.latest_checkpoint(output_dir) -> str | None`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_integrations.py
"""Adapters build workloads; they never import their frameworks at module
level (four-axes rule: framework code runs in the user's process only)."""

from __future__ import annotations

import sys

import pytest


def test_importing_adapters_pulls_no_frameworks():
    preloaded = {m for m in ("torch", "sklearn", "transformers") if m in sys.modules}
    import flashruntime.integrations.huggingface  # noqa: F401
    import flashruntime.integrations.pytorch  # noqa: F401
    import flashruntime.integrations.sklearn  # noqa: F401

    for mod in ("torch", "sklearn", "transformers"):
        if mod not in preloaded:
            assert mod not in sys.modules, f"adapter import pulled in {mod}"


def test_sklearn_hpo_builds_cartesian_fanout():
    from flashruntime.integrations import sklearn as fr_sklearn

    wl = fr_sklearn.hpo("train.py", {"model": ["logreg", "rf"], "C": [0.1, 1.0]}, source="/proj")
    assert wl.resolved_mode() == "independent_tasks"
    assert len(wl.task_params) == 4
    # flags for every grid key, sorted, placeholder-form
    assert wl.command == ["python", "train.py", "--C", "{C}", "--model", "{model}"]
    assert wl.outputs.primary_metric == "accuracy_mean"
    assert wl.source.path == "/proj"


def test_pytorch_ddp_builds_torchrun_command():
    from flashruntime.integrations import pytorch as fr_torch

    wl = fr_torch.ddp("train.py", source="/proj", nproc_per_node=4, script_args="--steps 100")
    assert wl.command[:4] == ["torchrun", "--nproc-per-node=4", "--nnodes=1", "--standalone"]
    assert wl.command[4:] == ["train.py", "--steps", "100"]
    assert wl.resolved_mode() == "coordinated"
    assert wl.checkpoint is not None


def test_pytorch_multinode_not_yet():
    from flashruntime.integrations import pytorch as fr_torch

    with pytest.raises(NotImplementedError, match="multi-node"):
        fr_torch.ddp("train.py", nnodes=2)


def test_hf_trainer_is_torchrun_shaped():
    from flashruntime.integrations import huggingface as fr_hf

    wl = fr_hf.trainer("finetune.py", nproc_per_node=2)
    assert wl.command[0] == "torchrun"
    assert wl.resolved_mode() == "coordinated"


def test_hf_latest_checkpoint_reads_manifests(tmp_path):
    from flashruntime.checkpoint.local import write_manifest
    from flashruntime.integrations import huggingface as fr_hf

    d = tmp_path / "checkpoint-40"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    write_manifest(d, job_id="j", attempt_id="a", step=40)
    assert fr_hf.latest_checkpoint(tmp_path) == str(d)
    assert fr_hf.latest_checkpoint(tmp_path / "none") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_integrations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flashruntime.integrations'`

- [ ] **Step 3: Implement**

```python
# flashruntime/integrations/__init__.py
"""Framework adapters: each builds CommandWorkloads from one framework's
LAUNCH AND CHECKPOINT CONVENTIONS — never its model code, never a
module-level framework import (four-axes rule). Import the submodule you
need: `from flashruntime.integrations import sklearn, pytorch, huggingface`.
"""
```

```python
# flashruntime/integrations/sklearn.py
"""sklearn adapter: distribute across runs, never inside .fit().

The contract with the user's script is pure convention: CLI flags in,
metrics.json out. No sklearn import here — the script owns the estimator.
"""

from __future__ import annotations

import itertools

from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source


def sweep(
    script: str,
    task_params: list[dict],
    *,
    source: str = ".",
    metric: str = "accuracy_mean",
    maximize: bool = True,
    python: str = "python",
) -> CommandWorkload:
    """One independent task per params dict. Every dict must carry every
    key (the CLI flags are built from the union)."""
    keys = sorted({k for p in task_params for k in p})
    command = [python, script]
    for key in keys:
        command += [f"--{key}", "{" + key + "}"]
    return CommandWorkload(
        command=command,
        source=Source(path=source),
        task_params=task_params,
        mode="independent_tasks",
        outputs=OutputSpec(collect=["metrics.json"], primary_metric=metric, maximize=maximize),
    )


def hpo(script: str, grid: dict[str, list], **kwargs) -> CommandWorkload:
    """Cartesian grid search: {"model": ["logreg","rf"], "C": [0.1, 1]} → 4 trials."""
    keys = sorted(grid)
    trials = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    return sweep(script, trials, **kwargs)
```

```python
# flashruntime/integrations/pytorch.py
"""PyTorch adapter: launch conventions only. torchrun starts N processes
and hands each RANK/WORLD_SIZE — the user's code (or
flashruntime.torch.prepare) wires DDP from there. No torch import here.
"""

from __future__ import annotations

import shlex

from flashruntime.protocol.plan_v1alpha1 import CheckpointPolicy
from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source


def ddp(
    script: str,
    *,
    source: str = ".",
    nproc_per_node: int = 2,
    nnodes: int = 1,
    script_args: str = "",
    env: dict[str, str] | None = None,
) -> CommandWorkload:
    if nnodes > 1:
        raise NotImplementedError(
            "multi-node rendezvous is a launcher concern — later slice (spec §10); "
            "--standalone below is single-node by definition"
        )
    command = [
        "torchrun",
        f"--nproc-per-node={nproc_per_node}",
        f"--nnodes={nnodes}",
        "--standalone",
        script,
        *shlex.split(script_args),
    ]
    return CommandWorkload(
        command=command,
        source=Source(path=source),
        env=env or {},
        mode="coordinated",
        checkpoint=CheckpointPolicy(
            backend="local_manifest",
            note="flashruntime.torch.checkpoint: parts-first/manifest-last under FLASHML_CKPT_DIR",
        ),
        outputs=OutputSpec(collect=["metrics.json"]),
    )
```

```python
# flashruntime/integrations/huggingface.py
"""Hugging Face adapter: HF Trainer already wraps DDP/FSDP internally when
launched by torchrun, so launching is the pytorch adapter's job. What HF
adds is its callback seam — `flashruntime_callback()` commits Trainer
checkpoints as verified manifests. transformers is imported only inside
that factory, in the user's training process.
"""

from __future__ import annotations

import os
from pathlib import Path

from flashruntime.checkpoint.local import latest_valid_manifest
from flashruntime.integrations.pytorch import ddp
from flashruntime.workloads.command import CommandWorkload


def trainer(
    script: str, *, source: str = ".", nproc_per_node: int = 1, script_args: str = ""
) -> CommandWorkload:
    return ddp(script, source=source, nproc_per_node=nproc_per_node, script_args=script_args)


def latest_checkpoint(output_dir: str | Path) -> str | None:
    """Newest Trainer checkpoint dir with a VALID manifest — pass as
    `trainer.train(resume_from_checkpoint=...)`. None means fresh start."""
    manifest = latest_valid_manifest(Path(output_dir), pattern="checkpoint-*")
    return None if manifest is None else manifest.storage_prefix


def flashruntime_callback():
    """Build the TrainerCallback (transformers import paid here, in the
    user's process only): on_save commits a manifest, on_log relays metrics."""
    from transformers import TrainerCallback  # noqa: PLC0415 — user process only

    from flashruntime.checkpoint.local import write_manifest

    class FlashRuntimeCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                step_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if step_dir.is_dir():
                    write_manifest(
                        step_dir,
                        job_id=os.environ.get("FLASHML_JOB_ID", "local"),
                        attempt_id=os.environ.get("FLASHML_ATTEMPT_ID", "local"),
                        step=state.global_step,
                        framework="transformers",
                    )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                from flashruntime.torch import log_metrics

                log_metrics({**logs, "step": state.global_step})
            return control

    return FlashRuntimeCallback()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_integrations.py -v` → all PASS

- [ ] **Step 5: Full suite + commit**

```bash
pytest && git add flashruntime/integrations tests/test_integrations.py \
  && git commit -m "feat(integrations): sklearn/pytorch/huggingface adapters (launch conventions only)"
```

---

### Task 7: `flashruntime.torch` in-script helper

**Files:**
- Create: `flashruntime/torch/__init__.py`
- Test: `tests/test_torch_helper.py`

**Interfaces:**
- Consumes: `write_manifest`, `latest_valid_manifest` (Task 4). Env contract from Task 3 (`FLASHML_OUTPUT_DIR`, `FLASHML_CKPT_DIR`, `FLASHML_JOB_ID`, `FLASHML_ATTEMPT_ID`) plus torchrun's `RANK`/`WORLD_SIZE`/`LOCAL_RANK`.
- Produces (complete surface — the guardrail): `prepare(model, optimizer=None, dataloader=None)`, `checkpoint(model, optimizer=None, *, step, every=None)`, `log_metrics(dict)`, `rank()`, `world_size()`, `is_main()`, `start_step()`. Test hook: module global `_restored_step` (reset in tests).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_torch_helper.py
"""flashruntime.torch: single-process behavior + checkpoint/resume. The
2-process gloo path is exercised end-to-end in test_examples_e2e.py."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")


@pytest.fixture()
def ft(monkeypatch, tmp_path):
    import flashruntime.torch as ft_mod

    monkeypatch.setenv("FLASHML_OUTPUT_DIR", str(tmp_path / "out"))
    monkeypatch.setenv("FLASHML_CKPT_DIR", str(tmp_path / "ckpt"))
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    monkeypatch.delenv("RANK", raising=False)
    monkeypatch.setattr(ft_mod, "_restored_step", 0)
    return ft_mod


def _model():
    torch.manual_seed(0)
    return torch.nn.Linear(4, 2)


def test_prepare_is_noop_single_process(ft):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loader = object()  # must be passed through untouched
    m2, o2, l2 = ft.prepare(model, opt, loader)
    assert m2 is model and o2 is opt and l2 is loader
    assert ft.world_size() == 1 and ft.rank() == 0 and ft.is_main()
    assert ft.start_step() == 0


def test_checkpoint_every_gating(ft, tmp_path):
    model = _model()
    ft.checkpoint(model, step=7, every=5)
    assert not list((tmp_path / "ckpt").glob("step-*"))
    ft.checkpoint(model, step=10, every=5)
    assert (tmp_path / "ckpt" / "step-000010" / "manifest.json").is_file()


def test_checkpoint_then_resume_restores_weights_and_step(ft):
    model = _model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    with torch.no_grad():
        model.weight.fill_(3.14)
    ft.checkpoint(model, opt, step=5)

    import flashruntime.torch as ft_mod

    ft_mod._restored_step = 0
    fresh = torch.nn.Linear(4, 2)
    fresh_opt = torch.optim.SGD(fresh.parameters(), lr=0.1)
    m2, _, _ = ft.prepare(fresh, fresh_opt, None)
    assert float(m2.weight[0, 0]) == pytest.approx(3.14)
    assert ft.start_step() == 5


def test_corrupted_checkpoint_is_never_restored(ft, tmp_path):
    model = _model()
    ft.checkpoint(model, step=5)
    ft.checkpoint(model, step=10)
    (tmp_path / "ckpt" / "step-000010" / "model.pt").write_bytes(b"garbage")

    import flashruntime.torch as ft_mod

    ft_mod._restored_step = 0
    ft.prepare(torch.nn.Linear(4, 2), None, None)
    assert ft.start_step() == 5  # fell back to the older VALID manifest


def test_log_metrics_appends_jsonl_and_never_raises(ft, tmp_path, monkeypatch):
    ft.log_metrics({"loss": 1.0})
    ft.log_metrics({"loss": 0.5})
    lines = (tmp_path / "out" / "metrics.jsonl").read_text().splitlines()
    assert [json.loads(l)["loss"] for l in lines] == [1.0, 0.5]
    # unwritable target must not kill training
    monkeypatch.setenv("FLASHML_OUTPUT_DIR", "/dev/null/nope")
    ft.log_metrics({"loss": 0.1})  # must not raise
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_torch_helper.py -v`
Expected: if torch is installed — FAIL with `ModuleNotFoundError: No module named 'flashruntime.torch'`; if torch is absent — all SKIP (then install the CPU wheel to develop this task: `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`).

- [ ] **Step 3: Implement**

```python
# flashruntime/torch/__init__.py
"""In-training-script helper: one import makes a PyTorch script
launch-anywhere and fault-tolerant.

    import flashruntime.torch as ft
    model, opt, loader = ft.prepare(model, opt, loader)
    ...
    ft.checkpoint(model, opt, step=step, every=100)
    ft.log_metrics({"loss": float(loss)})

Launched by torchrun (WORLD_SIZE>1): prepare() wires torch's OWN DDP +
DistributedSampler and restores the newest VALID checkpoint manifest.
Launched as plain `python train.py`: prepare() is a no-op passthrough.

GUARDRAIL (ADR-0003 — do not rebuild Accelerate): the complete surface is
prepare / checkpoint / log_metrics / rank / world_size / is_main /
start_step. No FSDP policies, no autocast, no DeepSpeed config — users
wanting those use the real framework features, which the launcher still
launches correctly.

torch is imported inside functions only: flashruntime's core never
depends on it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

__all__ = [
    "prepare",
    "checkpoint",
    "log_metrics",
    "rank",
    "world_size",
    "is_main",
    "start_step",
]

_restored_step = 0


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def start_step() -> int:
    """First step the training loop should run: 0 fresh, >0 after a
    resume (set by prepare() when it restores a checkpoint)."""
    return _restored_step


def _output_dir() -> Path:
    return Path(os.environ.get("FLASHML_OUTPUT_DIR", "."))


def _ckpt_root() -> Path:
    # job-scoped (NOT attempt-scoped): a restarted attempt must find its
    # predecessor's manifests — the launcher exports FLASHML_CKPT_DIR
    root = os.environ.get("FLASHML_CKPT_DIR")
    return Path(root) if root else _output_dir() / "ckpt"


def prepare(model, optimizer=None, dataloader=None):
    """Wire distributed execution (when launched distributed) and restore
    the newest valid checkpoint (when one exists). Returns the possibly
    wrapped/rebuilt (model, optimizer, dataloader) triple."""
    global _restored_step
    import torch

    if world_size() > 1:
        import torch.distributed as dist

        if not dist.is_initialized():
            backend = "nccl" if torch.cuda.is_available() else "gloo"
            dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        model = torch.nn.parallel.DistributedDataParallel(model)
        if dataloader is not None:
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler

            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=DistributedSampler(dataloader.dataset),
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers,
                drop_last=dataloader.drop_last,
            )

    from flashruntime.checkpoint.local import latest_valid_manifest

    manifest = latest_valid_manifest(_ckpt_root())
    if manifest is not None:
        step_dir = Path(manifest.storage_prefix)
        target = model.module if hasattr(model, "module") else model
        target.load_state_dict(torch.load(step_dir / "model.pt", map_location="cpu"))
        if optimizer is not None and (step_dir / "optimizer.pt").is_file():
            optimizer.load_state_dict(torch.load(step_dir / "optimizer.pt", map_location="cpu"))
        _restored_step = manifest.step

    return model, optimizer, dataloader


def checkpoint(model, optimizer=None, *, step: int, every: int | None = None) -> None:
    """Write a resumable checkpoint under the manifest contract (parts
    first, manifest last). rank 0 writes; every rank synchronizes on the
    barrier so no one races past a half-written checkpoint."""
    if every is not None and (step == 0 or step % every != 0):
        return
    import torch

    if is_main():
        from flashruntime.checkpoint.local import write_manifest

        step_dir = _ckpt_root() / f"step-{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        target = model.module if hasattr(model, "module") else model
        torch.save(target.state_dict(), step_dir / "model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
        write_manifest(
            step_dir,
            job_id=os.environ.get("FLASHML_JOB_ID", "local"),
            attempt_id=os.environ.get("FLASHML_ATTEMPT_ID", "local"),
            step=step,
            world_size=world_size(),
            framework=f"pytorch-{torch.__version__.split('+')[0]}",
        )
    if world_size() > 1:
        import torch.distributed as dist

        dist.barrier()


def log_metrics(metrics: dict) -> None:
    """Append one JSON record to FLASHML_OUTPUT_DIR/metrics.jsonl (rank 0
    only). Never raises — metrics must never kill training."""
    if not is_main():
        return
    try:
        path = _output_dir() / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception:  # noqa: BLE001 — by contract, swallow everything
        pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_torch_helper.py -v` → all PASS (or all SKIP without torch — then they must PASS on a torch-equipped venv before commit)

- [ ] **Step 5: Verify the core stays torch-free**

Run: `python -c "import flashruntime, flashruntime.integrations.pytorch, sys; assert 'torch' not in sys.modules; print('core clean')"`
Expected: `core clean`

- [ ] **Step 6: Full suite + commit**

```bash
pytest && git add flashruntime/torch tests/test_torch_helper.py \
  && git commit -m "feat(torch): in-script helper — prepare/checkpoint/log_metrics"
```

---

### Task 8: Isolation-aware placement (fail-closed)

**Files:**
- Modify: `flashruntime/scheduler/__init__.py` (append class), `flashruntime/leases/manager.py` (`claim()` signature)
- Test: `tests/test_scheduler_isolation.py`

**Interfaces:**
- Consumes: `PlacementPolicy`, `NodeView` (existing, `scheduler/__init__.py`); `TaskSpec`.
- Produces: `scheduler.IsolationAwarePlacement` (a `PlacementPolicy`); `LeaseManager.claim(node_id, job_id=None, now=None, policy=None, node=None)` — with `policy=None` behavior is bit-identical to today (existing tests prove it). The policy object is duck-typed (`choose(pending: list[TaskSpec], node: dict) -> TaskSpec | None`) so `leases/` gains no import on `scheduler/`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scheduler_isolation.py
"""The one fail-closed placement rule the isolation contract requires:
sandboxed tasks go only to sandbox_capable nodes (AGENTS.md rule 3)."""

from __future__ import annotations


def _task(task_id: str, tier: str | None = None, allow_fallback: bool = False):
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {}
    if tier is not None:
        payload["isolation"] = {"tier": tier, "allowFallback": allow_fallback}
    return TaskSpec(task_id=task_id, job_id="j1", commit_key=f"j1/{task_id}", payload=payload)


def test_eligibility_matrix():
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    capable = {"node_id": "n1", "sandbox_capable": True}
    incapable = {"node_id": "n2", "sandbox_capable": False}
    unknown = {"node_id": "n3"}  # missing key ⇒ NOT capable (fail closed)

    assert policy.eligible(_task("t", tier="standard"), incapable)
    assert policy.eligible(_task("t"), incapable)  # no isolation payload ⇒ standard
    assert policy.eligible(_task("t", tier="sandboxed"), capable)
    assert not policy.eligible(_task("t", tier="sandboxed"), incapable)
    assert not policy.eligible(_task("t", tier="sandboxed"), unknown)
    assert policy.eligible(_task("t", tier="sandboxed", allow_fallback=True), incapable)


def test_claim_with_policy_fails_closed_and_preserves_fifo():
    from flashruntime.leases import LeaseManager
    from flashruntime.scheduler import IsolationAwarePlacement

    mgr = LeaseManager()
    mgr.add_task(_task("t-sandboxed", tier="sandboxed"))
    mgr.add_task(_task("t-standard"))
    policy = IsolationAwarePlacement()

    # incapable node: must skip the sandboxed head-of-queue and get the standard task
    lease = mgr.claim("n2", policy=policy, node={"node_id": "n2", "sandbox_capable": False})
    assert lease.task_id == "t-standard"

    # capable node: gets the sandboxed task
    lease2 = mgr.claim("n1", policy=policy, node={"node_id": "n1", "sandbox_capable": True})
    assert lease2.task_id == "t-sandboxed"

    # nothing left ⇒ None, and the sandboxed task was never mis-leased
    assert mgr.claim("n2", policy=policy, node={"node_id": "n2"}) is None


def test_claim_without_policy_is_unchanged():
    from flashruntime.leases import LeaseManager

    mgr = LeaseManager()
    mgr.add_task(_task("first"))
    mgr.add_task(_task("second"))
    assert mgr.claim("n1").task_id == "first"  # FIFO, exactly as before
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scheduler_isolation.py -v`
Expected: FAIL — `ImportError: cannot import name 'IsolationAwarePlacement'`

- [ ] **Step 3: Implement**

Append to `flashruntime/scheduler/__init__.py` (after `FifoPlacement`; add `"IsolationAwarePlacement"` to `__all__`):

```python
class IsolationAwarePlacement(PlacementPolicy):
    """FIFO plus the one fail-closed capability gate the isolation contract
    requires: a task whose payload demands `sandboxed` execution may only
    go to a node advertising `sandbox_capable` — an ABSENT capability
    counts as NOT capable (security-relevant fields fail closed, AGENTS.md
    rule 3). The task's own `allowFallback: true` waives the requirement
    explicitly. Everything else keeps the fail-open placement default."""

    def eligible(self, task: TaskSpec, node: NodeView) -> bool:
        isolation = task.payload.get("isolation") or {}
        if isolation.get("tier") != "sandboxed":
            return True
        if isolation.get("allowFallback"):
            return True
        return bool(node.get("sandbox_capable"))
```

In `flashruntime/leases/manager.py`, replace the `claim` signature and the `record = self._store.next_pending(job_id)` line:

```python
    def claim(
        self,
        node_id: str,
        job_id: str | None = None,
        now: datetime | None = None,
        policy: object | None = None,
        node: dict | None = None,
    ) -> Lease | None:
        """Claim the next PENDING task for `node_id`, or None when nothing is
        claimable. Expired leases are swept first so a claim never starves
        behind a dead worker.

        `policy`/`node` are the scheduler seam (flashruntime/scheduler):
        the store yields queue-ordered candidates, the policy filters and
        picks. Duck-typed (`choose(pending_specs, node) -> TaskSpec|None`)
        so this package gains no scheduler import. Without a policy,
        behavior is bit-identical to the original FIFO claim."""
        now = now or _utcnow()
        self.sweep(now=now)
        if policy is None:
            record = self._store.next_pending(job_id)
        else:
            pending = [r for r in self._store.all(job_id) if r.state == TaskState.PENDING]
            chosen = policy.choose([r.spec for r in pending], node or {"node_id": node_id})
            record = None
            if chosen is not None:
                record = next(r for r in pending if r.spec.task_id == chosen.task_id)
        if record is None:
            return None
```

(The rest of the method body is unchanged.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scheduler_isolation.py tests/test_leases.py tests/test_leases_sqlite.py -v`
Expected: all PASS (existing lease tests prove the no-policy path is untouched)

- [ ] **Step 5: Full suite + commit**

```bash
pytest && git add flashruntime/scheduler/__init__.py flashruntime/leases/manager.py \
  tests/test_scheduler_isolation.py \
  && git commit -m "feat(scheduler): fail-closed isolation-aware placement, wired into claim"
```

---

### Task 9: `CommandRecipe` + service wiring

**Files:**
- Create: `flashruntime/recipes/command.py`
- Modify: `flashruntime/service/modea.py` (recipe dispatch in `expand_tasks`; isolation-aware claim endpoint)
- Test: `tests/test_service_command_recipe.py`

**Interfaces:**
- Consumes: `WorkloadRecipe`, `register_recipe`, `recipe_for` (existing `recipes/__init__.py`); `to_jobspec` (Task 1); `IsolationAwarePlacement` (Task 8).
- Produces: workload type `"command"` accepted by `modea.expand_tasks` / `POST /v1alpha1/jobs`. Task payloads carry: `argv`, `env`, `inputs`, `output_prefix`, `task_id`, `image`, `isolation{tier,allowFallback}`, optional `checkpoint`. The claim endpoint passes `IsolationAwarePlacement()` + a node view including `sandbox_capable`. **Cross-repo flag:** flashnode's executor must learn `argv` payloads before service-side commands actually run — this task delivers the coordinator half only.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_service_command_recipe.py
"""CommandRecipe: JobSpec{type: command} → lease tasks, isolation stamped,
dispatched through the recipe registry by expand_tasks."""

from __future__ import annotations

import pytest


def _jobspec(**over):
    from flashruntime.protocol.v1alpha1 import ImageSpec, IsolationSpec
    from flashruntime.workloads.command import CommandWorkload, to_jobspec

    defaults = dict(command="python train.py --lr {lr}", task_params=[{"lr": 0.1}, {"lr": 0.01}])
    defaults.update(over)
    wl = CommandWorkload(**defaults)
    return to_jobspec(wl, name="cmd-job", image=ImageSpec(repository="ghcr.io/me/img", tag="1.0"))


def test_expand_substitutes_params_and_stamps_isolation():
    from flashruntime.recipes.command import CommandRecipe
    from flashruntime.protocol.v1alpha1 import IsolationSpec

    spec = _jobspec(isolation=IsolationSpec(tier="sandboxed", allowFallback=False))
    tasks = CommandRecipe().expand("job1", spec)
    assert [t.task_id for t in tasks] == ["task-000", "task-001"]
    assert tasks[0].payload["argv"] == ["python", "train.py", "--lr", "0.1"]
    assert tasks[1].payload["argv"] == ["python", "train.py", "--lr", "0.01"]
    assert tasks[0].payload["isolation"] == {"tier": "sandboxed", "allowFallback": False}
    assert tasks[0].payload["image"] == "ghcr.io/me/img:1.0"
    assert tasks[0].commit_key == "jobs/job1/task-000/metrics.json"
    assert tasks[0].max_attempts == spec.spec.retryPolicy.maxTaskAttempts


def test_expand_single_task_when_no_fanout():
    from flashruntime.recipes.command import CommandRecipe

    tasks = CommandRecipe().expand("job1", _jobspec(command="python eval.py", task_params=None))
    assert len(tasks) == 1
    assert tasks[0].payload["argv"] == ["python", "eval.py"]


def test_expand_rejects_bad_params():
    from flashruntime.recipes.command import CommandRecipe

    spec = _jobspec()
    spec.spec.workload.parameters["command"] = "not-a-list"
    with pytest.raises(ValueError, match="argv"):
        CommandRecipe().expand("job1", spec)

    spec2 = _jobspec()
    spec2.spec.workload.parameters["task_params"] = [{"seed": 1}]  # {lr} unfilled
    with pytest.raises(ValueError, match="placeholder"):
        CommandRecipe().expand("job1", spec2)


def test_expand_tasks_dispatches_command_type_via_registry():
    from flashruntime.service import modea

    tasks = modea.expand_tasks("job1", _jobspec())
    assert len(tasks) == 2
    assert tasks[0].payload["argv"][0] == "python"


def test_legacy_expansions_still_work():
    from flashruntime.protocol.v1alpha1 import (
        ExecutionSpec, ImageSpec, JobMetadata, JobSpec, JobSpecInner, WorkloadSpec,
    )
    from flashruntime.service import modea

    spec = JobSpec(
        metadata=JobMetadata(name="sweep"),
        spec=JobSpecInner(
            execution=ExecutionSpec(backend="leases"),
            image=ImageSpec(repository="r", tag="1"),
            workload=WorkloadSpec(
                type="hyperparameter_search",
                parameters={"trials": [{"model": "logreg", "C": 0.1}]},
            ),
        ),
    )
    tasks = modea.expand_tasks("job2", spec)
    assert tasks[0].payload["module"] == "flashml_workloads.sklearn_trial"


def test_claim_endpoint_fails_closed_for_sandboxed_tasks():
    """Full HTTP path: a sandboxed command task is never leased to a
    non-sandbox node. Mirrors tests/test_service_modea.py conventions."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import IsolationSpec
    from flashruntime.service.modea import ModeAState, build_router, expand_tasks

    state = ModeAState(LeaseManager(), artifacts_dir=__import__("pathlib").Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)

    for task in expand_tasks("job1", _jobspec(isolation=IsolationSpec(tier="sandboxed"))):
        state.manager.add_task(task)

    def register(node_id: str, sandbox: bool):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={
                "node_id": node_id,
                "kubernetes_node": "",
                "hostname": node_id,
                "capabilities": {},
                "sandbox_capable": sandbox,
            },
        )
        assert r.status_code == 200

    register("plain-node", sandbox=False)
    register("sandbox-node", sandbox=True)

    # fail closed: plain node gets nothing
    assert client.post("/v1alpha1/leases/claim", json={"node_id": "plain-node"}).status_code == 204
    # sandbox-capable node gets the task
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "sandbox-node"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_service_command_recipe.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'flashruntime.recipes.command'`

- [ ] **Step 3: Implement the recipe**

```python
# flashruntime/recipes/command.py
"""The generic command recipe: JobSpec{workload.type: "command"} → lease tasks.

The first concrete WorkloadRecipe. Payloads carry `argv` — the §2.2
executor contract generalized from `module` — plus the isolation
requirement the placement gate enforces fail-closed. Executing argv
payloads is flashnode's runner tier (cross-repo, versioned change); this
recipe defines the coordinator half of that contract. Until flashnode
ships it, command jobs expand and lease correctly but only argv-aware
executors can run them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from flashruntime.protocol.v1alpha1 import JobSpec, TaskSpec
from flashruntime.recipes import WorkloadRecipe, register_recipe


class CommandRecipe(WorkloadRecipe):
    kind: ClassVar[str] = "command"
    #: argv payloads name no task module — the isolation tier, not a module
    #: allowlist, is the security control for this workload type.
    task_module: ClassVar[str] = ""

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        command = params.get("command")
        if (
            not command
            or not isinstance(command, list)
            or not all(isinstance(t, str) for t in command)
        ):
            problems.append("'command' must be a non-empty argv list of strings")
        for name, uri in (params.get("inputs") or {}).items():
            if not str(uri).startswith("artifact://"):
                problems.append(f"input '{name}' must be an artifact:// URI")
        task_params = params.get("task_params")
        if task_params is not None and (
            not isinstance(task_params, list)
            or not all(isinstance(p, dict) for p in task_params)
        ):
            problems.append("'task_params' must be a list of objects")
        return problems

    def expand(self, job_id: str, spec: JobSpec) -> list[TaskSpec]:
        p = spec.spec.workload.parameters
        problems = self.validate_params(p)
        if problems:
            raise ValueError("; ".join(problems))

        param_sets: list[dict | None] = p.get("task_params") or [None]
        env: dict[str, str] = dict(p.get("env") or {})
        inputs = dict(p.get("inputs") or {})
        isolation = {
            "tier": spec.spec.isolation.tier,
            "allowFallback": spec.spec.isolation.allowFallback,
        }

        tasks: list[TaskSpec] = []
        for i, params in enumerate(param_sets):
            task_id = f"task-{i:03d}"
            try:
                argv = [t.format(**params) for t in p["command"]] if params else list(p["command"])
                task_env = {
                    k: (v.format(**params) if params else v) for k, v in env.items()
                }
            except KeyError as exc:
                raise ValueError(
                    f"task {i}: placeholder {exc} has no value in task_params[{i}]"
                ) from None
            payload: dict[str, Any] = {
                "argv": argv,
                "env": task_env,
                "inputs": inputs,
                "output_prefix": f"jobs/{job_id}/{task_id}/",
                "task_id": task_id,
                "image": spec.spec.image.reference,
                "isolation": isolation,
            }
            if p.get("checkpoint") is not None:
                payload["checkpoint"] = p["checkpoint"]
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    job_id=job_id,
                    commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                    max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                    lease_seconds=float(p.get("lease_seconds", 60.0)),
                    payload=payload,
                )
            )
        return tasks

    def validate_output(self, metrics: dict[str, Any]) -> None:
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json must contain a JSON object")


register_recipe(CommandRecipe())
```

- [ ] **Step 4: Wire the service (`flashruntime/service/modea.py`)**

Add imports (with the existing imports):

```python
import flashruntime.recipes.command  # noqa: F401 — registers the "command" recipe
from flashruntime.recipes import recipe_for
from flashruntime.scheduler import IsolationAwarePlacement
```

At the top of `expand_tasks`, before the `if workload.type == "sharded_kmeans":` branch:

```python
    workload = spec.spec.workload
    try:
        recipe = recipe_for(workload.type)
    except LookupError:
        recipe = None
    if recipe is not None:
        try:
            return recipe.expand(job_id, spec)
        except ValueError as exc:
            raise ExpansionError(str(exc)) from None
```

(Delete the now-duplicated `workload = spec.spec.workload` line that follows.)

Replace the claim endpoint body in `build_router`:

```python
    @router.post("/leases/claim")
    async def claim(req: ClaimRequest):
        entry = state.nodes.get(req.node_id)
        if entry is None:
            raise HTTPException(status_code=403, detail="unregistered node — register first")
        node_view = {
            "node_id": req.node_id,
            "sandbox_capable": entry.registration.sandbox_capable,
            "capabilities": entry.registration.capabilities.model_dump(),
        }
        lease = manager.claim(
            req.node_id,
            job_id=req.job_id,
            policy=IsolationAwarePlacement(),
            node=node_view,
        )
        if lease is None:
            return Response(status_code=204)  # nothing claimable right now
        return lease
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_service_command_recipe.py tests/test_service_modea.py -v`
Expected: all PASS (existing Mode A service tests must stay green — the old claim behavior for module tasks is unchanged because module payloads carry no `isolation` key ⇒ eligible everywhere)

- [ ] **Step 6: Full suite + commit**

```bash
pytest && git add flashruntime/recipes/command.py flashruntime/service/modea.py \
  tests/test_service_command_recipe.py \
  && git commit -m "feat(service): command workload type — recipe expansion + fail-closed isolation claim"
```

---

### Task 10: Example user repos + end-to-end demo

**Files:**
- Create: `examples/user_sklearn/train.py`, `examples/user_pytorch/train.py`, `examples/user_pytorch_vanilla/train.py`, `examples/bring_your_code_demo.py`
- Test: `tests/test_examples_e2e.py`

**Interfaces:**
- Consumes: everything above. `examples/user_pytorch/train.py` CLI: `--steps INT --lr FLOAT --checkpoint-every INT --kill-at-step INT|absent`; writes `metrics.json` with keys `steps`, `resumed_from`, `final_loss`.
- Produces: the acceptance evidence — sklearn sweep end-to-end, 2-process CPU DDP end-to-end, kill-and-resume end-to-end.

- [ ] **Step 1: Write the example scripts**

```python
# examples/user_sklearn/train.py
"""An ordinary sklearn script — NO flashruntime import anywhere.

FlashRuntime's whole contract with this file is convention:
CLI flags in, metrics.json out.

    python train.py --model rf --n_estimators 100        # by hand
    examples/bring_your_code_demo.py                     # operated sweep
"""
import argparse
import json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="logreg")
    parser.add_argument("--C", type=float, default=1.0)
    parser.add_argument("--n_estimators", type=int, default=50)
    args = parser.parse_args()

    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score

    X, y = make_classification(n_samples=600, n_features=12, random_state=0)
    if args.model == "logreg":
        estimator = LogisticRegression(C=args.C, max_iter=500, random_state=0)
    elif args.model == "rf":
        estimator = RandomForestClassifier(n_estimators=args.n_estimators, random_state=0)
    else:
        raise SystemExit(f"unknown model {args.model!r} (logreg|rf)")

    scores = cross_val_score(estimator, X, y, cv=3)
    metrics = {
        "model": args.model,
        "C": args.C,
        "n_estimators": args.n_estimators,
        "accuracy_mean": round(float(scores.mean()), 4),
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(metrics)


if __name__ == "__main__":
    main()
```

```python
# examples/user_pytorch/train.py
"""A PyTorch script whose ONLY FlashRuntime coupling is flashruntime.torch.

The model, loss, data, and loop are ordinary PyTorch. The same file runs
three ways:

    python train.py --steps 200                          # single process
    torchrun --nproc-per-node=2 --standalone train.py    # DDP by hand
    flash.submit(integrations.pytorch.ddp(...))          # operated by FlashRuntime

Deterministic on CPU (fixed seeds, no shuffle) so a killed-and-resumed run
reproduces the uninterrupted result — recovery must not change the math.
"""
import argparse
import json

import torch
from torch.utils.data import DataLoader, TensorDataset

import flashruntime.torch as ft


def make_data(n: int = 512, d: int = 16, seed: int = 0) -> TensorDataset:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = ((x @ w).squeeze(1) > 0).long()
    return TensorDataset(x, y)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--checkpoint-every", type=int, default=50)
    parser.add_argument(
        "--kill-at-step",
        type=int,
        default=None,
        help="simulate a crash (fresh runs only; resumed retries finish)",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    model = torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 2)
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loader = DataLoader(make_data(), batch_size=32, shuffle=False)

    model, optimizer, loader = ft.prepare(model, optimizer, loader)
    start = ft.start_step()

    step = start
    loss = torch.tensor(0.0)
    while step < args.steps:
        for x, y in loader:
            if step >= args.steps:
                break
            loss = torch.nn.functional.cross_entropy(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            ft.checkpoint(model, optimizer, step=step, every=args.checkpoint_every)
            ft.log_metrics({"step": step, "loss": round(loss.item(), 6)})
            if args.kill_at_step and start == 0 and step >= args.kill_at_step:
                raise SystemExit(3)  # fresh run only — the retry resumes past this

    ft.checkpoint(model, optimizer, step=step)  # final checkpoint
    if ft.is_main():
        metrics = {
            "steps": step,
            "resumed_from": start,
            "final_loss": round(loss.item(), 6),
        }
        with open("metrics.json", "w") as f:
            json.dump(metrics, f)
        print(metrics)


if __name__ == "__main__":
    main()
```

```python
# examples/user_pytorch_vanilla/train.py
"""Plain torch DDP — NO flashruntime import. Proves the launcher operates
unmodified code: the helper in ../user_pytorch is optional sugar."""
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        dist.init_process_group(backend="gloo")

    torch.manual_seed(0)
    model = torch.nn.Linear(16, 2)
    if world_size > 1:
        model = DistributedDataParallel(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05)

    g = torch.Generator().manual_seed(0)
    x, y = torch.randn(256, 16, generator=g), torch.randint(0, 2, (256,), generator=g)
    loss = torch.tensor(0.0)
    for _ in range(100):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        with open("metrics.json", "w") as f:
            json.dump({"final_loss": round(loss.item(), 6)}, f)
        print("final loss", loss.item())
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

```python
# examples/bring_your_code_demo.py
"""End-to-end demo: FlashRuntime operating YOUR code.

    python examples/bring_your_code_demo.py

Runs (1) an sklearn hyperparameter sweep, and — when torch+torchrun are
installed — (2) a 2-process CPU DDP training run, then (3) the
kill-and-resume story: crash mid-training, resubmit, watch it resume from
the last valid checkpoint manifest.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch
from flashruntime.integrations import sklearn as fr_sklearn

EXAMPLES = Path(__file__).parent


def sklearn_sweep() -> None:
    print("=== 1. sklearn hyperparameter sweep (Mode A shape, local) ===")
    run = flash.submit(
        fr_sklearn.hpo(
            "train.py",
            {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50]},
            source=str(EXAMPLES / "user_sklearn"),
        )
    )
    print(f"state={run.state.value}  trials={len(run.trials)}")
    print("best:", run.best_trial())


def pytorch_ddp() -> None:
    print("\n=== 2. PyTorch DDP, 2 processes on CPU (gloo) ===")
    run = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args="--steps 120 --checkpoint-every 40",
        )
    )
    print(f"state={run.state.value}  metrics={run.trials}")

    print("\n=== 3. kill at step 60, then resume from the last valid checkpoint ===")
    workdir = Path(tempfile.mkdtemp(prefix="flashruntime-demo-"))
    crash = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args="--steps 120 --checkpoint-every 40 --kill-at-step 60",
        ),
        output_dir=workdir,
    )
    print(f"crashed run: state={crash.state.value} (expected FAILED)")
    resume = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args="--steps 120 --checkpoint-every 40 --kill-at-step 60",
        ),
        output_dir=workdir,  # same dir ⇒ same job ⇒ same checkpoint tree
    )
    print(f"resumed run: state={resume.state.value}  metrics={resume.trials}")
    if resume.trials:
        print(f"resumed_from step {resume.trials[0].get('resumed_from')} — recovery, not a restart")


def main() -> None:
    sklearn_sweep()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n(torch not installed — skipping the PyTorch demos: pip install torch)")
        return
    if shutil.which("torchrun") is None:
        print("\n(torchrun not on PATH — skipping the DDP demos)")
        return
    pytorch_ddp()


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write the e2e tests**

```python
# tests/test_examples_e2e.py
"""Acceptance tests (spec §9): real user code, operated end to end.
Auto-skip per missing dependency; they run on any dev laptop with the
extras installed."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_sklearn_sweep_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    import flashruntime as flash
    from flashruntime.integrations import sklearn as fr_sklearn

    run = flash.submit(
        fr_sklearn.hpo(
            "train.py",
            {"model": ["logreg", "rf"], "C": [0.1], "n_estimators": [30]},
            source=str(EXAMPLES / "user_sklearn"),
        ),
        output_dir=tmp_path,
    )
    assert run.state.value == "SUCCEEDED"
    assert len(run.trials) == 2
    best = run.best_trial()
    assert 0.5 < best["accuracy_mean"] <= 1.0


@pytest.mark.parametrize("example", ["user_pytorch", "user_pytorch_vanilla"])
def test_ddp_two_processes_on_cpu(tmp_path, example):
    pytest.importorskip("torch")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    import flashruntime as flash
    from flashruntime.integrations import pytorch as fr_torch

    args = "--steps 60 --checkpoint-every 20" if example == "user_pytorch" else ""
    run = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / example),
            nproc_per_node=2,
            script_args=args,
        ),
        output_dir=tmp_path,
    )
    assert run.state.value == "SUCCEEDED", run.logs()
    assert run.trials, run.logs()


def test_kill_and_resume_reproduces_uninterrupted_result(tmp_path):
    """Spec §9 criterion 3b: crash mid-training, resubmit, resume from the
    newest valid manifest — and land on the same final loss."""
    pytest.importorskip("torch")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    import flashruntime as flash
    from flashruntime.integrations import pytorch as fr_torch

    def ddp(extra: str = ""):
        return fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args=f"--steps 80 --checkpoint-every 20 {extra}".strip(),
        )

    baseline = flash.submit(ddp(), output_dir=tmp_path / "baseline")
    assert baseline.state.value == "SUCCEEDED", baseline.logs()

    workdir = tmp_path / "crashy"
    crashed = flash.submit(ddp("--kill-at-step 40"), output_dir=workdir)
    assert crashed.state.value == "FAILED"

    resumed = flash.submit(ddp("--kill-at-step 40"), output_dir=workdir)
    assert resumed.state.value == "SUCCEEDED", resumed.logs()
    assert resumed.trials[0]["resumed_from"] == 40
    assert resumed.trials[0]["final_loss"] == pytest.approx(
        baseline.trials[0]["final_loss"], abs=1e-6
    )
```

- [ ] **Step 3: Run the e2e tests**

Run: `pytest tests/test_examples_e2e.py -v`
Expected: sklearn test PASS (sklearn is in the dev venv); torch tests PASS if torch installed, else SKIP. **If torch is not installed, install the CPU wheel and re-run — criterion 3/3b must be seen green at least once:** `uv pip install torch --index-url https://download.pytorch.org/whl/cpu`

- [ ] **Step 4: Run the demo by hand and read its output**

Run: `python examples/bring_your_code_demo.py`
Expected: sweep prints 4 trials + a best; DDP prints SUCCEEDED; the kill/resume block prints `FAILED` then `SUCCEEDED` with `resumed_from step 40`-ish.

- [ ] **Step 5: Full suite + commit**

```bash
pytest && git add examples tests/test_examples_e2e.py \
  && git commit -m "feat(examples): user sklearn/pytorch repos + bring-your-code e2e demo"
```

---

### Task 11: Documentation + logging protocol

**Files:**
- Create: `docs/guides/bring-your-code.md`
- Modify: `README.md` (one pointer line), `AGENTS.md` (current-state bullet), workspace-root `PROGRESS.md` (work-log entry)

**Interfaces:** none — documentation of everything above. Use the exact APIs as implemented (verify names against the code, not from memory).

- [ ] **Step 1: Write `docs/guides/bring-your-code.md`**

Structure (write full prose; keep code blocks copy-paste runnable):

```markdown
# Bring your own code

FlashRuntime operates your training job — it never rewrites your model.

| You own | FlashRuntime owns |
|---|---|
| model, loop, loss, data, framework | launch, env vars, tracking, checkpoint validity, retry, recovery, artifacts |

## Run any repository (no changes to your code)

    import flashruntime as flash

    run = flash.submit(flash.CommandWorkload(
        command="python train.py --config configs/train.yaml",
        source="~/my-project",
        outputs=flash.OutputSpec(collect=["metrics.json", "checkpoints/**"]),
    ))
    print(run.state, run.artifacts)
    print(run.logs())

`command` is shlex-split (no shell); use `bash -c "..."` explicitly for pipes.
Your script's convention: write `metrics.json` to its working directory.

## scikit-learn: hyperparameter sweeps

[hpo() example from examples/bring_your_code_demo.py; explain: one task per
trial, `{placeholders}` → CLI flags, best_trial(); FlashRuntime never
splits a single .fit()]

## PyTorch: DDP

[vanilla path with fr_torch.ddp() — zero code changes if the script is
already DDP-ready; then the flashruntime.torch path with the full
examples/user_pytorch/train.py walkthrough: prepare / checkpoint /
log_metrics / start_step; the same-file-three-ways table; kill-and-resume
demo instructions. State the guardrail: the helper wraps torch's own DDP
and stops — FSDP/DeepSpeed users use those frameworks directly.]

## Hugging Face

[fr_hf.trainer() launch; flashruntime_callback() one-liner;
latest_checkpoint() → resume_from_checkpoint. HF Trainer already does DDP
internally when launched by torchrun.]

## Submitting to a coordinator (JobSpec)

[to_jobspec() example with a pinned image; POST /v1alpha1/jobs; isolation
tiers: standard (your machines / RunPod) vs sandboxed (community nodes —
fail-closed: a sandboxed task is only leased to sandbox_capable nodes).
Honest box: WHAT RUNS WHERE TODAY — local SDK runs now; service-side
command execution needs the flashnode argv runner (in progress);
multi-node DDP and remote providers are later slices.]

## Built-in algorithms

[algorithms/ + Cluster.train() remain as batteries-included examples —
never the required path.]
```

- [ ] **Step 2: Add pointers**

- `README.md`: under the Quickstart sections add one line: `- **Bring your own code** (sklearn / PyTorch / Hugging Face): [docs/guides/bring-your-code.md](docs/guides/bring-your-code.md)`
- `AGENTS.md` "Current state" list, append one bullet summarizing: `workloads/` + `flash.submit`, first concrete launcher/recipe, `integrations/`, `flashruntime.torch`, isolation-aware claim (fail-closed), and the flashnode argv-runner dependency for service-side execution.

- [ ] **Step 3: Follow the workspace logging protocol**

Read the LOGGING PROTOCOL section at the top of workspace-root `PROGRESS.md` (`../PROGRESS.md`) and append a dated work-log entry in exactly its prescribed format covering: what shipped (Tasks 1–11), test counts before/after, the flashnode cross-repo dependency, and follow-ups (spec §10).

- [ ] **Step 4: Docs test + full suite**

Run: `pytest tests/test_documentation.py -v && pytest`
Expected: all PASS (if `test_documentation.py` checks doc inventories, satisfy whatever it asserts about new files).

- [ ] **Step 5: Commit**

```bash
git add docs/guides/bring-your-code.md README.md AGENTS.md ../PROGRESS.md
git commit -m "docs: bring-your-code guide (sklearn/pytorch/hf) + progress log"
```

---

## Self-Review Notes

- **Spec coverage:** §4.1→Task 1, §4.2 (compile)→Task 2, §4.2 (launcher)→Task 3, local manifests→Task 4, §4.7 submit→Task 5, §4.5 adapters→Task 6, §4.6 helper→Task 7, §4.4 isolation gate→Tasks 8–9, §4.4 recipe→Task 9, §6 examples→Task 10, §7 docs→Task 11. Acceptance §9: 1→Tasks 5/10, 2→Task 10, 3/3b→Task 10, 4→Task 9, 5→every task's full-suite step, 6→Task 11.
- **Known deviations from spec (intentional, noted inline):** `compile_workload` is a function, not a `StrategyCompiler` subclass (no argv in `StrategyPlan`); service-side command *execution* is expansion-only until flashnode's argv runner lands.
- **Type consistency spot-checks:** `CommandWorkload.argv(params)` (Tasks 1/2/9), `LocalLaunchHandle.output_dir` (Tasks 3/5), `write_manifest(step_dir, *, job_id, attempt_id, step, world_size, framework)` (Tasks 4/6/7), payload key set (Tasks 8/9), `_restored_step` reset hook (Task 7 tests).
