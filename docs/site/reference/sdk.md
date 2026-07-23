# Reference: SDK (`flashruntime`)

The `flashruntime` top-level package. The core is pydantic-only — `import
flashruntime` pulls in **only pydantic**; the bring-your-own-code helpers
(`submit`, `CommandWorkload`, `integrations`) resolve lazily so the planner
stays a minimal import.

Signatures below are exact. Each entry says, in one line, *why it exists*.

---

## Running a workload

### `flash.submit`

```python
def submit(workload, output_dir=None, wait=True, max_restarts=0, watch=None) -> Run: ...
```

The local entry point — compiles a `CommandWorkload` into a launch spec, runs
it as a real subprocess (once per param set), collects artifacts, and returns a
`Run`.

- `output_dir` — where `run.json` and artifacts land; a temp dir if `None`.
  Reusing the **same** dir reuses the job id, so a checkpointed script resumes.
- `wait` — `True` drives the launch loop inline and returns a finished `Run`;
  `False` returns immediately and drives it on a daemon thread (watch it live).
- `max_restarts` — the automatic fault-tolerance budget (default `0` = no
  retry). A FAILED attempt is classified and run against the versioned recovery
  policy; a deterministic app error fails fast, anything else relaunches from
  the job-scoped checkpoint up to this many times.
- `watch` — open the live viewer and record its URL on `run.viewer_url`;
  `None` (default) auto-decides (on at an interactive terminal, off in
  pipes/CI).

### `Run`

The result handle, fully populated by the time a `wait=True` `submit()` returns.

| Attribute / method | Meaning |
|---|---|
| `run.state` | `LaunchState` — `SUCCEEDED` if every task succeeded, else `FAILED` |
| `run.trials` | list of parsed `metrics.json` dicts (one per task; fan-out merges its `params`) |
| `run.artifacts` | list of collected file `Path`s |
| `run.output_dir` | root the run wrote under |
| `run.viewer_url` | live-page URL if `watch` opened one, else `None` |
| `run.events` | snapshot copy of the append-only event log |
| `run.attempts` | snapshot copy of the per-launch attempt rows |
| `run.run_json_path` | path to the `viewer_v1` `run.json` the Run mirrors itself to |
| `run.wait(timeout=None)` | block until terminal (event-based, no poll); returns the state |
| `run.logs(tail_lines=200)` | captured stdout+stderr (tail) |
| `run.best_trial(metric=None, maximize=None)` | best trial by `metric` (defaults from the workload's `OutputSpec`); `None` if none reported it |

---

## Describing a workload

### `flash.CommandWorkload`

```python
class CommandWorkload(BaseModel):
    command: str | list[str]                 # shlex-split (NO shell) or an argv list
    source: Source = Source()                # where the user's code lives
    image: ImageSpec | None = None           # pinned image (required only for the service path)
    env: dict[str, str] = {}
    inputs: dict[str, str] = {}              # each value must be an artifact:// URI
    outputs: OutputSpec = OutputSpec()
    resources: Requirements = Requirements() # resource hints (dormant locally; for future providers)
    isolation: IsolationSpec = IsolationSpec()
    mode: str = "auto"                       # auto | local | independent_tasks | coordinated
    checkpoint: CheckpointPolicy | None = None
    task_params: list[dict] | None = None    # {name} placeholders filled per entry (Mode A fan-out)
```

The user-facing description of a "bring your own code" workload — *what* to run
and what FlashRuntime should do around it. It never describes *how* distributed
math happens; that belongs to your code. `command` is `shlex`-split (there is no
shell — pipes need an explicit `command="bash -c '...'"`).

### `flash.Source`

```python
class Source(BaseModel):
    path: str = "."                          # local dir; ~ is expanded
    git_revision: str | None = None          # reserved for remote packaging (later slice)
```

Where the user's code lives — a `flash.Source`, not a bare string.

### `flash.OutputSpec`

```python
class OutputSpec(BaseModel):
    prefix: str = "artifact://jobs/{job_id}/"
    collect: list[str] = ["metrics.json"]    # globs resolved against the script's cwd
    primary_metric: str | None = None        # the metrics.json key best_trial() ranks by
    maximize: bool = True
```

What to keep after a run, and how to rank trials. `metrics.json` in `collect`
(the default) is what populates `run.trials`.

---

## Planning a job

### `flash.plan`

```python
def plan(request: PlanRequest) -> PlanReport: ...
```

Deterministic, explainable strategy selection — turns model + hardware +
objective into a ranked, explained `PlanReport` (no cluster required). The
closed-form arithmetic is framework-import-free.

### `flash.render`

```python
def render(report: PlanReport) -> str: ...
```

Renders a `PlanReport` as human-readable text (the numbers, the chosen plan,
and the rejected alternatives with their reasons).

```python
import flashruntime as flash

report = flash.plan(flash.PlanRequest(
    workload=flash.TransformerFineTune(model="Qwen/Qwen2.5-7B", method="lora"),
    resources=flash.Resources(gpus=4, gpu_type="RTX4090"),
    objective=flash.Objective(mode="balanced", deadline_minutes=240),
))
print(flash.render(report))
```

### Plan inputs

```python
class Resources(BaseModel):
    gpus: int = 0                            # 0 = CPU-only
    gpu_type: str | None = None              # 'A100-40GB', 'L40S', 'RTX4090', ...
    vram_gb: float | None = None             # per-GPU VRAM override
    hosts: int = 1
    cpu_ram_gb: float = 32.0
    cpu_cores: int = 8
    hourly_cost_usd_per_gpu: float | None = None

class Objective(BaseModel):
    mode: str = "balanced"                   # cheapest | fastest | balanced | reliable
    max_cost_usd: float | None = None
    deadline_minutes: float | None = None
    allow_quantization: bool = True
    allow_cpu_offload: bool = True
    allow_nvme_offload: bool = False
```

Workload intents: `TransformerFineTune`, `PyTorchTraining`, `ClassicalML`,
`IndependentTasks`. See the planner walkthrough (in the repo under
`docs/planner/`) for the estimator arithmetic.

### `flash.run` — designed, not yet built

```python
def run(plan, coordinator_url=None): ...     # raises NotImplementedError
```

The plan-to-execution bridge. It is designed (the docstring describes the
intended pipeline) but not implemented, so it raises `NotImplementedError`
rather than half-running. Today you `flash.plan()` and submit the JobSpec
yourself.

---

## Related references

- [Integrations](integrations.md) — the `fr_torch` / `fr_sklearn` / `fr_hf`
  adapters that build a `CommandWorkload` for you.
- [torch helper](torch-helper.md) — the seven `flashruntime.torch` functions.
- [CLI](cli.md) — the `flashruntime` command.
