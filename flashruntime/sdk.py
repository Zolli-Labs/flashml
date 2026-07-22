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
        # Fan-out trials are DIFFERENT workloads (distinct params) — each needs
        # its own checkpoint tree, or trial i could restore trial j's weights
        # (FLASHML_CKPT_DIR is per job id). The single-workload path keeps the
        # stable "local" id so a resubmit against the same output_dir resumes.
        job_id = f"local-{i:03d}" if fanout else _JOB_ID
        spec = compile_workload(workload, params)
        started_at = time.time()
        handle = launcher.launch(spec, job_id, attempt_id)
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
