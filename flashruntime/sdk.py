"""flash.submit(): run a CommandWorkload locally and hand back a Run.

The execution model is compile → launch → wait → collect, once per Mode A
param set (mirroring the M0 engine's replay model). Sequential-by-design
keeps collection correct (a trial's outputs are copied out of the source
dir before the next trial can overwrite them). Service submission is a
different door: `workloads.command.to_jobspec()` POSTed to the coordinator.

`wait=True` (default) drives that loop inline and returns a finished Run —
byte-for-byte the old synchronous behavior. `wait=False` returns the Run
immediately and drives the loop on a daemon thread, so the caller can watch
it live. Either way the Run mirrors itself to `<output_dir>/run.json` after
every state change (the `viewer_v1` contract Task 6 renders from disk).
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import webbrowser
from pathlib import Path

from flashruntime.launchers import LaunchState
from flashruntime.launchers.local import LocalProcessLauncher
from flashruntime.protocol.v1alpha1 import RecoveryActionType
from flashruntime.recovery import classify, decide
from flashruntime.recovery.signals import from_local_launch
from flashruntime.strategies.command import compile_workload
from flashruntime.workloads.command import CommandWorkload

_JOB_ID = "local"  # deterministic: rerunning with the same output_dir resumes checkpoints

# Viewers opened by watch=True. A RunViewerServer runs on a daemon thread, so
# it never blocks interpreter exit; we hold a reference here so it survives for
# the whole process lifetime (a wait=True submit returns before the human is
# done looking at the page, and the server must outlive that return).
_VIEWERS: list = []


class Run:
    """Result handle for one submit().

    Lock discipline (stated once, here): a `wait=False` submit drives the
    launch loop on a daemon thread while the caller reads state/events live.
    A single `threading.Lock` (`_lock`) guards the live-read fields
    (`state`, `finished_at`, `_events`, `_attempts`) *and* the run.json
    write, so the file is always serialized from a consistent snapshot and
    a concurrent reader never observes a half-updated Run. `trials`,
    `artifacts`, and `_logs` are mutated only by the single driver thread
    and serialized only under the lock — they need no lock of their own.
    Readers take copies under the lock; the driver holds it only for the
    brief mutate-then-persist critical section. `_done` (a `threading.Event`)
    is set exactly once, on the terminal transition, so `wait()` blocks
    without polling.
    """

    def __init__(self, workload: CommandWorkload, output_dir: Path, max_restarts: int = 0):
        self.workload = workload
        self.output_dir = Path(output_dir)
        self.max_restarts = max_restarts  # the real recovery budget the viewer reports
        self.state: LaunchState = LaunchState.PENDING
        self.trials: list[dict] = []
        self.artifacts: list[Path] = []
        self.viewer_url: str | None = None  # set when submit(watch=True) opens a live viewer
        self.started_at: float = time.time()
        self.finished_at: float | None = None
        self._logs: list[str] = []
        self._events: list[dict] = []
        self._attempts: list[dict] = []
        self._lock = threading.Lock()
        self._done = threading.Event()
        with self._lock:  # materialize the PENDING record so run.json exists at once
            self._write_run_json()

    @property
    def run_json_path(self) -> Path:
        """Path to the viewer_v1 document this Run mirrors itself to."""
        return self.output_dir / "run.json"

    @property
    def events(self) -> list[dict]:
        """Snapshot copy of the append-only event log — safe to read while
        the driver thread appends (a bare reference could tear)."""
        with self._lock:
            return [dict(e) for e in self._events]

    @property
    def attempts(self) -> list[dict]:
        """Snapshot copy of the per-launch attempt rows (see `events` for why
        it copies)."""
        with self._lock:
            return [dict(a) for a in self._attempts]

    def wait(self, timeout: float | None = None) -> LaunchState:
        """Block until the run is terminal (or `timeout` seconds elapse),
        then return the current state. Event-based, not a poll loop: the
        driver sets `_done` on the terminal transition, so a waiter wakes the
        instant the run finishes with zero CPU spin — a polling loop would
        burn cycles and add up-to-interval latency. Callers check
        `.terminal` on the result (a timeout returns the last state seen)."""
        self._done.wait(timeout)
        return self.state

    def record_event(self, type: str, message: str) -> None:
        """Append one event to the append-only log and re-persist run.json.

        Reason: lifecycle/recovery decisions must reach the viewer (and Task
        4's recovery wiring) the moment they happen. Takes `_lock` so the
        append and the file write are one critical section."""
        with self._lock:
            self._events.append({"ts": time.time(), "type": type, "message": message})
            self._write_run_json()

    def _set_state(self, state: LaunchState) -> None:
        """Transition to `state` and re-persist run.json. On a terminal
        state, stamp `finished_at` and release `wait()`ers (sets `_done`).
        Under `_lock` so state + finished_at + write land atomically."""
        with self._lock:
            self.state = state
            if state.terminal and self.finished_at is None:
                self.finished_at = time.time()
            self._write_run_json()
            if state.terminal:
                self._done.set()

    def _add_attempt(self, attempt_id: str, job_id: str, handle, started_at: float) -> None:
        """Append a RUNNING attempt row (one per launch) and persist. `pid`
        is the launcher's execution_id (a string), so the viewer can show
        what is live."""
        with self._lock:
            self._attempts.append(
                {
                    "attempt_id": attempt_id,
                    "job_id": job_id,
                    "state": LaunchState.RUNNING.value,
                    "pid": handle.execution_id,
                    "started_at": started_at,
                    "finished_at": None,
                    "output_dir": str(handle.output_dir),
                }
            )
            self._write_run_json()

    def _finish_attempt(self, attempt_id: str, state: LaunchState) -> None:
        """Settle an attempt's terminal state + finished_at and persist.
        Matches the last row with this id, so restarts (Task 4) that append a
        fresh row settle the right one."""
        with self._lock:
            for row in reversed(self._attempts):
                if row["attempt_id"] == attempt_id:
                    row["state"] = state.value
                    row["finished_at"] = time.time()
                    break
            self._write_run_json()

    def _write_run_json(self) -> None:
        """Serialize the Run to <output_dir>/run.json atomically.

        Write a temp file in the same directory, then `os.replace` it onto
        the final path: os.replace is atomic on POSIX and Windows, so a
        viewer reading concurrently sees either the whole old file or the
        whole new one — never a torn, half-written document. Callers hold
        `_lock`, so the snapshot serialized here is internally consistent."""
        doc = {
            "contract": "viewer_v1",
            "workload": {
                "command": self.workload.argv(),
                "mode": self.workload.resolved_mode(),
                "source": self.workload.source.path,
            },
            "state": self.state.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "max_restarts": self.max_restarts,
            "attempts": [dict(a) for a in self._attempts],
            "events": [dict(e) for e in self._events],
            "trials": [dict(t) for t in self.trials],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.output_dir / f".run.json.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(doc, indent=2))
        os.replace(tmp, self.run_json_path)

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


def _watch_enabled(watch: bool | None) -> bool:
    """Resolve the `watch` tri-state. Explicit True/False wins; None is auto —
    on only at an interactive terminal AND outside CI. A pipeline or a piped
    stdout must never open a browser or hold a server open, so auto stays off
    there even though the run is identical."""
    if watch is not None:
        return watch
    return sys.stdout.isatty() and os.environ.get("CI") is None


def _open_viewer(run: Run) -> None:
    """Start a read-only RunViewerServer over the run's output dir, record its
    URL on the Run, open a browser at it, and keep the server alive for the
    process lifetime (`_VIEWERS`). Best-effort by design: the viewer is a
    convenience layered on top of the run, never load-bearing — a headless box
    with no browser, or a port that will not bind, must still run the job. So
    every failure here is swallowed and the submit proceeds unwatched."""
    try:
        from flashruntime.viewer.server import RunViewerServer

        server = RunViewerServer(run.output_dir)
        url = server.start()
        _VIEWERS.append(server)  # hold a reference: daemon thread, freed at exit
        run.viewer_url = url
        print(f"viewer: {url}")
        try:
            webbrowser.open(url)
        except Exception:  # noqa: BLE001 — headless / no browser must not fail the run
            pass
    except Exception:  # noqa: BLE001 — the viewer must never be able to fail a submit
        pass


def submit(
    workload: CommandWorkload,
    output_dir: str | Path | None = None,
    wait: bool = True,
    max_restarts: int = 0,
    watch: bool | None = None,
) -> Run:
    """Run a CommandWorkload locally. `max_restarts` (default 0 — no retry, the
    old behavior) is the automatic fault-tolerance budget: a FAILED attempt is
    classified and run against the versioned recovery policy, and unless the
    failure is a deterministic application error (FAIL_JOB → fail fast) the
    same spec is relaunched from the job-scoped checkpoint, up to this many
    times. Applies to a single launch and per-trial in a fan-out.

    `watch` opens the live run page (`flashruntime.viewer`) in a browser and
    prints its URL (recorded on `Run.viewer_url`). None (default) auto-decides:
    on at an interactive terminal, off in pipes/CI. The viewer server binds
    127.0.0.1 on a daemon thread, so it works for both `wait=True` (it outlives
    the synchronous return, so the page stays viewable) and `wait=False`."""
    out_root = Path(output_dir) if output_dir else Path(tempfile.mkdtemp(prefix="flashruntime-run-"))
    run = Run(workload, out_root, max_restarts=max_restarts)
    launcher = LocalProcessLauncher(out_root)

    # Open the viewer before driving so the page is live for the whole run —
    # including a wait=False submit the caller watches from the first frame.
    if _watch_enabled(watch):
        _open_viewer(run)

    fanout = workload.resolved_mode() == "independent_tasks" and workload.task_params
    param_sets: list[dict | None] = list(workload.task_params) if fanout else [None]

    if wait:
        # inline drive == today's synchronous behavior (Run finished on return)
        _drive(run, workload, launcher, param_sets)
    else:
        # daemon so an unwaited Run never blocks interpreter shutdown
        threading.Thread(
            target=_drive, args=(run, workload, launcher, param_sets), daemon=True
        ).start()
    return run


def _drive(
    run: Run, workload: CommandWorkload, launcher: LocalProcessLauncher, param_sets: list
) -> None:
    """The compile → launch → wait → collect loop, once per param set, then
    settle the Run's terminal state. Extracted from submit() so wait=True can
    call it inline (identical to the old synchronous path) and wait=False can
    run it on a daemon thread.

    Around each param set is the automatic-recovery loop (Task 4): on a FAILED
    launch, translate the exit into FailureSignals, classify it, and run the
    versioned policy — fail fast on a deterministic bug (FAIL_JOB), else
    relaunch from the job-scoped checkpoint until the restart budget is spent.
    `run.max_restarts == 0` reduces this to exactly one launch, the pre-Task-4
    behavior."""
    run._set_state(LaunchState.RUNNING)
    source_dir = Path(workload.source.path).expanduser()

    fanout = workload.resolved_mode() == "independent_tasks" and workload.task_params
    # The policy's blast-radius axis: independent trials retry one task; a
    # single/coordinated run restarts the whole (coordinated-training) group.
    mode = "independent_tasks" if fanout else "coordinated_training"

    states: list[LaunchState] = []
    for i, params in enumerate(param_sets):
        base_attempt = f"task-{i:03d}"
        # Fan-out trials are DIFFERENT workloads (distinct params) — each needs
        # its own checkpoint tree, or trial i could restore trial j's weights
        # (FLASHML_CKPT_DIR is per job id). The single-workload path keeps the
        # stable "local" id so a resubmit against the same output_dir resumes.
        job_id = f"local-{i:03d}" if fanout else _JOB_ID
        spec = compile_workload(workload, params)

        final_state = LaunchState.FAILED
        trial_metrics: dict | None = None
        # restart 0 is the first try; 1..max_restarts are recovery attempts.
        for restart in range(run.max_restarts + 1):
            # Restart attempts append -r1/-r2… but keep the SAME job_id, so the
            # launcher's job-scoped FLASHML_CKPT_DIR is shared across attempt
            # names — that is the whole trick: the child's ft.prepare() resumes
            # from the predecessor's newest valid manifest even though this
            # attempt has a different id.
            attempt_id = base_attempt if restart == 0 else f"{base_attempt}-r{restart}"
            started_at = time.time()
            handle = launcher.launch(spec, job_id, attempt_id)
            run.record_event("LAUNCH_STARTED", f"{attempt_id} launched (pid {handle.execution_id})")
            run._add_attempt(attempt_id, job_id, handle, started_at)
            final_state = handle.wait()
            run._logs.append(f"--- {attempt_id} ({final_state.value}) ---\n{handle.logs()}")
            run._finish_attempt(attempt_id, final_state)

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
                    trial_metrics = metrics  # the winning attempt's metrics (one trial per param set)

            if final_state is LaunchState.SUCCEEDED:
                break

            # FAILED: consult the recovery policy. Signals → class → decision,
            # both events recorded (with the decision's human reason) so the
            # ledger/viewer can explain every retry.
            exit_code = handle.exit_code
            decision = decide(classify(from_local_launch(exit_code, handle.logs())), mode)
            run.record_event(
                "FAILURE_CLASSIFIED", f"{attempt_id}: {decision.failure_class.value} (exit {exit_code})"
            )
            run.record_event(
                "RECOVERY_ACTION_SELECTED", f"{attempt_id}: {decision.action.value} — {decision.reason}"
            )
            # Fail fast on a deterministic application error — a retry only
            # re-hits the same bug. Any other action means "retry", and the
            # loop's range() enforces the budget, so it cannot spin forever.
            if decision.action is RecoveryActionType.FAIL_JOB:
                break

        if trial_metrics is not None:
            run.trials.append(trial_metrics)
        states.append(final_state)

    run._set_state(
        LaunchState.SUCCEEDED
        if states and all(s is LaunchState.SUCCEEDED for s in states)
        else LaunchState.FAILED
    )


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
