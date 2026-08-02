"""Launchers: start a compiled `LaunchSpec` and watch it run.

The launcher owns everything environment-shaped that compilation must not
touch: resolving rendezvous addresses/ports, materializing `spec.files`
into a workdir, injecting credentials, spawning the processes (locally,
via torchrun, on Kubernetes, on Slurm), and reporting their fate.

Division of labor (the four-axes rule, HANDBOOK §2.1):
  compiler = *what* to run · launcher = *how processes start* ·
  provider = *where machines come from* · recipe = *what the user's code is*.

Failure semantics: a launcher NEVER retries or recovers — it reports.
Recovery is the coordinator's job (`recovery.decide()` chooses
RESTART_GROUP etc.); a launcher that silently relaunched would corrupt the
ledger's story and double-spend the user's budget.

Status: interface complete (final surface); `TorchrunLauncher` is the
first concrete (SPRINT_PLAN Week 2), then kubernetes/slurm/skypilot per
demand — see FLASHRUNTIME_EVALUATION §J for each substrate's role.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from typing import ClassVar

from flashruntime.strategies import LaunchSpec

__all__ = ["LaunchState", "LaunchHandle", "Launcher", "LaunchError"]


class LaunchError(Exception):
    """Launch could not start (preflight or spawn failure). Once processes
    are running, failures are reported through `LaunchHandle.poll`, never
    raised — the coordinator must always get a story, not a stack trace."""


class LaunchState(str, enum.Enum):
    """Coarse lifecycle every substrate can map onto (K8s pod phases,
    process exit codes, Slurm job states all reduce to these five)."""

    PENDING = "PENDING"      # accepted, processes not yet running
    RUNNING = "RUNNING"      # at least one worker alive, none failed
    SUCCEEDED = "SUCCEEDED"  # all workers exited 0
    FAILED = "FAILED"        # any worker exited non-zero / infra killed it
    CANCELLED = "CANCELLED"  # cancel() honored

    @property
    def terminal(self) -> bool:
        return self in (LaunchState.SUCCEEDED, LaunchState.FAILED, LaunchState.CANCELLED)


class LaunchHandle(ABC):
    """A live (or finished) launch. Handles are cheap views over substrate
    state — polling must be side-effect free and safe to call forever."""

    @abstractmethod
    def poll(self) -> LaunchState:
        """Current state. Must be non-blocking and idempotent. After a
        terminal state is returned once, every later call returns the same
        terminal state (substrates that garbage-collect finished jobs must
        cache the outcome)."""

    @abstractmethod
    def cancel(self) -> None:
        """Best-effort stop: idempotent, non-blocking, never raises on an
        already-terminal launch. The next `poll()` after a successful
        cancel eventually reports CANCELLED (grace periods allowed)."""

    def wait(self, timeout_seconds: float | None = None) -> LaunchState:
        """Convenience: poll until terminal or timeout; returns the last
        observed state (callers check `.terminal`). Default implementation
        polls at 1 s; substrates with native wait primitives override."""
        import time

        deadline = None if timeout_seconds is None else time.monotonic() + timeout_seconds
        while True:
            state = self.poll()
            if state.terminal or (deadline and time.monotonic() >= deadline):
                return state
            time.sleep(1.0)

    def logs(self, tail_lines: int = 200) -> str:
        """Recent combined output for the ledger/dashboard. Optional —
        default is an honest empty string, never fabricated content."""
        return ""

    @property
    def execution_id(self) -> str:
        """Substrate-native identifier (pid list, RayJob name, Slurm job
        id) for the ledger's `runtime_execution_id`. Default: unnamed."""
        return ""


class Launcher(ABC):
    """One way of starting process groups.

    Contract:
    - `healthy()` is the ONLY place environment preflight lives (binary on
      PATH, cluster reachable, quota available). Compilers stay pure.
    - `launch()` must either raise `LaunchError` before anything started,
      or return a handle — never leave orphaned half-started state without
      a handle that can cancel it.
    - Secrets/credentials are injected here (from the launcher's own
      config), merged UNDER `spec.env` so a plan can never override
      security-relevant variables.
    """

    #: Name matching `StrategyPlan.launcher` values
    #: ("local", "torchrun", "kubernetes", "slurm", "skypilot",
    #:  "flashruntime-leases").
    name: ClassVar[str]

    def healthy(self) -> tuple[bool, str]:
        """Preflight: can this launcher launch right now? Returns
        (ok, human-readable reason). Called before spending money; a False
        here routes the coordinator to PAUSE, not to a doomed launch."""
        return True, "no preflight implemented"

    @abstractmethod
    def launch(self, spec: LaunchSpec, job_id: str, attempt_id: str) -> LaunchHandle:
        """Start the group described by `spec`.

        Inputs: the compiled spec; job/attempt ids for labeling substrate
        resources (the flashml.dev/* label convention) so operators can
        trace any pod/process back to its ledger entry.
        Output: a `LaunchHandle` (see its contract).
        Raises: `LaunchError` only for failures *before* anything runs.
        """
