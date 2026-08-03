"""Trusted-pool argv execution: no container, no sandbox, by explicit
operator opt-in only.

This runner exists for hosts that CANNOT run Docker — Colab notebooks and
provider pods are themselves containers — inside a team pool whose members
chose to trust each other. It is not a security boundary and never claims
to be: the placement contract (pool + allowFallback + the operator's
--runner trusted opt-in) is what keeps strangers' code away from it.

Same interface as SubprocessRunner/ArgvDockerRunner:
run(payload, workdir, inputs) -> outdir.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from flashnode.executor.runner import TaskExecutionError, task_env

_CONTAINER_WORKDIR = "/work"


class TrustedArgvRunner:
    def __init__(self, timeout_seconds: float = 600.0):
        self.timeout_seconds = timeout_seconds
        # Evidence attributes, same contract and same reset rule as
        # SubprocessRunner: a stale value is a measurement of a DIFFERENT
        # run wearing this one's name.
        self.last_exit_code: int | None = None
        #: Always "" — no image bytes executed; echoing the payload's image
        #: reference would claim a container ran when none did.
        self.last_image_digest: str = ""

    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path:
        self.last_exit_code = None
        workdir = Path(workdir)
        argv = payload.get("argv")
        if not isinstance(argv, list) or not argv or not all(
            isinstance(a, str) for a in argv
        ):
            raise TaskExecutionError(
                "trusted runner requires a payload with a string-list 'argv'"
            )
        # Rewrite /work-prefixed TOKENS onto the real workdir. Token-wise,
        # never substring: an argument that merely contains "/work" belongs
        # to the submitter. The compiled argv uses /work because the docker
        # runners bind the workdir there; this runner has no container, so
        # /work is a naming convention to honour, not a mount to make.
        rewritten = [
            str(workdir) + a[len(_CONTAINER_WORKDIR):]
            if a == _CONTAINER_WORKDIR or a.startswith(_CONTAINER_WORKDIR + "/")
            else a
            for a in argv
        ]
        outdir = workdir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        try:
            proc = subprocess.run(
                rewritten,
                cwd=workdir,
                env=task_env(),
                timeout=self.timeout_seconds,
                capture_output=True,
            )
        except subprocess.TimeoutExpired as exc:
            raise TaskExecutionError(
                f"task exceeded {self.timeout_seconds}s"
            ) from exc
        except OSError as exc:
            raise TaskExecutionError(f"could not start task: {exc}") from exc
        self.last_exit_code = proc.returncode
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-2000:]
            raise TaskExecutionError(
                f"task exited {proc.returncode}: {tail}"
            )
        if not (outdir / "metrics.json").is_file():
            # Same rule as both sibling runners: an exit-0 workload that
            # wrote no metrics is a task failure HERE, attributably — not a
            # mysterious commit rejection three hops later.
            raise TaskExecutionError("task produced no metrics.json — nothing to commit")
        return outdir
