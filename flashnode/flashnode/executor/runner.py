"""Task runners: how a claimed payload actually executes.

Tier 1 — `SubprocessRunner` (this file, dev/trusted profile): runs an
**allowlisted Python module** in a fresh subprocess with a wall-clock
timeout and an isolated working directory. Refuses argv payloads (which would
run unsandboxed on the host). Suitable for the local loop and trusted pools;
it is not a security boundary against malicious code.

Tier 2 — Docker (`docker run` with cpu/memory limits, `--network none`,
non-root, read-only rootfs, image allowlist) implements this same
`run(payload, workdir, inputs) → outdir` interface next; the executor loop
does not change when the tier does. That is the point of the interface.

Both tiers fail closed: an unlisted module/image is refused before anything
executes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

DEFAULT_ALLOWED_MODULES = frozenset(
    {
        "flashml_workloads.sklearn_trial",
        "flashml_workloads.kmeans_shard",
        "flashml_workloads.sgd_trainer",
        "flashml_workloads.fedavg_worker",
    }
)

# The only environment a task subprocess inherits. Everything else — join
# codes, coordinator URLs, cloud credentials — is the *agent's* business and
# must never leak into workload code.
_TASK_ENV_WHITELIST = ("PATH", "HOME", "PYTHONPATH", "LANG", "LC_ALL", "TMPDIR")


def task_env() -> dict[str, str]:
    return {k: os.environ[k] for k in _TASK_ENV_WHITELIST if k in os.environ}


class TaskExecutionError(Exception):
    """The task refused to start or exited non-zero."""


class SubprocessRunner:
    def __init__(
        self,
        allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES,
        timeout_seconds: float = 600.0,
    ):
        self.allowed_modules = allowed_modules
        self.timeout_seconds = timeout_seconds

    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path:
        """Execute one task payload; return the output directory.

        Contract with the task module:
          spec.json  ← {task_id, params, inputs: {name: local path}}
          argv       ← python -m <module> --spec spec.json --out out/
          outputs    → files written under out/ (metrics.json required)
        """
        # Tier 1 has no isolation, so it must never execute a caller-supplied
        # command line. Argv workloads are container-only (ArgvDockerRunner);
        # refusing here keeps a misrouted payload from silently running
        # unsandboxed on the host.
        if "argv" in payload:
            raise TaskExecutionError(
                "argv payloads require a sandboxed runner — "
                "start the agent with --runner argv"
            )
        module = payload.get("module", "")
        if module not in self.allowed_modules:
            raise TaskExecutionError(f"module {module!r} is not allowlisted — refusing to run")

        workdir = Path(workdir)
        outdir = workdir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        spec_path = workdir / "spec.json"
        spec_path.write_text(
            json.dumps(
                {
                    "task_id": payload.get("task_id", ""),
                    "params": payload.get("params", {}),
                    "inputs": {name: str(path) for name, path in inputs.items()},
                }
            )
        )

        try:
            proc = subprocess.run(
                [sys.executable, "-m", module, "--spec", str(spec_path), "--out", str(outdir)],
                cwd=workdir,
                capture_output=True,
                timeout=self.timeout_seconds,
                check=False,
                env=task_env(),
            )
        except subprocess.TimeoutExpired:
            raise TaskExecutionError(f"task exceeded {self.timeout_seconds}s wall clock")
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-800:]
            raise TaskExecutionError(f"task exited {proc.returncode}: {tail}")
        if not (outdir / "metrics.json").is_file():
            raise TaskExecutionError("task produced no metrics.json — nothing to commit")
        return outdir
