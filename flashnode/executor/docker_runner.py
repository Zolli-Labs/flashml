"""Tier-2 task runner: allowlisted container execution.

Same `run(payload, workdir, inputs) → outdir` interface as
`SubprocessRunner`; execution happens inside `docker run` with the AGENTS.md
security contract: image allowlist (fail closed, checked before any
subprocess), `--network none`, cpu/memory limits, read-only rootfs with the
work directory as the only writable mount, host-uid user mapping (no root in
the container's world), and a wall-clock timeout. The docker socket is used
*by the agent* to launch the task — it is never mounted into the task.

The task sees exactly one directory: its workdir bound at `/work`
(spec.json in, inputs under /work/inputs, outputs to /work/out) — so
spec.json is written with *container* paths, not host paths.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args
from flashnode.executor.runner import DEFAULT_ALLOWED_MODULES, TaskExecutionError


class DockerRunner:
    def __init__(
        self,
        allowed_images: frozenset[str],
        allowed_modules: frozenset[str] = DEFAULT_ALLOWED_MODULES,
        cpus: float = 2.0,
        memory_gb: float = 2.0,
        timeout_seconds: float = 900.0,
    ):
        self.allowed_images = allowed_images
        self.allowed_modules = allowed_modules
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.timeout_seconds = timeout_seconds

    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path:
        module = payload.get("module", "")
        if module not in self.allowed_modules:
            raise TaskExecutionError(f"module {module!r} is not allowlisted — refusing to run")
        image = payload.get("image")
        if not image:
            raise TaskExecutionError("payload carries no image — the docker runner requires one")
        if image not in self.allowed_images:
            raise TaskExecutionError(f"image {image!r} is not allowlisted — refusing to run")

        workdir = Path(workdir)
        outdir = workdir / "out"
        outdir.mkdir(parents=True, exist_ok=True)
        (workdir / "spec.json").write_text(
            json.dumps(
                {
                    "task_id": payload.get("task_id", ""),
                    "params": payload.get("params", {}),
                    # container paths: the task only ever sees /work
                    "inputs": {
                        name: f"{CONTAINER_WORKDIR}/{Path(path).relative_to(workdir)}"
                        for name, path in inputs.items()
                    },
                }
            )
        )

        argv = [
            "docker", "run", "--rm",
            *harden_args(workdir, cpus=self.cpus, memory_gb=self.memory_gb),
            image,
            "python", "-m", module,
            "--spec", f"{CONTAINER_WORKDIR}/spec.json",
            "--out", f"{CONTAINER_WORKDIR}/out",
        ]
        try:
            proc = subprocess.run(
                argv, capture_output=True, timeout=self.timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired:
            raise TaskExecutionError(f"task exceeded {self.timeout_seconds}s wall clock")
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-800:]
            raise TaskExecutionError(f"task exited {proc.returncode}: {tail}")
        if not (outdir / "metrics.json").is_file():
            raise TaskExecutionError("task produced no metrics.json — nothing to commit")
        return outdir
