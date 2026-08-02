"""Tier-2 argv runner: run the user's own command inside a hardened container.

The module runners execute an allowlisted `python -m <module>`. This runner
executes whatever argv the job carried, which is what makes an arbitrary
machine a useful compute resource — and is why it is container-only. There
is no unsandboxed argv path here, by default or otherwise.

The security control is the isolation tier plus the operator's IMAGE
allowlist (what this volunteer consents to run), not a code allowlist: the
user's code inside a permitted image is unrestricted.

The task sees exactly one directory: its workdir bound at /work. Inputs are
pre-staged by the agent at /work/inputs; outputs are written to /work/out.
With --network none the job cannot fetch anything itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from flashnode.executor.hardening import container_name, harden_args
from flashnode.executor.images import DEFAULT_ALLOWED_IMAGE_PREFIXES, image_is_allowed
from flashnode.executor.runner import TaskExecutionError

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ArgvDockerRunner:
    def __init__(
        self,
        allowed_images: frozenset[str] = frozenset(),
        cpus: float = 2.0,
        memory_gb: float = 2.0,
        timeout_seconds: float = 3600.0,
        max_output_bytes: int = 2 * 1024**3,
        allowed_image_prefixes: frozenset[str] = DEFAULT_ALLOWED_IMAGE_PREFIXES,
    ):
        self.allowed_images = allowed_images
        self.allowed_image_prefixes = allowed_image_prefixes
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path:
        argv = payload.get("argv")
        if not argv or not isinstance(argv, list) or not all(isinstance(t, str) for t in argv):
            raise TaskExecutionError("payload 'argv' must be a non-empty list of strings")

        # Checked BEFORE any subprocess call, so a hostile value such as
        # "--privileged" can never reach docker's flag parser.
        image = payload.get("image")
        if not image or not image_is_allowed(
            image, self.allowed_images, self.allowed_image_prefixes
        ):
            raise TaskExecutionError(f"image {image!r} is not allowlisted — refusing to run")

        env_args: list[str] = []
        for key, value in (payload.get("env") or {}).items():
            if not _ENV_KEY.match(str(key)):
                raise TaskExecutionError(f"illegal env key {key!r} — refusing to run")
            env_args += ["--env", f"{key}={value}"]

        workdir = Path(workdir)
        outdir = workdir / "out"
        outdir.mkdir(parents=True, exist_ok=True)

        name = container_name(payload.get("task_id"))
        command = [
            "docker", "run", "--rm", "--name", name,
            *harden_args(
                workdir, cpus=self.cpus, memory_gb=self.memory_gb,
                # The host owner's local datasets this task asked for. Refused
                # here (before any subprocess) if this host does not lend them
                # — see hardening.local_data_mounts.
                local_inputs=payload.get("local_inputs"),
                # The device count the job asked for. Absent for every job
                # that does not ask, and validated in hardening.gpu_flags
                # rather than here, so both runners cannot drift on it.
                gpus=payload.get("gpus"),
            ),
            *env_args,
            image,          # argv follows the image, where docker treats it
            *argv,          # as the container command: leading '-' is inert
        ]
        try:
            proc = subprocess.run(
                command, capture_output=True, timeout=self.timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired:
            # subprocess.run's timeout kills the docker CLIENT process, not
            # the daemon-side container — it keeps running unless we kill it
            # by name ourselves. Best-effort: the container may already be
            # gone by the time we ask, and either way the wall-clock error
            # below is what the caller needs to see, not a kill failure.
            try:
                subprocess.run(["docker", "kill", name], capture_output=True, timeout=10, check=False)
            except Exception:
                pass
            raise TaskExecutionError(f"task exceeded {self.timeout_seconds}s wall clock")
        except OSError as exc:
            # `docker` missing/removed mid-run raises FileNotFoundError here
            # (a subclass of OSError). Degrade to a failed task, not a dead
            # agent — execute_one only catches TaskExecutionError/LeaseLost.
            raise TaskExecutionError(f"docker is unavailable: {exc}") from exc
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-800:]
            raise TaskExecutionError(f"task exited {proc.returncode}: {tail}")

        # metrics.json is load-bearing, not a preference: CommandRecipe sets
        # commit_key to <prefix>/metrics.json and the coordinator validates
        # the artifact at that key by sha256. Failing here turns an opaque
        # commit rejection into a clear task error.
        if not (outdir / "metrics.json").is_file():
            raise TaskExecutionError("task produced no metrics.json — nothing to commit")

        total = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file())
        if total > self.max_output_bytes:
            raise TaskExecutionError(
                f"task output {total} B exceeds the {self.max_output_bytes} B cap"
            )
        return outdir
