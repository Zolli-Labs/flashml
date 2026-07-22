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
        # Popen dups the fd into the child, so the parent's handle can (and
        # must) be closed immediately — leaving it open leaks a descriptor
        # and trips ResourceWarning; logs() re-reads the file from disk.
        log_file = open(log_path, "w")
        try:
            proc = subprocess.Popen(
                spec.argv,
                cwd=str(workdir),
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
        except OSError as exc:
            raise LaunchError(f"failed to start {spec.argv[0]!r}: {exc}") from exc
        finally:
            log_file.close()
        return LocalLaunchHandle(proc, log_path, outdir)
