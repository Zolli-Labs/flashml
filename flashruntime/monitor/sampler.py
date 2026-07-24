"""ResourceSampler — per-attempt machine + process-tree telemetry.

A daemon thread `flash.submit` starts next to each launched attempt. Every
`period_s` it appends one JSON line to `<output_dir>/telemetry.jsonl`:

    {"ts": ..., "machine": {...}, "processes": [...]}

The viewer tail-reads this file exactly like metrics.jsonl (bounded window,
torn last line skipped), so plain appends are the right write discipline.

psutil is OPTIONAL (the `monitor` extra): with it, cpu/mem and per-process
stats are real; without it the machine sample carries `"limited": true` and
only stdlib facts (hostname, cpu_count, load_avg) — the dashboard renders
the gaps honestly instead of faking numbers. GPU facts come from nvidia-smi
when present (macOS/CPU boxes simply get `[]`).

TOTAL by contract, like everything that watches a run: a tick swallows every
exception, and the thread can never affect the training process it observes.
psutil is imported lazily inside `_psutil()` so `import flashruntime` stays
pydantic-only (the clean-venv core smoke pins this).
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from pathlib import Path

_GPU_QUERY = [
    "nvidia-smi",
    "--query-gpu=name,utilization.gpu,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]


def _psutil():
    """The psutil module, or None when the `monitor` extra is not installed.
    A function (not a module-level try/import) so tests can monkeypatch the
    seam and so the import cost is paid only inside the sampler thread."""
    try:
        import psutil

        return psutil
    except Exception:  # noqa: BLE001 — any import failure means "not available"
        return None


def _machine_sample() -> dict:
    """One machine-level sample. stdlib facts always; psutil facts when
    available; `limited` is True whenever the psutil numbers are absent
    (missing OR failing) so the UI can say 'install flashruntime[monitor]'."""
    sample: dict = {
        "hostname": socket.gethostname(),
        "cpu_count": os.cpu_count(),
        "load_avg": list(os.getloadavg()) if hasattr(os, "getloadavg") else None,
        "cpu_percent": None,
        "mem_total": None,
        "mem_used": None,
        "gpus": _gpu_sample(),
        "limited": True,
    }
    ps = _psutil()
    if ps is not None:
        try:
            sample["cpu_percent"] = ps.cpu_percent(interval=None)
            vm = ps.virtual_memory()
            sample["mem_total"] = vm.total
            sample["mem_used"] = vm.total - vm.available
            sample["limited"] = False
        except Exception:  # noqa: BLE001 — degrade to the stdlib-only sample
            pass
    return sample


def _gpu_sample() -> list[dict]:
    """GPU name/util/memory rows via nvidia-smi, or [] when there is no
    nvidia-smi (macOS, CPU boxes) or it misbehaves. A subprocess (not torch)
    so the sampler works for non-torch workloads and never imports a
    framework."""
    try:
        out = subprocess.run(_GPU_QUERY, capture_output=True, text=True, timeout=2)
        if out.returncode != 0:
            return []
        gpus = []
        for line in out.stdout.strip().splitlines():
            name, util, used, total = [p.strip() for p in line.split(",")]
            gpus.append(
                {
                    "name": name,
                    "util_percent": float(util),
                    "mem_used_mb": float(used),
                    "mem_total_mb": float(total),
                }
            )
        return gpus
    except Exception:  # noqa: BLE001 — no GPU story is ever worth an exception
        return []


def _process_tree(root_pid: int) -> list[dict]:
    """The launched process and all its descendants (torchrun's ranks, a
    sweep's children), one dict per process. Without psutil we can still name
    the root pid (the launcher knows it); with a vanished/denied root we
    return [] — the attempt likely just finished between ticks."""
    ps = _psutil()
    if ps is None:
        return [
            {
                "pid": root_pid,
                "ppid": None,
                "cmd": None,
                "cpu_percent": None,
                "rss_bytes": None,
                "create_time": None,
                "status": None,
            }
        ]
    try:
        root = ps.Process(root_pid)
        procs = [root] + root.children(recursive=True)
    except Exception:  # noqa: BLE001 — process gone / access denied
        return []
    out: list[dict] = []
    for p in procs:
        try:
            with p.oneshot():
                out.append(
                    {
                        "pid": p.pid,
                        "ppid": p.ppid(),
                        "cmd": " ".join(p.cmdline()[:3]) or p.name(),
                        "cpu_percent": p.cpu_percent(interval=None),
                        "rss_bytes": p.memory_info().rss,
                        "create_time": p.create_time(),
                        "status": p.status(),
                    }
                )
        except Exception:  # noqa: BLE001 — a process may exit mid-inspection
            continue
    return out


class ResourceSampler:
    """Sample machine + process-tree telemetry for one launched attempt.

    `start()` spawns a daemon thread that ticks immediately (a sub-period
    command must still leave one sample) and then every `period_s`;
    `stop()` wakes and joins it. Both are idempotent-enough for the SDK's
    best-effort use: stop on a never-started sampler is a no-op.
    """

    def __init__(self, output_dir: Path | str, root_pid: int, period_s: float = 2.0):
        self.output_dir = Path(output_dir)
        self.root_pid = int(root_pid)
        self.period_s = period_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.period_s + 1)
            self._thread = None

    def _run(self) -> None:
        # tick FIRST, then wait: a command that finishes inside one period
        # still gets a sample, and stop() during the wait exits promptly.
        while True:
            try:
                self._tick()
            except Exception:  # noqa: BLE001 — total by contract (module docstring)
                pass
            if self._stop.wait(self.period_s):
                return

    def _tick(self) -> None:
        sample = {
            "ts": time.time(),
            "machine": _machine_sample(),
            "processes": _process_tree(self.root_pid),
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.output_dir / "telemetry.jsonl", "a") as f:
            f.write(json.dumps(sample) + "\n")
