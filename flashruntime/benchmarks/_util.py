"""Shared measurement primitives for the benchmark scenarios.

Kept out of ``registry.py`` (which is just the schema + the dict) so a
scenario file stays short and readable. Nothing here imports torch / sklearn /
flashruntime at module load — scenarios pull heavy deps in lazily inside
``run()`` so ``import benchmarks`` (and the registry) stays cheap and
dependency-free.
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = REPO_ROOT / "examples"
SNIPPETS = Path(__file__).resolve().parent / "scenarios" / "snippets"


class ScenarioUnavailable(Exception):
    """A scenario's hard dependency is missing on this host (e.g. torch or
    torchrun). The runner reports it as a skip and emits no row — distinct
    from a comparator being unavailable, which is a `notes` line on a row that
    still measures flashruntime itself."""


# --------------------------------------------------------------------------
# venv / PATH handling
# --------------------------------------------------------------------------
# The venv's bin dir, derived from THIS interpreter — not from the caller's
# PATH. torchrun lives at ``.venv/bin/torchrun``; a subprocess that inherits a
# bare or system PATH silently fails to find it (a recurring env bug in this
# repo — the e2e suite hit it too). Every subprocess we spawn, and flash.submit's
# own launcher (which inherits os.environ), must see this dir on PATH.
VENV_BIN = Path(sys.executable).parent


def ensure_venv_on_path() -> None:
    """Prepend the venv bin dir to ``os.environ['PATH']`` in-process, once, so
    flash.submit's LocalProcessLauncher (which copies os.environ into the
    child) can resolve ``torchrun``. Idempotent."""
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if str(VENV_BIN) not in parts:
        os.environ["PATH"] = os.pathsep.join([str(VENV_BIN), *parts])


def bench_env(**extra: str) -> dict[str, str]:
    """A subprocess env with the venv bin EXPLICITLY prepended to PATH (see
    VENV_BIN above for why we don't trust the inherited PATH) plus any extra
    contract vars (e.g. FLASHML_CKPT_DIR to isolate a run's checkpoints)."""
    env = dict(os.environ)
    env["PATH"] = os.pathsep.join([str(VENV_BIN), env.get("PATH", "")])
    env.update({k: str(v) for k, v in extra.items()})
    return env


# --------------------------------------------------------------------------
# timing + stats
# --------------------------------------------------------------------------
def timed(fn) -> tuple[float, object]:
    """Wall-clock (perf_counter) of calling ``fn`` plus its return value."""
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def time_subprocess(argv: list[str], cwd: str | Path, env: dict[str, str]) -> float:
    """Wall-clock of one ``subprocess.run`` (stdout/stderr swallowed). Raises
    with the captured output on a nonzero exit so a broken comparator fails
    loudly instead of silently reporting a bogus time."""
    start = time.perf_counter()
    proc = subprocess.run(argv, cwd=str(cwd), env=env, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if proc.returncode != 0:
        raise RuntimeError(f"{argv[0]} exited {proc.returncode}:\n{proc.stdout}\n{proc.stderr}")
    return elapsed


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile (q in [0,1]); ``statistics.quantiles``
    needs >=2 points, but a repeats=1 smoke run has one — this degrades to that
    single value instead of raising."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    frac = pos - lo
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * frac


def median(values: list[float]) -> float:
    return percentile(values, 0.5)


def maxrss_mb() -> float:
    """Peak RSS of the largest child process so far, in MB. ru_maxrss is a
    non-resettable high-water mark; its UNIT differs by platform — bytes on
    macOS, kilobytes on Linux — which we normalise here."""
    import resource

    raw = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    divisor = 1024**2 if sys.platform == "darwin" else 1024
    return round(raw / divisor, 1)


def count_loc(path: Path) -> int:
    """Non-blank, non-comment source lines of ``path`` — the "setup LOC" a user
    writes. Snippets carry a leading source-URL comment (cited, not counted)."""
    n = 0
    for line in Path(path).read_text().splitlines():
        stripped = line.strip()
        if stripped and not stripped.startswith("#"):
            n += 1
    return n


# --------------------------------------------------------------------------
# host descriptor (bench_v1 "host" block)
# --------------------------------------------------------------------------
def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, timeout=5,
            )
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip()
        except Exception:  # noqa: BLE001 — best-effort label only
            pass
    return platform.processor() or platform.machine()


def _ram_gb() -> float | None:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            return round(int(out.stdout.strip()) / 1024**3, 1)
        pages = os.sysconf("SC_PHYS_PAGES")
        return round(pages * os.sysconf("SC_PAGE_SIZE") / 1024**3, 1)
    except Exception:  # noqa: BLE001 — RAM is descriptive metadata, never load-bearing
        return None


def _optional_version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:  # noqa: BLE001
        return None


def host_info() -> dict:
    from importlib.metadata import version

    return {
        "os": platform.platform(),
        "cpu": _cpu_name(),
        "cores": os.cpu_count(),
        "ram_gb": _ram_gb(),
        "python": platform.python_version(),
        "torch": _optional_version("torch"),
        "flashruntime": version("flashruntime"),
    }
