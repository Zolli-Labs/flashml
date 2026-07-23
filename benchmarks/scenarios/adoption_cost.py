"""adoption_cost — how much does it cost to ADOPT flashruntime vs the alternatives?

HYPOTHESIS: making a plain training script fault-tolerant with flashruntime is a
handful of lines and one tiny (pydantic-only) dependency — cheaper to adopt, and
faster to import, than ray or accelerate.

MEASUREMENT METHOD (auditable from this file alone) — no timing except imports:
  * adoption LOC: difflib between a vanilla script and its framework-ready
    counterpart (committed, cited snippets), counting inserted/changed non-blank
    lines. Reported for flashruntime and (from its own docs' snippet) accelerate.
  * import time: `python -c "import X"` wall-clock, median of 5, for flashruntime
    and torch (and ray/accelerate IF importable — else a skip note).
  * dependency count: core (non-extra) requirements via importlib.metadata for
    flashruntime, and for ray/accelerate when installed.
"""

from __future__ import annotations

import difflib
import importlib.util
import sys
from importlib.metadata import PackageNotFoundError, requires
from pathlib import Path

from benchmarks._util import (
    REPO_ROOT,
    SNIPPETS,
    bench_env,
    median,
    time_subprocess,
)
from benchmarks.schema import ResultRow

name = "adoption_cost"
hypothesis = "Adopting flashruntime is a handful of lines and a tiny dependency footprint — less code and a faster import than ray or accelerate."


def _adopt_loc(vanilla: Path, adapted: Path) -> int:
    a, b = vanilla.read_text().splitlines(), adapted.read_text().splitlines()
    added = 0
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag in ("insert", "replace"):
            added += sum(1 for line in b[j1:j2] if line.strip())
    return added


def _import_ms(module: str, env: dict) -> float | None:
    if importlib.util.find_spec(module) is None:
        return None
    times = [time_subprocess([sys.executable, "-c", f"import {module}"], REPO_ROOT, env) for _ in range(5)]
    return round(median([t * 1000 for t in times]), 1)


def _core_dep_count(package: str) -> int | None:
    try:
        reqs = requires(package) or []
    except PackageNotFoundError:
        return None
    return sum(1 for r in reqs if "; extra" not in r and "extra ==" not in r)


def run(repeats: int) -> ResultRow:
    env = bench_env()
    flash_loc = _adopt_loc(SNIPPETS / "adopt_vanilla.py", SNIPPETS / "adopt_flashruntime.py")
    accel_loc = _adopt_loc(SNIPPETS / "adopt_vanilla.py", SNIPPETS / "adopt_accelerate.py")

    comparators: dict[str, float] = {
        "accelerate_adopt_loc": float(accel_loc),
        "flashruntime_core_deps": float(_core_dep_count("flashruntime") or 0),
    }
    notes = [
        "adoption LOC = inserted/changed non-blank lines from a vanilla script to its framework-ready "
        "form (difflib); snippets are cited from each project's own docs",
        "LOC is deterministic — repeats do not vary it (import time is the median of 5 subprocess timings)",
    ]
    for label, module in [("flashruntime", "flashruntime"), ("torch", "torch"),
                          ("ray", "ray"), ("accelerate", "accelerate")]:
        ms = _import_ms(module, env)
        if ms is None:
            notes.append(f"{label} not installed — import-time/dep-count comparator skipped")
        else:
            comparators[f"{label}_import_ms"] = ms
    for pkg in ("ray", "accelerate"):
        deps = _core_dep_count(pkg)
        if deps is not None:
            comparators[f"{pkg}_core_deps"] = float(deps)

    return ResultRow(
        scenario=name,
        unit="lines to adopt",
        median=float(flash_loc),
        p10=float(flash_loc),
        p90=float(flash_loc),
        repeats=repeats,
        comparators=comparators,
        notes=notes,
    )
