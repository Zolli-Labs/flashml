"""The scenario registry — a dict, and re-exports of the result schema.

Deliberately tiny (readability §2b): the schema lives in ``benchmarks.schema``
(a leaf module, so scenarios import ``ResultRow`` without a cycle); here we just
map name → scenario. Adding a scenario is one new file under ``scenarios/`` that
exposes ``name`` / ``hypothesis`` / ``run(repeats)`` plus one line in the
``_MODULES`` tuple below — nothing else in the suite changes.
"""

from __future__ import annotations

from benchmarks.schema import ResultRow, Scenario
from benchmarks.scenarios import (
    adoption_cost,
    checkpoint_integrity,
    crash_storm,
    fanout_throughput,
    fault_recovery_matrix,
    hpo_sweep,
    launch_overhead,
    lease_recovery_latency,
    loop_overhead,
    recovery_economics,
    submit_latency,
)

__all__ = ["ResultRow", "Scenario", "SCENARIOS"]

# One line per scenario. Each module satisfies ``Scenario`` structurally
# (module-level ``name`` / ``hypothesis`` / ``run``), so the file *is* the
# scenario — no class boilerplate.
_MODULES = (
    launch_overhead,
    loop_overhead,
    recovery_economics,
    hpo_sweep,
    adoption_cost,
    fault_recovery_matrix,
    checkpoint_integrity,
    crash_storm,
    submit_latency,
    fanout_throughput,
    lease_recovery_latency,
)

SCENARIOS: dict[str, Scenario] = {m.name: m for m in _MODULES}
