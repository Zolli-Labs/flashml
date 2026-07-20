"""Admission and workload probes: measured capability, never self-reported.

Short benchmarks run at join time (and periodically after) so the
coordinator's placement/capability decisions rest on measurements, not on
what a host *claims* (the Vast.ai lesson — master report §9.1: hosts earn
eligibility through measurement). Results ride the existing
`NodeRegistration.capabilities`/`labels` — raw numbers each, NEVER a
single synthetic score (§9.2: a composite hides exactly the bottleneck
that matters for a given workload).

Standard probe names (implement as separate AdmissionProbe subclasses so
each stays independently budgeted and independently failable):
  cpu_hash_mbps       — sustained single-core throughput (sha256 over RAM)
  mem_bandwidth_mbps  — large-buffer copy rate
  disk_write_mbps     — workdir sequential write (the checkpoint path!)
  net_down_mbps       — artifact-download rate from the coordinator
  gpu_*               — deferred to the GPU tier (nvml presence first)

Budget discipline: probes must respect their `budget_seconds` — admission
runs while a human waits on `flashnode work` startup; the whole suite gets
a few seconds, not minutes. Re-probing cadence (daily? on capability
change?) is a coordinator policy, not the agent's.

Status: interface + orchestration complete; concrete probes land with the
GPU/community tier (HANDBOOK missing-list; flashnode AGENTS §profiles).
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

__all__ = ["ProbeResult", "AdmissionProbe", "run_admission"]


class ProbeResult(BaseModel):
    """One raw measurement. `value=None` means the probe failed — recorded
    honestly (with `detail`), never fabricated and never fatal to the
    node's admission (a laptop with a broken disk probe can still serve
    CPU tasks; the *coordinator* decides what a missing number implies)."""

    name: str
    value: float | None = None
    unit: str = ""
    duration_s: float = Field(ge=0, default=0.0)
    detail: str = ""


class AdmissionProbe(ABC):
    """One benchmark, one number.

    Contract: `run` must (a) finish within `budget_seconds` or return its
    best partial measurement, (b) touch only its own scratch space, (c) be
    safe to re-run at any time on a live node — probes may execute while
    tasks run, so they must be gentle (no cache-thrashing full-RAM sweeps).
    """

    #: Stable name from the standard list above — it becomes the
    #: capability key the planner/scheduler reads; renaming one is a
    #: protocol-level event, treat with the same care as a schema field.
    name: ClassVar[str]

    @abstractmethod
    def run(self, budget_seconds: float) -> ProbeResult:
        """Measure and return. Should not raise — but `run_admission`
        guards anyway, because one broken probe must never sink a node's
        whole admission."""


def run_admission(
    probes: list[AdmissionProbe], total_budget_seconds: float = 10.0
) -> dict[str, ProbeResult]:
    """Run a probe suite under a shared budget; return name → result.

    Budget is split evenly across remaining probes and re-balanced as
    probes finish early. Failures are captured as `value=None` results
    (isolation rule) so callers always get one entry per probe — the
    difference between "not measured" and "measured badly" stays visible.
    """
    results: dict[str, ProbeResult] = {}
    remaining = list(probes)
    deadline = time.monotonic() + total_budget_seconds
    while remaining:
        probe = remaining.pop(0)
        per_probe = max(0.1, (deadline - time.monotonic()) / (len(remaining) + 1))
        started = time.monotonic()
        try:
            result = probe.run(budget_seconds=per_probe)
        except Exception as exc:  # isolation: a broken probe is a data point
            result = ProbeResult(
                name=probe.name,
                value=None,
                duration_s=time.monotonic() - started,
                detail=f"probe failed: {exc}",
            )
        results[probe.name] = result
    return results
