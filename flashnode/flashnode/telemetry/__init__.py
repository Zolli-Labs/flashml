"""Telemetry: what this machine is doing, as data the network can use.

Scope boundary — telemetry is *resource observation only*. Lease/attempt
heartbeats and task progress already live in `executor/loop.py` and must
stay there (they are correctness signals, not metrics; mixing them with
best-effort telemetry would let a metrics hiccup look like a dead task).

Intended flow: the work loop calls `collector.sample()` on its node-
heartbeat cadence and attaches the sample to the heartbeat. NOTE for the
implementer: `protocol.v1alpha1.NodeHeartbeat` does not carry a telemetry
field yet — adding `telemetry: dict | None = None` there is an *additive*
protocol change (allowed within v1alpha1) and is the actual wiring step;
do it protocol-first, then here.

Privacy stance (trust-through-transparency, AGENTS hard rule 5): samples
describe the MACHINE (utilization, free space, temperatures), never the
user of the machine (no process lists, no window titles, no network
destinations). Every field an owner might question must be visible in this
schema — that is why it is a typed model, not a dict.

Status: interface complete; the psutil-based collector is a small slice
(psutil is already an agent dependency via inventory/) — lands with the
community-pool tier alongside `benchmark/`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from pydantic import BaseModel, Field

__all__ = ["GpuSample", "TelemetrySample", "TelemetryCollector"]


class GpuSample(BaseModel):
    """One GPU's instantaneous state (NVML-shaped; all optional because
    partial visibility is normal — record what's readable, omit the rest)."""

    index: int = 0
    utilization_percent: float | None = None
    memory_used_bytes: int | None = None
    memory_total_bytes: int | None = None
    temperature_c: float | None = None
    power_watts: float | None = None


class TelemetrySample(BaseModel):
    """A point-in-time snapshot of machine health.

    Fields are chosen for what the *coordinator* can act on: scheduling
    away from thermally-throttling or disk-full nodes, and (later) the
    reliability score's thermal-stability input (master report §9.1)."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    cpu_percent: float | None = Field(default=None, ge=0)
    memory_used_bytes: int | None = Field(default=None, ge=0)
    memory_total_bytes: int | None = Field(default=None, ge=0)
    disk_free_bytes: int | None = Field(
        default=None, ge=0, description="Free space on the workdir volume — the checkpoint path"
    )
    load_1m: float | None = Field(default=None, ge=0)
    gpus: list[GpuSample] = Field(default_factory=list)


class TelemetryCollector(ABC):
    """Produces samples; owns nothing else.

    Contract: `sample()` must be cheap (called on the heartbeat cadence,
    default 5 s), must never raise for missing sensors (absent fields stay
    None — same honesty rule as probes), and must never block on hardware
    that is busy (a wedged NVML call gets a short internal timeout and an
    omitted field, not a stuck heartbeat — the heartbeat is a correctness
    signal and telemetry must never delay it).
    """

    @abstractmethod
    def sample(self) -> TelemetrySample:
        """Take one snapshot now."""
