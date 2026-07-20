"""Communication feasibility: is the network good enough for this strategy?

First-order per-step traffic model (evaluation §E):

- DDP: one gradient all-reduce per optimizer step. Ring all-reduce moves
  ≈ 2·(N−1)/N · gradient_bytes per GPU. For PEFT the gradients are only the
  adapters' — which is why LoRA+DDP tolerates weak links that full
  fine-tuning cannot.
- FSDP2/ZeRO-3: per layer, all-gather params (forward), re-gather
  (backward), reduce-scatter grads ⇒ ≈ 3× sharded-parameter bytes per step
  in ~3·L latency-sensitive collectives.

Verdict = scaling efficiency E = t_compute / (t_compute + t_exposed_comm),
with half the compute assumed overlappable. E < REJECT ⇒ infeasible on this
link; REJECT ≤ E < WARN ⇒ feasible with a warning. Thresholds are starting
values to be recalibrated from ledger data. [assumption]
"""

from __future__ import annotations

from dataclasses import dataclass

from flashruntime.planner.catalog import INTERCONNECT_GBPS
from flashruntime.planner.resolve import GB

E_REJECT = 0.5
E_WARN = 0.7
OVERLAP_FRACTION = 0.5  # fraction of comm hidden under backward compute [assumption]


@dataclass(frozen=True)
class CommVerdict:
    efficiency: float  # 0..1; 1.0 = no visible communication cost
    feasible: bool
    note: str


def _link(interconnect: str) -> tuple[float, float]:
    return INTERCONNECT_GBPS[interconnect]


def ddp_efficiency(
    grad_bytes: float,
    world: int,
    interconnect: str,
    compute_time_per_step_s: float,
) -> CommVerdict:
    """Efficiency of DDP's per-step gradient all-reduce on this link."""
    if world <= 1:
        return CommVerdict(1.0, True, "single worker — no collective traffic")
    bw, lat = _link(interconnect)
    volume = 2.0 * (world - 1) / world * grad_bytes
    t_comm = volume / (bw * GB) + lat
    return _verdict(t_comm, compute_time_per_step_s, interconnect, volume)


def fsdp_efficiency(
    param_bytes: float,
    layers: int,
    world: int,
    interconnect: str,
    compute_time_per_step_s: float,
) -> CommVerdict:
    """Efficiency of FSDP2/ZeRO-3 per-layer all-gather + reduce-scatter."""
    if world <= 1:
        return CommVerdict(1.0, True, "single worker — no collective traffic")
    bw, lat = _link(interconnect)
    volume = 3.0 * param_bytes * (world - 1) / world
    t_comm = volume / (bw * GB) + 3.0 * layers * lat
    return _verdict(t_comm, compute_time_per_step_s, interconnect, volume)


def _verdict(t_comm: float, t_compute: float, interconnect: str, volume: float) -> CommVerdict:
    if t_compute <= 0:
        # No throughput estimate available: fall back to a link-class gate.
        ok = interconnect not in ("wan",)
        return CommVerdict(
            1.0 if ok else 0.0,
            ok,
            "no compute-time estimate — verdict from link class only",
        )
    exposed = max(0.0, t_comm - OVERLAP_FRACTION * t_compute)
    e = t_compute / (t_compute + exposed)
    mb = volume / 1e6
    if e < E_REJECT:
        note = (
            f"scaling efficiency {e:.2f} < {E_REJECT} on {interconnect} "
            f"({mb:.0f} MB/step) — communication dominates"
        )
        return CommVerdict(round(e, 2), False, note)
    if e < E_WARN:
        note = f"scaling efficiency {e:.2f} on {interconnect} — workable but wasteful"
    else:
        note = f"scaling efficiency {e:.2f} on {interconnect}"
    return CommVerdict(round(e, 2), True, note)
