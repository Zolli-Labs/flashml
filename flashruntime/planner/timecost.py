"""Static time and cost estimation.

Transformer training throughput uses the standard FLOPs identity:
    FLOPs per token (forward + backward) ≈ 6 · parameters
    tokens/s/GPU ≈ MFU · peak_FLOPS / (6 · parameters)
with an assumed MFU (catalog.ASSUMED_MFU). Multiplied by GPU count and
scaling efficiency for the fleet rate; penalized for slow strategy knobs.
LoRA/QLoRA save optimizer *memory*, not backward FLOPs — activation
gradients still flow through every frozen layer, so 6·P stands.

Cost = wall-clock × GPUs × hourly rate (when the user provided one).
Everything here is `basis: static`; a profiling run replaces these numbers
with measurements, and honest absence beats invented precision — when
inputs are missing, estimates are omitted, never guessed silently.
"""

from __future__ import annotations

from flashruntime.planner.catalog import ASSUMED_MFU
from flashruntime.protocol.plan_v1alpha1 import Estimate

# Multiplicative step-time penalties for strategy knobs. [assumption]
PENALTY_ACT_CKPT = 1.30  # recompute in backward
PENALTY_QLORA = 1.35  # dequantization overhead
PENALTY_CPU_OFFLOAD = 4.0  # optimizer step across PCIe — order-of-magnitude, profile it


def tokens_per_second_per_gpu(params: float, gpu_tflops: float) -> float:
    """Static single-GPU throughput from the 6·P FLOPs identity."""
    if gpu_tflops <= 0 or params <= 0:
        return 0.0
    return ASSUMED_MFU * gpu_tflops * 1e12 / (6.0 * params)


def compute_time_per_micro_step_s(
    params: float, seq_len: int, micro_batch: int, gpu_tflops: float
) -> float:
    """Seconds of pure compute for one micro-batch on one GPU (feeds the
    communication overlap model)."""
    tps = tokens_per_second_per_gpu(params, gpu_tflops)
    if tps <= 0:
        return 0.0
    return seq_len * micro_batch / tps


def training_time_minutes(
    params: float,
    train_tokens: float,
    gpus: int,
    gpu_tflops: float,
    scaling_efficiency: float,
    *,
    activation_checkpointing: bool = False,
    qlora: bool = False,
    cpu_offload: bool = False,
) -> Estimate | None:
    """Wall-clock estimate for the whole run, or None when inputs are missing."""
    tps = tokens_per_second_per_gpu(params, gpu_tflops)
    if tps <= 0 or train_tokens <= 0:
        return None
    fleet = tps * max(1, gpus) * max(0.01, scaling_efficiency)
    penalty = 1.0
    notes = [f"MFU {ASSUMED_MFU} assumed"]
    if activation_checkpointing:
        penalty *= PENALTY_ACT_CKPT
        notes.append("activation-checkpointing recompute +30%")
    if qlora:
        penalty *= PENALTY_QLORA
        notes.append("QLoRA dequantization +35%")
    if cpu_offload:
        penalty *= PENALTY_CPU_OFFLOAD
        notes.append("CPU-offload optimizer ×4 — profile before trusting")
    minutes = train_tokens * penalty / fleet / 60.0
    return Estimate(value=round(minutes, 1), unit="min", basis="static", note="; ".join(notes))


def cost_usd(time_min: Estimate | None, gpus: int, hourly_per_gpu: float | None) -> Estimate | None:
    if time_min is None or hourly_per_gpu is None:
        return None
    usd = time_min.value / 60.0 * max(1, gpus) * hourly_per_gpu
    return Estimate(value=round(usd, 2), unit="usd", basis="static", note=time_min.note)
