"""Static per-GPU memory estimation for training strategies.

The component model (evaluation §D). All figures are *lower bounds by
construction* plus a flat overhead; the safety bands in `catalog` — not the
formulas — are what make launch decisions safe. Real memory depends on
fragmentation, kernel workspace, and framework version; an OOM in a launched
job is treated as a planner defect and logged as a regression case.

Baseline (mixed-precision AdamW, bytes per parameter):
  weights 2 (bf16) + gradients 2 + fp32 master 4 + Adam m 4 + Adam v 4 = 16
— the ZeRO paper's classic 16 bytes/param. PEFT changes *which* parameters
train: frozen weights keep their 2 (or ~0.55 quantized NF4) bytes, while the
16-byte training state applies only to the small trainable set. Sharding
divides state across N GPUs; offloading moves it to host RAM (which must
also be checked — planners that only check VRAM kill hosts).
"""

from __future__ import annotations

from dataclasses import dataclass

from flashruntime.planner.catalog import FLAT_OVERHEAD_GB
from flashruntime.planner.resolve import GB, ResolvedTransformer
from flashruntime.protocol.plan_v1alpha1 import MemoryBreakdown, TransformerFineTune

# Bytes per parameter by dtype.
_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0}
# NF4 4-bit quantized frozen weights incl. quantization constants. [assumption]
_QLORA_FROZEN_BYTES = 0.55

# Optimizer-state bytes per *trainable* parameter (fp32 master + moments).
_OPTIMIZER_BYTES = {
    "adamw": 12.0,  # 4 master + 4 m + 4 v
    "adamw_8bit": 6.0,  # 4 master + 1 m + 1 v (bitsandbytes 8-bit) [assumption]
    "sgd_momentum": 8.0,  # 4 master + 4 momentum
}

# Activation bytes per (token × hidden) per layer, bf16 + flash-attention era.
# The single least-predictable constant in this file — pessimistic on
# purpose; profiling replaces it. [assumption]
_ACT_BYTES_PER_LAYER = 18.0


@dataclass(frozen=True)
class ShardingConfig:
    """How a strategy family divides training state across `world` GPUs.

    zero1: optimizer sharded. zero2: + gradients. fsdp2/zero3: + weights
    (with a transient per-layer all-gather peak). `offload_optimizer` moves
    optimizer state to host RAM instead.
    """

    world: int = 1
    shard_weights: bool = False
    shard_gradients: bool = False
    shard_optimizer: bool = False
    offload_optimizer: bool = False


def transformer_memory(
    w: TransformerFineTune,
    rt: ResolvedTransformer,
    sharding: ShardingConfig,
    activation_checkpointing: bool,
) -> MemoryBreakdown:
    """Estimate per-GPU peak memory for one transformer training candidate."""
    n = max(1, sharding.world)
    quantized = w.method == "qlora"

    # --- weights -----------------------------------------------------------
    frozen_bytes = _QLORA_FROZEN_BYTES if quantized else _BYTES[w.precision]
    if w.method == "full":
        weights = rt.params * _BYTES[w.precision]
    else:
        # frozen base + LoRA adapters at compute precision
        weights = rt.params * frozen_bytes + rt.trainable_params * _BYTES[w.precision]
    if sharding.shard_weights:
        weights /= n

    # --- gradients + optimizer (trainable params only) ---------------------
    gradients = rt.trainable_params * _BYTES[w.precision]
    if sharding.shard_gradients:
        gradients /= n
    optimizer = rt.trainable_params * _OPTIMIZER_BYTES[w.optimizer]
    cpu_offload = 0.0
    if sharding.offload_optimizer:
        cpu_offload = optimizer / n if sharding.shard_optimizer else optimizer
        optimizer = 0.0
    elif sharding.shard_optimizer:
        optimizer /= n

    # --- activations -------------------------------------------------------
    # Full: L layers × seq × micro_batch × hidden × const.
    # With activation checkpointing only layer *inputs* (2 bytes/elem) are
    # kept plus one live layer's activations; recompute costs ~+30% step
    # time (charged in timecost). [assumption]
    tokens = w.seq_len * w.micro_batch_per_gpu
    per_layer = tokens * rt.hidden * _ACT_BYTES_PER_LAYER
    if activation_checkpointing:
        activations = rt.layers * tokens * rt.hidden * 2.0 + per_layer
    else:
        activations = rt.layers * per_layer

    # --- transient sharding peak ------------------------------------------
    # FSDP2/ZeRO-3 re-materialize one layer group's full parameters during
    # forward/backward all-gather.
    transient = 0.0
    if sharding.shard_weights:
        layer_params = rt.params / rt.layers
        transient = layer_params * _BYTES[w.precision] * 2.0

    total = weights + gradients + optimizer + activations + transient + FLAT_OVERHEAD_GB * GB

    return MemoryBreakdown(
        weights_gb=round(weights / GB, 2),
        gradients_gb=round(gradients / GB, 2),
        optimizer_gb=round(optimizer / GB, 2),
        activations_gb=round(activations / GB, 2),
        transient_gb=round(transient / GB, 2),
        overhead_gb=FLAT_OVERHEAD_GB,
        total_gb=round(total / GB, 2),
        cpu_offload_gb=round(cpu_offload / GB, 2),
        basis="static",
    )


def generic_training_memory(
    parameters_m: float,
    trainable_fraction: float,
    precision: str,
    optimizer: str,
    activation_gb: float,
    sharding: ShardingConfig,
) -> MemoryBreakdown:
    """Same component model for non-transformer PyTorch training, with
    user-supplied (or pessimistic-default) activation memory."""
    n = max(1, sharding.world)
    params = parameters_m * 1e6
    trainable = params * trainable_fraction

    weights = params * _BYTES[precision]
    if sharding.shard_weights:
        weights /= n
    gradients = trainable * _BYTES[precision]
    if sharding.shard_gradients:
        gradients /= n
    optimizer_b = trainable * _OPTIMIZER_BYTES[optimizer]
    cpu_offload = 0.0
    if sharding.offload_optimizer:
        cpu_offload = optimizer_b / n if sharding.shard_optimizer else optimizer_b
        optimizer_b = 0.0
    elif sharding.shard_optimizer:
        optimizer_b /= n

    total = weights + gradients + optimizer_b + activation_gb * GB + FLAT_OVERHEAD_GB * GB
    return MemoryBreakdown(
        weights_gb=round(weights / GB, 2),
        gradients_gb=round(gradients / GB, 2),
        optimizer_gb=round(optimizer_b / GB, 2),
        activations_gb=round(activation_gb, 2),
        transient_gb=0.0,
        overhead_gb=FLAT_OVERHEAD_GB,
        total_gb=round(total / GB, 2),
        cpu_offload_gb=round(cpu_offload / GB, 2),
        basis="static",
    )
