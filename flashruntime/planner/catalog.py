"""Reference data the estimators consult: known models, GPU classes, links.

Everything here is an *approximation with a source-of-truth elsewhere* —
model cards, vendor datasheets, measured benchmarks. Values are deliberately
conservative; the ledger (measured runs) is meant to override them over
time. Adding an entry is the supported way to extend planner coverage — no
code changes required.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """Structural facts about a known transformer, for estimation only."""

    parameters_b: float
    hidden_size: int
    num_layers: int


# Keys are matched case-insensitively on the trailing path component, so
# "Qwen/Qwen2.5-7B", "qwen2.5-7b" and "Qwen2.5-7B-Instruct" all resolve.
# Parameter counts are the published totals (approximate).
MODEL_CATALOG: dict[str, ModelInfo] = {
    "qwen2.5-0.5b": ModelInfo(0.49, 896, 24),
    "qwen2.5-1.5b": ModelInfo(1.54, 1536, 28),
    "qwen2.5-3b": ModelInfo(3.09, 2048, 36),
    "qwen2.5-7b": ModelInfo(7.62, 3584, 28),
    "qwen2.5-14b": ModelInfo(14.7, 5120, 48),
    "qwen2.5-32b": ModelInfo(32.8, 5120, 64),
    "qwen2.5-72b": ModelInfo(72.7, 8192, 80),
    "llama-3.1-8b": ModelInfo(8.03, 4096, 32),
    "llama-3.1-70b": ModelInfo(70.6, 8192, 80),
    "llama-3.2-1b": ModelInfo(1.24, 2048, 16),
    "llama-3.2-3b": ModelInfo(3.21, 3072, 28),
    "mistral-7b": ModelInfo(7.25, 4096, 32),
    "gemma-2-9b": ModelInfo(9.24, 3584, 42),
    "gpt2": ModelInfo(0.124, 768, 12),
}


def lookup_model(name: str) -> ModelInfo | None:
    """Resolve a model name to catalog info; forgiving about org prefixes
    and -Instruct/-Chat suffixes."""
    tail = name.strip().lower().split("/")[-1]
    for suffix in ("-instruct", "-chat", "-base", "-hf"):
        tail = tail.removesuffix(suffix)
    return MODEL_CATALOG.get(tail)


def derive_transformer_shape(parameters_b: float) -> tuple[int, int]:
    """Guess (hidden_size, num_layers) from parameter count when the user
    gave neither. Dense-transformer typicals; the activation estimate built
    on this is labeled as derived. [assumption]"""
    table = [
        (0.2, (768, 12)),
        (0.7, (1024, 24)),
        (2.0, (2048, 24)),
        (4.0, (3072, 28)),
        (9.0, (4096, 32)),
        (16.0, (5120, 40)),
        (40.0, (6656, 60)),
        (90.0, (8192, 80)),
    ]
    for ceiling, shape in table:
        if parameters_b <= ceiling:
            return shape
    return (12288, 96)


@dataclass(frozen=True)
class GPUInfo:
    """One GPU class. `bf16_tflops` is a *conservative dense* figure — never
    the sparsity marketing number."""

    vram_gb: float
    bf16_tflops: float


GPU_CATALOG: dict[str, GPUInfo] = {
    "h100": GPUInfo(80, 700),
    "a100-80gb": GPUInfo(80, 312),
    "a100-40gb": GPUInfo(40, 312),
    "a100": GPUInfo(40, 312),
    "l40s": GPUInfo(48, 362),
    "rtx4090": GPUInfo(24, 165),
    "rtx3090": GPUInfo(24, 71),
    "a10": GPUInfo(24, 125),
    "l4": GPUInfo(24, 121),
    "v100": GPUInfo(16, 112),
    "t4": GPUInfo(16, 65),
}


def lookup_gpu(gpu_type: str | None) -> GPUInfo | None:
    if not gpu_type:
        return None
    return GPU_CATALOG.get(gpu_type.strip().lower())


# Effective point-to-point bandwidth per interconnect class, GB/s, and a
# per-collective latency floor in seconds. Conservative effective figures,
# not line rates. [assumption — FlashNode's network benchmark replaces these
# with measurements per pool]
INTERCONNECT_GBPS: dict[str, tuple[float, float]] = {
    "same_host_nvlink": (250.0, 0.00001),
    "same_host_pcie": (20.0, 0.00002),
    "multi_node_ib": (25.0, 0.00005),
    "multi_node_100g": (10.0, 0.0001),
    "multi_node_10g": (1.1, 0.0003),
    "wan": (0.05, 0.03),
}

# Model-FLOPs-utilization assumed for static throughput estimates.
# Real MFU varies 0.2–0.5 by model/kernel/stack; 0.30 is a defensible
# middle. [assumption — profiling replaces this]
ASSUMED_MFU = 0.30

# Flat per-GPU overhead: CUDA context + NCCL buffers + framework workspace.
# [assumption]
FLAT_OVERHEAD_GB = 2.5

# Static-only safety bands (ADR-0003 / evaluation §D): auto-launch below
# AUTO_OK, require profiling in the band, infeasible above HARD_LIMIT.
AUTO_OK_FRACTION = 0.80
HARD_LIMIT_FRACTION = 0.95
