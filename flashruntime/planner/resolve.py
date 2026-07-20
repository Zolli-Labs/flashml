"""Resolve a PlanRequest into the concrete numbers the estimators need.

Turns "Qwen/Qwen2.5-7B, LoRA r=16, 4×24 GB" into parameter counts, model
shape, trainable-parameter counts, and per-GPU VRAM — recording which values
were user-given, catalog-known, or derived, so estimates can say so.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flashruntime.planner import catalog
from flashruntime.protocol.plan_v1alpha1 import PlanRequest, Resources, TransformerFineTune

GB = 1e9  # this module talks in GB = 1e9 bytes throughout


@dataclass
class ResolvedTransformer:
    """A TransformerFineTune with every blank filled in."""

    params: float  # absolute parameter count
    hidden: int
    layers: int
    trainable_params: float  # LoRA adapters, or all params for full FT
    shape_derived: bool  # True when hidden/layers came from a heuristic
    notes: list[str] = field(default_factory=list)


def resolve_transformer(w: TransformerFineTune) -> ResolvedTransformer:
    """Fill parameter count and shape from (in priority order) user input,
    the model catalog, or the size-class heuristic."""
    notes: list[str] = []
    info = catalog.lookup_model(w.model)

    if w.parameters_b is not None:
        params_b = w.parameters_b
    elif info is not None:
        params_b = info.parameters_b
        notes.append(f"parameter count {params_b}B taken from model catalog for '{w.model}'")
    else:
        raise ValueError(
            f"model '{w.model}' is not in the planner catalog — set workload.parameters_b "
            "(billions of parameters) explicitly"
        )

    shape_derived = False
    if w.hidden_size and w.num_layers:
        hidden, layers = w.hidden_size, w.num_layers
    elif info is not None:
        hidden, layers = info.hidden_size, info.num_layers
    else:
        hidden, layers = catalog.derive_transformer_shape(params_b)
        shape_derived = True
        notes.append(
            f"model shape derived from size class ({hidden} hidden × {layers} layers) [assumption]"
        )

    params = params_b * 1e9
    if w.method == "full":
        trainable = params
    else:
        # LoRA on the four attention projections (q,k,v,o), the common
        # default: each gets A(h×r)+B(r×h) = 2·h·r params → 8·h·r per layer.
        # [assumption: adapters on attention only; all-linear LoRA is ~2–3×]
        trainable = 8.0 * hidden * w.lora_rank * layers
        notes.append(
            f"LoRA trainable params ≈ {trainable / 1e6:.1f}M "
            f"(8·hidden·rank·layers, attention projections) [assumption]"
        )

    return ResolvedTransformer(
        params=params,
        hidden=hidden,
        layers=layers,
        trainable_params=trainable,
        shape_derived=shape_derived,
        notes=notes,
    )


@dataclass
class ResolvedGPU:
    """Per-GPU capability actually used by the estimators."""

    vram_gb: float
    bf16_tflops: float
    tflops_known: bool
    notes: list[str] = field(default_factory=list)


def resolve_gpu(r: Resources) -> ResolvedGPU:
    """VRAM: user value wins, then catalog. Throughput: catalog only —
    without it, time/cost estimates are skipped rather than invented."""
    info = catalog.lookup_gpu(r.gpu_type)
    notes: list[str] = []

    if r.vram_gb is not None:
        vram = r.vram_gb
    elif info is not None:
        vram = info.vram_gb
        notes.append(f"VRAM {vram} GB taken from GPU catalog for '{r.gpu_type}'")
    else:
        raise ValueError(
            f"unknown GPU type '{r.gpu_type}' — set resources.vram_gb explicitly"
        )

    if info is not None:
        return ResolvedGPU(vram, info.bf16_tflops, True, notes)
    notes.append("GPU throughput unknown — time/cost estimates unavailable")
    return ResolvedGPU(vram, 0.0, False, notes)


def request_digest(req: PlanRequest, planner_version: str) -> str:
    """Stable content hash over the request + planner version — the identity
    of a planning decision (same inputs ⇒ same digest ⇒ same plan)."""
    import hashlib

    payload = req.model_dump_json() + planner_version
    return hashlib.sha256(payload.encode()).hexdigest()[:16]
