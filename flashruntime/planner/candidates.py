"""Candidate generation and evaluation.

The planner does not search an open space — it walks a *curated menu* of
strategy families per workload class (ADR-0003) and evaluates each candidate
with the static estimators:

  transformer/pytorch training : single_gpu · ddp · fsdp2 · zero3_cpu_offload
                                 (+ QLoRA variants when quantization is allowed)
  classical ML                 : local_process · sharded_partial_fit (Mode A)
  independent tasks            : lease_tasks (Mode A) · local_sequential

Evaluation per candidate: memory fit (with the safety bands), host-RAM fit
for offload, communication efficiency for the topology, then time/cost and
the user's hard constraints. Every elimination keeps its arithmetic — the
rejections are half the product.

When `activation_checkpointing` is unset, the planner tries without first
and enables it only if that is what makes the candidate fit (recompute costs
~+30% step time, so it is never on by default).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flashruntime.planner import comm, memory, timecost
from flashruntime.planner.catalog import AUTO_OK_FRACTION, HARD_LIMIT_FRACTION
from flashruntime.planner.memory import ShardingConfig
from flashruntime.planner.resolve import (
    ResolvedGPU,
    ResolvedTransformer,
    resolve_gpu,
    resolve_transformer,
)
from flashruntime.protocol.plan_v1alpha1 import (
    CandidateVerdict,
    ClassicalML,
    Estimate,
    IndependentTasks,
    LibraryRef,
    MemoryBreakdown,
    PlanRequest,
    PyTorchTraining,
    TransformerFineTune,
)

_PRECISION_BYTES = {"bf16": 2.0, "fp16": 2.0, "fp32": 4.0}


@dataclass
class Evaluated:
    """One evaluated candidate: its public verdict plus everything the
    selector needs to rank it and to mint a StrategyPlan from the winner."""

    verdict: CandidateVerdict
    workload_mode: str = "coordinated_training"
    launcher: str = "torchrun"
    gpus_per_worker: int = 1
    colocated: bool = True
    precision: str | None = None
    quantization: str | None = None
    peft: str | None = None
    micro_batch: int | None = None
    grad_accum: int | None = None
    act_ckpt: bool | None = None
    offload: str = "none"
    libraries: list[LibraryRef] = field(default_factory=list)
    headroom_gb: float = 0.0
    vram_deficit_gb: float = 0.0  # >0 when memory-infeasible: how far off it was


# ---------------------------------------------------------------------------
# Library stacks — the "which libraries, in which roles" answer.
# ---------------------------------------------------------------------------


def _training_stack(
    family: str, world: int, method: str, quantized: bool
) -> list[LibraryRef]:
    libs: list[LibraryRef] = [LibraryRef(name="pytorch", role="framework", purpose="training loop, autograd")]
    if family == "ddp" and world > 1:
        libs.append(LibraryRef(name="torch DDP", role="strategy", purpose="per-step gradient all-reduce"))
    elif family == "fsdp2":
        libs.append(
            LibraryRef(
                name="torch FSDP2 (fully_shard)",
                role="strategy",
                purpose="parameter/gradient/optimizer sharding on DTensor",
            )
        )
    elif family == "zero3_cpu_offload":
        libs.append(
            LibraryRef(
                name="DeepSpeed ZeRO-3 + CPU offload",
                role="strategy",
                purpose="full sharding with optimizer state in host RAM",
            )
        )
    if method in ("lora", "qlora") or quantized:
        libs.append(LibraryRef(name="transformers", role="workload", purpose="model + tokenizer + Trainer recipe"))
        libs.append(LibraryRef(name="peft", role="workload", purpose="LoRA adapter injection"))
    if quantized:
        libs.append(LibraryRef(name="bitsandbytes", role="workload", purpose="NF4 4-bit frozen-weight quantization"))
    libs.append(
        LibraryRef(
            name="torchrun" if world > 1 else "local process",
            role="launcher",
            purpose="worker-group launch + restart on membership change" if world > 1 else "single-process launch",
        )
    )
    libs.append(
        LibraryRef(name="pytorch DCP", role="checkpoint", purpose="parallel save/load with resharding on restore")
    )
    libs.append(
        LibraryRef(
            name="flashruntime",
            role="runtime",
            purpose="event ledger, checkpoint catalog, failure taxonomy, recovery",
        )
    )
    return libs


def _lease_stack(needs_docker: bool = True) -> list[LibraryRef]:
    libs = [
        LibraryRef(
            name="flashruntime leases",
            role="launcher",
            purpose="pull-based task leases: claim → heartbeat → idempotent commit",
        )
    ]
    if needs_docker:
        libs.append(LibraryRef(name="docker (via flashnode)", role="executor", purpose="sandboxed task execution on devices"))
    libs.append(LibraryRef(name="flashruntime", role="runtime", purpose="event ledger, retries, artifact commit"))
    return libs


# ---------------------------------------------------------------------------
# Transformer fine-tuning (and generic PyTorch via the same skeleton)
# ---------------------------------------------------------------------------


def _world_sizes(gpus: int) -> list[int]:
    """1, powers of two, and the full count — smallest first (the planner
    prefers the smallest world size that meets the deadline)."""
    sizes = {1}
    n = 2
    while n < gpus:
        sizes.add(n)
        n *= 2
    if gpus >= 1:
        sizes.add(gpus)
    return sorted(sizes)


_FAMILY_SHARDING = {
    "single_gpu": ShardingConfig(),
    "ddp": ShardingConfig(),
    "fsdp2": ShardingConfig(shard_weights=True, shard_gradients=True, shard_optimizer=True),
    "zero3_cpu_offload": ShardingConfig(
        shard_weights=True, shard_gradients=True, shard_optimizer=True, offload_optimizer=True
    ),
}


def transformer_candidates(req: PlanRequest) -> tuple[list[Evaluated], list[str]]:
    w = req.workload
    assert isinstance(w, TransformerFineTune)
    rt = resolve_transformer(w)
    gpu = resolve_gpu(req.resources)
    notes = rt.notes + gpu.notes

    if req.resources.gpus < 1:
        return [], notes + ["transformer fine-tuning requires at least one GPU in v1"]

    variants: list[tuple[str, TransformerFineTune, ResolvedTransformer]] = [("", w, rt)]
    if w.method == "lora" and req.objective.allow_quantization:
        qw = w.model_copy(update={"method": "qlora"})
        variants.append(("qlora_", qw, resolve_transformer(qw)))

    out: list[Evaluated] = []
    for prefix, wv, rtv in variants:
        for world in _world_sizes(req.resources.gpus):
            families = ["single_gpu"] if world == 1 else ["ddp", "fsdp2", "zero3_cpu_offload"]
            for family in families:
                if family == "zero3_cpu_offload" and not req.objective.allow_cpu_offload:
                    continue
                if family == "fsdp2" and wv.method == "qlora":
                    continue  # quantized frozen weights + full param sharding: unsupported combo in v1
                out.append(_evaluate_training(req, wv, rtv, gpu, f"{prefix}{family}", family, world))
    return out, notes


def _evaluate_training(
    req: PlanRequest,
    w: TransformerFineTune,
    rt: ResolvedTransformer,
    gpu: ResolvedGPU,
    name: str,
    family: str,
    world: int,
) -> Evaluated:
    reasons: list[str] = []
    base_sharding = _FAMILY_SHARDING[family]
    sharding = ShardingConfig(
        world=world,
        shard_weights=base_sharding.shard_weights,
        shard_gradients=base_sharding.shard_gradients,
        shard_optimizer=base_sharding.shard_optimizer,
        offload_optimizer=base_sharding.offload_optimizer,
    )

    # -- memory, with auto activation-checkpointing -------------------------
    act = w.activation_checkpointing
    mem = memory.transformer_memory(w, rt, sharding, activation_checkpointing=bool(act))
    if act is None and mem.total_gb > gpu.vram_gb * HARD_LIMIT_FRACTION:
        retry = memory.transformer_memory(w, rt, sharding, activation_checkpointing=True)
        if retry.total_gb <= gpu.vram_gb * HARD_LIMIT_FRACTION:
            mem, act = retry, True
            name = f"{name}+ackpt"
            reasons.append("activation checkpointing enabled to fit (recompute ≈ +30% step time)")
        else:
            act = False
    act = bool(act)

    hard_cap = gpu.vram_gb * HARD_LIMIT_FRACTION
    auto_cap = gpu.vram_gb * AUTO_OK_FRACTION
    deficit = max(0.0, mem.total_gb - hard_cap)
    if deficit > 0:
        reasons.append(
            f"estimated peak {mem.total_gb} GB/GPU exceeds {gpu.vram_gb} GB × "
            f"{HARD_LIMIT_FRACTION} = {hard_cap:.1f} GB — infeasible"
        )
        return _reject(name, family, world, mem, reasons, deficit=deficit)
    profiling_required = mem.total_gb > auto_cap
    reasons.append(
        f"estimated peak {mem.total_gb} GB/GPU of {gpu.vram_gb} GB "
        f"({'requires profiling before launch' if profiling_required else 'fits with static margin'})"
    )

    ram_budget = req.resources.cpu_ram_gb * 0.8
    if mem.cpu_offload_gb > ram_budget:
        reasons.append(
            f"offloaded optimizer state {mem.cpu_offload_gb} GB exceeds 80% of host RAM "
            f"({req.resources.cpu_ram_gb} GB) — infeasible"
        )
        return _reject(name, family, world, mem, reasons)

    # -- communication ------------------------------------------------------
    interconnect = _effective_interconnect(req, world)
    t_micro = timecost.compute_time_per_micro_step_s(
        rt.params, w.seq_len, w.micro_batch_per_gpu, gpu.bf16_tflops
    )
    prec_bytes = _PRECISION_BYTES[w.precision]
    if family == "ddp" or family == "single_gpu":
        cv = comm.ddp_efficiency(
            rt.trainable_params * prec_bytes, world, interconnect, t_micro * w.grad_accum
        )
    else:
        cv = comm.fsdp_efficiency(rt.params * prec_bytes, rt.layers, world, interconnect, t_micro)
    reasons.append(cv.note)
    if not cv.feasible:
        return _reject(name, family, world, mem, reasons, efficiency=cv.efficiency)

    # -- time, cost, hard constraints ---------------------------------------
    quantized = w.method == "qlora"
    offload = family == "zero3_cpu_offload"
    t = timecost.training_time_minutes(
        rt.params,
        (w.train_tokens_m or 0) * 1e6,
        world,
        gpu.bf16_tflops,
        cv.efficiency,
        activation_checkpointing=act,
        qlora=quantized,
        cpu_offload=offload,
    )
    c = timecost.cost_usd(t, world, req.resources.hourly_cost_usd_per_gpu)

    status = "feasible"
    if t is not None and req.objective.deadline_minutes and t.value > req.objective.deadline_minutes:
        status = "rejected_policy"
        reasons.append(
            f"estimated {t.value:.0f} min misses the {req.objective.deadline_minutes:.0f} min deadline"
        )
    if c is not None and req.objective.max_cost_usd and c.value > req.objective.max_cost_usd:
        status = "rejected_policy"
        reasons.append(f"estimated ${c.value:.2f} exceeds budget ${req.objective.max_cost_usd:.2f}")
    if t is not None and status == "feasible":
        reasons.append(f"estimated {t.value:.0f} min on {world} GPU(s)" + (f", ${c.value:.2f}" if c else ""))

    verdict = CandidateVerdict(
        name=name,
        strategy_family=family,
        workers=world,
        status=status,
        memory=mem,
        est_time_min=t,
        est_cost_usd=c,
        scaling_efficiency=cv.efficiency,
        profiling_required=profiling_required,
        reasons=reasons,
    )
    return Evaluated(
        verdict=verdict,
        workload_mode="local" if world == 1 else "coordinated_training",
        launcher="torchrun" if world > 1 else "local",
        colocated=world <= max(1, req.resources.gpus // req.resources.hosts),
        precision=w.precision,
        quantization="nf4" if quantized else None,
        peft=f"lora(r={w.lora_rank})" if w.method in ("lora", "qlora") else None,
        micro_batch=w.micro_batch_per_gpu,
        grad_accum=w.grad_accum,
        act_ckpt=act,
        offload="optimizer_cpu" if offload else "none",
        libraries=_training_stack(family, world, w.method, quantized),
        headroom_gb=round(gpu.vram_gb - mem.total_gb, 2),
    )


def _effective_interconnect(req: PlanRequest, world: int) -> str:
    """A worker group that fits on one host uses that host's local link;
    only groups spanning hosts pay the cluster interconnect."""
    gpus_per_host = max(1, req.resources.gpus // req.resources.hosts)
    if req.resources.hosts == 1 or world <= gpus_per_host:
        return (
            req.resources.interconnect
            if req.resources.interconnect.startswith("same_host")
            else "same_host_pcie"
        )
    return req.resources.interconnect


def _reject(
    name: str,
    family: str,
    world: int,
    mem: MemoryBreakdown,
    reasons: list[str],
    *,
    deficit: float = 0.0,
    efficiency: float | None = None,
) -> Evaluated:
    return Evaluated(
        verdict=CandidateVerdict(
            name=name,
            strategy_family=family,
            workers=world,
            status="infeasible",
            memory=mem,
            scaling_efficiency=efficiency,
            reasons=reasons,
        ),
        vram_deficit_gb=round(deficit, 2),
    )


# ---------------------------------------------------------------------------
# Generic PyTorch training — same skeleton, user-supplied activation memory
# ---------------------------------------------------------------------------

_DEFAULT_ACTIVATION_GB = 2.0  # pessimistic default when unspecified [assumption]


def pytorch_candidates(req: PlanRequest) -> tuple[list[Evaluated], list[str]]:
    w = req.workload
    assert isinstance(w, PyTorchTraining)
    notes: list[str] = []
    if req.resources.gpus < 1:
        return [], ["pytorch_training requires at least one GPU in v1 (CPU training: use classical_ml or independent_tasks)"]
    gpu = resolve_gpu(req.resources)
    notes.extend(gpu.notes)
    act_gb = w.activation_gb_per_gpu or _DEFAULT_ACTIVATION_GB
    if w.activation_gb_per_gpu is None:
        notes.append(f"activation memory defaulted to {_DEFAULT_ACTIVATION_GB} GB/GPU [assumption]")

    out: list[Evaluated] = []
    for world in _world_sizes(req.resources.gpus):
        families = ["single_gpu"] if world == 1 else ["ddp", "fsdp2"]
        for family in families:
            base = _FAMILY_SHARDING[family]
            sharding = ShardingConfig(
                world=world,
                shard_weights=base.shard_weights,
                shard_gradients=base.shard_gradients,
                shard_optimizer=base.shard_optimizer,
            )
            mem = memory.generic_training_memory(
                w.parameters_m, w.trainable_fraction, w.precision, w.optimizer, act_gb, sharding
            )
            reasons: list[str] = []
            gpu_cap = gpu.vram_gb * HARD_LIMIT_FRACTION
            if mem.total_gb > gpu_cap:
                reasons.append(f"estimated peak {mem.total_gb} GB/GPU exceeds {gpu_cap:.1f} GB cap — infeasible")
                out.append(_reject(family, family, world, mem, reasons, deficit=mem.total_gb - gpu_cap))
                continue
            profiling = mem.total_gb > gpu.vram_gb * AUTO_OK_FRACTION
            reasons.append(f"estimated peak {mem.total_gb} GB/GPU of {gpu.vram_gb} GB")

            interconnect = _effective_interconnect(req, world)
            prec = _PRECISION_BYTES[w.precision]
            grad_bytes = w.parameters_m * 1e6 * w.trainable_fraction * prec
            # No FLOPs identity for arbitrary models — comm verdict falls back
            # to the link-class gate when compute time is unknown.
            cv = (
                comm.ddp_efficiency(grad_bytes, world, interconnect, 0.0)
                if family in ("ddp", "single_gpu")
                else comm.fsdp_efficiency(w.parameters_m * 1e6 * prec, 32, world, interconnect, 0.0)
            )
            reasons.append(cv.note)
            if not cv.feasible:
                out.append(_reject(family, family, world, mem, reasons, efficiency=cv.efficiency))
                continue

            t = None
            if w.est_train_gpu_hours:
                minutes = w.est_train_gpu_hours * 60.0 / (world * max(0.01, cv.efficiency))
                t = Estimate(value=round(minutes, 1), unit="min", basis="static", note="scaled from user single-GPU estimate")
            c = timecost.cost_usd(t, world, req.resources.hourly_cost_usd_per_gpu)
            status = "feasible"
            if t and req.objective.deadline_minutes and t.value > req.objective.deadline_minutes:
                status = "rejected_policy"
                reasons.append(f"estimated {t.value:.0f} min misses the deadline")
            if c and req.objective.max_cost_usd and c.value > req.objective.max_cost_usd:
                status = "rejected_policy"
                reasons.append(f"estimated ${c.value:.2f} exceeds budget")

            out.append(
                Evaluated(
                    verdict=CandidateVerdict(
                        name=family if world == 1 else f"{family}_x{world}",
                        strategy_family=family,
                        workers=world,
                        status=status,
                        memory=mem,
                        est_time_min=t,
                        est_cost_usd=c,
                        scaling_efficiency=cv.efficiency,
                        profiling_required=profiling,
                        reasons=reasons,
                    ),
                    workload_mode="local" if world == 1 else "coordinated_training",
                    launcher="torchrun" if world > 1 else "local",
                    precision=w.precision,
                    libraries=_training_stack(family, world, "full", False),
                    headroom_gb=round(gpu.vram_gb - mem.total_gb, 2),
                )
            )
    return out, notes


# ---------------------------------------------------------------------------
# Classical ML and independent tasks (Mode 0 / Mode A)
# ---------------------------------------------------------------------------


def classical_candidates(req: PlanRequest) -> tuple[list[Evaluated], list[str]]:
    w = req.workload
    assert isinstance(w, ClassicalML)
    out: list[Evaluated] = []
    # Working-set rule of thumb: sklearn copies cost ~3× the dataset. [assumption]
    fits_ram = w.dataset_mb * 3 / 1000.0 <= req.resources.cpu_ram_gb * 0.8
    reasons = [
        f"dataset {w.dataset_mb:.0f} MB × 3 working-set ≈ {w.dataset_mb * 3 / 1000:.1f} GB vs "
        f"{req.resources.cpu_ram_gb} GB host RAM"
    ]
    out.append(
        Evaluated(
            verdict=CandidateVerdict(
                name="local_process",
                strategy_family="local_process",
                workers=1,
                status="feasible" if fits_ram else "infeasible",
                reasons=reasons + ([] if fits_ram else ["dataset does not fit in host RAM"]),
            ),
            workload_mode="local",
            launcher="local",
            gpus_per_worker=0,
            libraries=[
                LibraryRef(name=w.library, role="workload", purpose=f"{w.algorithm} fit/predict"),
                LibraryRef(name="flashruntime", role="runtime", purpose="job record + artifacts"),
            ],
        )
    )
    if w.supports_partial_fit:
        shards = max(2, min(16, int(w.dataset_mb // 500) or 2))
        out.append(
            Evaluated(
                verdict=CandidateVerdict(
                    name="sharded_partial_fit",
                    strategy_family="sharded_partial_fit",
                    workers=shards,
                    status="feasible",
                    reasons=[
                        f"{shards} data shards trained incrementally via leased tasks, "
                        "periodic reduce (the sharded K-means pattern)"
                    ],
                ),
                workload_mode="independent_tasks",
                launcher="flashruntime-leases",
                gpus_per_worker=0,
                libraries=_lease_stack() + [LibraryRef(name=w.library, role="workload", purpose="partial_fit per shard")],
            )
        )
    return out, []


def tasks_candidates(req: PlanRequest) -> tuple[list[Evaluated], list[str]]:
    w = req.workload
    assert isinstance(w, IndependentTasks)
    r = req.resources
    notes: list[str] = []

    if w.needs_gpu:
        slots = r.gpus
        gpu_note = f"{r.gpus} GPU slot(s)"
        if r.gpus and w.vram_gb_per_task:
            gpu = resolve_gpu(r)
            if w.vram_gb_per_task > gpu.vram_gb * HARD_LIMIT_FRACTION:
                return [
                    _reject(
                        "lease_tasks",
                        "lease_tasks",
                        0,
                        MemoryBreakdown(total_gb=w.vram_gb_per_task),
                        [f"each task needs {w.vram_gb_per_task} GB VRAM; GPUs have {gpu.vram_gb} GB"],
                        deficit=w.vram_gb_per_task - gpu.vram_gb,
                    )
                ], notes
    else:
        slots = r.hosts * max(1, r.cpu_cores // 4)  # ~4 cores per task [assumption]
        gpu_note = f"{r.hosts} host(s) × {max(1, r.cpu_cores // 4)} concurrent task(s)"
    slots = max(1, min(slots, w.task_count))

    waves = -(-w.task_count // slots)  # ceil division
    par_min = waves * w.est_minutes_per_task
    seq_min = w.task_count * w.est_minutes_per_task

    out = [
        Evaluated(
            verdict=CandidateVerdict(
                name="lease_tasks",
                strategy_family="lease_tasks",
                workers=slots,
                status="feasible",
                est_time_min=Estimate(
                    value=round(par_min, 1),
                    unit="min",
                    basis="static",
                    note=f"{w.task_count} tasks / {slots} slots = {waves} wave(s)",
                ),
                reasons=[
                    f"{w.task_count} independent {w.task_kind} tasks over {gpu_note}",
                    "per-task failure isolation: a lost node costs one task retry, never the job",
                ],
            ),
            workload_mode="independent_tasks",
            launcher="flashruntime-leases",
            gpus_per_worker=1 if w.needs_gpu else 0,
            colocated=False,
            libraries=_lease_stack(),
        ),
        Evaluated(
            verdict=CandidateVerdict(
                name="local_sequential",
                strategy_family="local_process",
                workers=1,
                status="feasible" if slots == 1 else "rejected_dominated",
                est_time_min=Estimate(value=round(seq_min, 1), unit="min", basis="static"),
                reasons=[f"baseline: {w.task_count} tasks one after another ≈ {seq_min:.0f} min"],
            ),
            workload_mode="local",
            launcher="local",
            gpus_per_worker=1 if w.needs_gpu else 0,
            libraries=[LibraryRef(name="flashruntime", role="runtime", purpose="job record + artifacts")],
        ),
    ]
    if req.objective.deadline_minutes and par_min > req.objective.deadline_minutes:
        out[0].verdict.status = "rejected_policy"
        out[0].verdict.reasons.append(
            f"even at {slots}-wide parallelism, {par_min:.0f} min misses the deadline"
        )
    return out, notes


def generate(req: PlanRequest) -> tuple[list[Evaluated], list[str]]:
    """Dispatch on workload kind → (evaluated candidates, resolution notes)."""
    kind = req.workload.kind
    if kind == "transformer_finetune":
        return transformer_candidates(req)
    if kind == "pytorch_training":
        return pytorch_candidates(req)
    if kind == "classical_ml":
        return classical_candidates(req)
    if kind == "independent_tasks":
        return tasks_candidates(req)
    raise ValueError(f"unsupported workload kind: {kind}")
