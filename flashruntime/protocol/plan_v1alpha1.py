"""FlashML planning protocol, version v1alpha1.

Wire-visible models for the strategy planner: what a user *asks for*
(`PlanRequest` = workload + resources + objective) and what the planner
*answers* (`PlanReport` = a selected `StrategyPlan` plus every candidate
verdict with its arithmetic). These schemas are the contract between the
SDK/CLI, FlashML Cloud, and — later — the execution layer that compiles a
`StrategyPlan` into a running job.

Design rules (ADR-0003):
- Backend-neutral: nothing here requires importing torch/ray/transformers to
  parse. Strategy families and launchers are *names*; strategy compilers
  translate them into real configuration at execution time.
- Every numeric estimate carries a `basis` (`static` | `profiled` | `ledger`)
  so consumers can tell arithmetic from measurement.
- Explanations — including *rejected* candidates — are part of the contract,
  not decoration.
- Additive changes only within v1alpha1; breaking changes go to a new module.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field

PLAN_API_VERSION = "flashml.dev/v1alpha1"


# ---------------------------------------------------------------------------
# Workloads — what the user wants to run.
#
# Classification is structural (what the computation looks like), never
# label-based: "LoRA" says which parameters train, not which distributed
# strategy to use. Each workload kind carries exactly the fields the
# estimators need.
# ---------------------------------------------------------------------------


class TransformerFineTune(BaseModel):
    """Fine-tune a transformer language model (full, LoRA, or QLoRA).

    `parameters_b` may be omitted for models in the planner's small catalog
    (e.g. "Qwen/Qwen2.5-7B"); otherwise it is required. `hidden_size` /
    `num_layers` improve the activation estimate; when absent the planner
    derives a typical shape from the parameter count and labels the estimate
    accordingly.
    """

    kind: Literal["transformer_finetune"] = "transformer_finetune"
    model: str = Field(description="Model name, e.g. 'Qwen/Qwen2.5-7B'")
    parameters_b: float | None = Field(
        default=None, gt=0, description="Total parameters in billions"
    )
    method: Literal["full", "lora", "qlora"] = "lora"
    lora_rank: int = Field(default=16, ge=1, le=256)
    precision: Literal["bf16", "fp16", "fp32"] = "bf16"
    optimizer: Literal["adamw", "adamw_8bit", "sgd_momentum"] = "adamw"
    seq_len: int = Field(default=2048, ge=64)
    micro_batch_per_gpu: int = Field(default=1, ge=1)
    grad_accum: int = Field(default=8, ge=1)
    activation_checkpointing: bool | None = Field(
        default=None, description="None = planner decides"
    )
    hidden_size: int | None = Field(default=None, ge=64)
    num_layers: int | None = Field(default=None, ge=1)
    train_tokens_m: float | None = Field(
        default=None,
        gt=0,
        description="Training set size in millions of tokens; enables time/cost/deadline estimates",
    )


class PyTorchTraining(BaseModel):
    """Generic PyTorch deep-learning training (CNNs, GNNs, custom models).

    Without a transformer's regular shape the planner cannot derive
    activation memory structurally, so the user supplies (or accepts a
    pessimistic default for) `activation_gb_per_gpu`.
    """

    kind: Literal["pytorch_training"] = "pytorch_training"
    model: str = Field(default="custom", description="Informational model name")
    parameters_m: float = Field(gt=0, description="Total parameters in millions")
    trainable_fraction: float = Field(default=1.0, gt=0, le=1.0)
    precision: Literal["bf16", "fp16", "fp32"] = "fp32"
    optimizer: Literal["adamw", "adamw_8bit", "sgd_momentum"] = "adamw"
    activation_gb_per_gpu: float | None = Field(
        default=None,
        gt=0,
        description="Measured/estimated activation memory per GPU; default is a pessimistic assumption",
    )
    est_train_gpu_hours: float | None = Field(
        default=None, gt=0, description="Single-GPU wall-clock estimate; enables time/cost"
    )


class ClassicalML(BaseModel):
    """Classical machine learning (scikit-learn / XGBoost style)."""

    kind: Literal["classical_ml"] = "classical_ml"
    library: Literal["sklearn", "xgboost"] = "sklearn"
    algorithm: str = Field(default="unspecified", description="e.g. 'kmeans', 'random_forest'")
    dataset_mb: float = Field(gt=0, description="In-memory dataset size in MB")
    supports_partial_fit: bool = Field(
        default=False,
        description="True if the estimator can learn incrementally (enables sharded Mode A execution)",
    )


class IndependentTasks(BaseModel):
    """A set of tasks with no communication between them (Mode A).

    Hyperparameter search, cross-validation folds, batch inference shards,
    per-seed synthetic-data generation, evaluation suites.
    """

    kind: Literal["independent_tasks"] = "independent_tasks"
    task_kind: Literal[
        "hyperparameter_search", "batch_inference", "evaluation", "preprocessing", "other"
    ] = "hyperparameter_search"
    task_count: int = Field(ge=1)
    est_minutes_per_task: float = Field(default=10.0, gt=0)
    needs_gpu: bool = False
    vram_gb_per_task: float = Field(default=0.0, ge=0)


Workload = Annotated[
    Union[TransformerFineTune, PyTorchTraining, ClassicalML, IndependentTasks],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Resources and objective — what the user has, and what they care about.
# ---------------------------------------------------------------------------

Interconnect = Literal[
    "same_host_nvlink",  # NVLink/NVSwitch inside one machine
    "same_host_pcie",  # multiple GPUs on one machine over PCIe
    "multi_node_ib",  # InfiniBand / RoCE between machines
    "multi_node_100g",  # 100 Gb/s Ethernet
    "multi_node_10g",  # 10 Gb/s Ethernet
    "wan",  # internet-grade links (community devices)
]


class Resources(BaseModel):
    """The compute the plan may use. GPU-count 0 means CPU-only."""

    gpus: int = Field(default=0, ge=0)
    gpu_type: str | None = Field(
        default=None, description="e.g. 'A100-40GB', 'L40S', 'RTX4090'; keys the GPU catalog"
    )
    vram_gb: float | None = Field(
        default=None, gt=0, description="Per-GPU VRAM; overrides the catalog value"
    )
    hosts: int = Field(default=1, ge=1, description="How many machines the GPUs are spread over")
    interconnect: Interconnect = "same_host_pcie"
    cpu_ram_gb: float = Field(default=32.0, gt=0, description="Host RAM per machine")
    cpu_cores: int = Field(default=8, ge=1)
    hourly_cost_usd_per_gpu: float | None = Field(default=None, ge=0)


class Objective(BaseModel):
    """Constraints and preference. Hard constraints reject plans; the mode
    orders the survivors."""

    mode: Literal["cheapest", "fastest", "balanced", "reliable"] = "balanced"
    max_cost_usd: float | None = Field(default=None, gt=0)
    deadline_minutes: float | None = Field(default=None, gt=0)
    allow_quantization: bool = True
    allow_cpu_offload: bool = True
    allow_nvme_offload: bool = False


class PlanRequest(BaseModel):
    """The planner's input: intent + resources + objective."""

    apiVersion: str = PLAN_API_VERSION
    kind: Literal["PlanRequest"] = "PlanRequest"
    workload: Workload
    resources: Resources = Field(default_factory=Resources)
    objective: Objective = Field(default_factory=Objective)


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------

EstimateBasis = Literal["static", "profiled", "ledger"]


class Estimate(BaseModel):
    """A number that admits where it came from."""

    value: float
    unit: str
    basis: EstimateBasis = "static"
    note: str = ""


class MemoryBreakdown(BaseModel):
    """Per-GPU peak memory estimate in GB (1 GB = 1e9 bytes), by component.

    `transient_gb` covers FSDP/ZeRO-3 per-layer all-gather peaks;
    `cpu_offload_gb` is host RAM claimed by offloaded state (checked against
    `Resources.cpu_ram_gb` — planners that only check VRAM kill hosts).
    """

    weights_gb: float = 0.0
    gradients_gb: float = 0.0
    optimizer_gb: float = 0.0
    activations_gb: float = 0.0
    transient_gb: float = 0.0
    overhead_gb: float = 0.0
    total_gb: float = 0.0
    cpu_offload_gb: float = 0.0
    basis: EstimateBasis = "static"


class LibraryRef(BaseModel):
    """One library the selected strategy is built on, and its role."""

    name: str
    role: str  # e.g. "launcher", "strategy", "workload", "checkpoint", "runtime"
    purpose: str = ""


CandidateStatus = Literal[
    "selected",  # the winner
    "feasible",  # would work; lost the ranking
    "infeasible",  # fails a physical check (memory, network)
    "rejected_policy",  # violates a user constraint (budget, deadline, allow_* flag)
    "rejected_dominated",  # feasible but strictly worse than another candidate
]


class CandidateVerdict(BaseModel):
    """One evaluated candidate, with its numbers and its fate.

    Rejections are half the product: a user who sees *why* DDP was infeasible
    trusts the plan that was chosen.
    """

    name: str
    strategy_family: str
    workers: int
    status: CandidateStatus
    memory: MemoryBreakdown | None = None
    est_time_min: Estimate | None = None
    est_cost_usd: Estimate | None = None
    scaling_efficiency: float | None = None
    profiling_required: bool = False
    reasons: list[str] = Field(default_factory=list)


class CheckpointPolicy(BaseModel):
    backend: str = "pytorch_dcp"
    interval_seconds: int = 300
    note: str = ""


class StrategyPlan(BaseModel):
    """The planner's answer: a frozen, backend-neutral execution plan.

    Frozen means: `plan_id` is a content hash of the request + decisions +
    planner version, recorded on the job attempt at execution time so every
    incident can answer "what did we think, and why".
    """

    apiVersion: str = PLAN_API_VERSION
    kind: Literal["StrategyPlan"] = "StrategyPlan"
    plan_id: str
    planner_version: str

    workload_mode: Literal["local", "independent_tasks", "coordinated_training"]
    strategy_family: str  # e.g. "ddp", "fsdp2", "zero3_cpu_offload", "lease_tasks", "local_process"
    launcher: str  # e.g. "torchrun", "flashruntime-leases", "local"

    workers: int = 1
    gpus_per_worker: int = 0
    colocated: bool = True

    precision: str | None = None
    quantization: str | None = None  # e.g. "nf4" for QLoRA frozen weights
    peft: str | None = None  # e.g. "lora(r=16)"
    micro_batch_per_gpu: int | None = None
    grad_accum: int | None = None
    activation_checkpointing: bool | None = None
    offload: Literal["none", "optimizer_cpu", "params_cpu", "nvme"] = "none"

    libraries: list[LibraryRef] = Field(default_factory=list)
    checkpoint: CheckpointPolicy | None = None

    memory: MemoryBreakdown | None = None
    est_time_min: Estimate | None = None
    est_cost_usd: Estimate | None = None
    scaling_efficiency: float | None = None
    profiling_required: bool = False

    selected_because: list[str] = Field(default_factory=list)


class PlanReport(BaseModel):
    """Everything the planner concluded: the winner and the full field.

    `selected` is None when no candidate survived; `no_valid_strategy_hint`
    then carries the nearest-miss analysis (the minimal relaxation that would
    unlock a plan) — a dead end must still be a useful answer.
    """

    apiVersion: str = PLAN_API_VERSION
    kind: Literal["PlanReport"] = "PlanReport"
    planner_version: str
    request_digest: str
    selected: StrategyPlan | None = None
    candidates: list[CandidateVerdict] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    no_valid_strategy_hint: str | None = None
