"""Deterministic selection: filter → rank by objective → mint the plan.

Identical inputs always produce the identical plan (the report carries the
planner version and a request digest so decisions are reproducible and
auditable — same principle as the recovery policy engine).

Ranking is lexicographic per objective mode. Hard constraints (deadline,
budget, allow_* flags) were already applied as `rejected_policy` during
evaluation — the ranker only orders survivors:

  cheapest : (cost, time)         — unknown values sort last, never win by omission
  fastest  : (time, cost)
  balanced : cost first (deadline is already a hard gate), time tiebreak
  reliable : fewest moving parts — smallest world size, no offload, no
             quantization, most memory headroom

When nothing survives, the report still answers usefully: the nearest-miss
hint names the minimal relaxation that would unlock a plan.
"""

from __future__ import annotations

import math

from flashruntime.planner.candidates import Evaluated
from flashruntime.protocol.plan_v1alpha1 import (
    CheckpointPolicy,
    PlanReport,
    PlanRequest,
    StrategyPlan,
)

_INF = math.inf


def _cost(e: Evaluated) -> float:
    return e.verdict.est_cost_usd.value if e.verdict.est_cost_usd else _INF


def _time(e: Evaluated) -> float:
    return e.verdict.est_time_min.value if e.verdict.est_time_min else _INF


def _rank_key(e: Evaluated, mode: str):
    if mode == "cheapest":
        return (_cost(e), _time(e), e.verdict.workers)
    if mode == "fastest":
        return (_time(e), _cost(e), e.verdict.workers)
    if mode == "reliable":
        return (
            e.verdict.workers,
            0 if e.offload == "none" else 1,
            0 if e.quantization is None else 1,
            -e.headroom_gb,
        )
    # balanced
    return (_cost(e), _time(e), e.verdict.workers)


def select(
    req: PlanRequest,
    evaluated: list[Evaluated],
    planner_version: str,
    request_digest: str,
    notes: list[str],
) -> PlanReport:
    feasible = [e for e in evaluated if e.verdict.status == "feasible"]
    report = PlanReport(
        planner_version=planner_version,
        request_digest=request_digest,
        candidates=[e.verdict for e in evaluated],
        warnings=list(notes),
    )

    if not feasible:
        report.no_valid_strategy_hint = _nearest_miss(req, evaluated)
        return report

    feasible.sort(key=lambda e: _rank_key(e, req.objective.mode))
    winner = feasible[0]
    winner.verdict.status = "selected"
    report.selected = _mint_plan(req, winner, planner_version, request_digest, feasible)
    return report


def _mint_plan(
    req: PlanRequest,
    e: Evaluated,
    planner_version: str,
    request_digest: str,
    feasible: list[Evaluated],
) -> StrategyPlan:
    v = e.verdict
    because = list(v.reasons)
    because.append(f"objective '{req.objective.mode}': best-ranked of {len(feasible)} feasible candidate(s)")
    runner_up = feasible[1] if len(feasible) > 1 else None
    if runner_up is not None:
        because.append(
            f"runner-up was '{runner_up.verdict.name}' "
            f"(time {_fmt(runner_up.verdict.est_time_min and runner_up.verdict.est_time_min.value, 'min')}, "
            f"cost {_fmt(runner_up.verdict.est_cost_usd and runner_up.verdict.est_cost_usd.value, 'usd')})"
        )

    checkpoint = None
    if e.workload_mode == "coordinated_training":
        # Static default; Young–Daly (τ* = √(2·C·MTBF)) needs a measured
        # checkpoint duration and pool failure rate — profiling/ledger work.
        checkpoint = CheckpointPolicy(
            backend="pytorch_dcp",
            interval_seconds=300,
            note="static default; tune with Young–Daly once checkpoint duration is measured",
        )

    return StrategyPlan(
        plan_id=f"pl_{request_digest}",
        planner_version=planner_version,
        workload_mode=e.workload_mode,  # type: ignore[arg-type]
        strategy_family=v.strategy_family,
        launcher=e.launcher,
        workers=v.workers,
        gpus_per_worker=e.gpus_per_worker,
        colocated=e.colocated,
        precision=e.precision,
        quantization=e.quantization,
        peft=e.peft,
        micro_batch_per_gpu=e.micro_batch,
        grad_accum=e.grad_accum,
        activation_checkpointing=e.act_ckpt,
        offload=e.offload,  # type: ignore[arg-type]
        libraries=e.libraries,
        checkpoint=checkpoint,
        memory=v.memory,
        est_time_min=v.est_time_min,
        est_cost_usd=v.est_cost_usd,
        scaling_efficiency=v.scaling_efficiency,
        profiling_required=v.profiling_required,
        selected_because=because,
    )


def _nearest_miss(req: PlanRequest, evaluated: list[Evaluated]) -> str:
    """Turn a dead end into the minimal unlocking relaxation."""
    hints: list[str] = []
    mem_missed = [e for e in evaluated if e.vram_deficit_gb > 0]
    if mem_missed:
        closest = min(mem_missed, key=lambda e: e.vram_deficit_gb)
        hints.append(
            f"closest candidate '{closest.verdict.name}' missed VRAM by "
            f"{closest.vram_deficit_gb:.1f} GB/GPU — add GPUs or larger GPUs"
        )
        if not req.objective.allow_quantization:
            hints.append("allowing quantization (objective.allow_quantization) may unlock QLoRA")
        if not req.objective.allow_cpu_offload:
            hints.append("allowing CPU offload (objective.allow_cpu_offload) may unlock ZeRO-3 offload")
    policy_missed = [e for e in evaluated if e.verdict.status == "rejected_policy"]
    if policy_missed:
        hints.append(
            f"{len(policy_missed)} candidate(s) were feasible but violated the "
            "deadline/budget — relaxing objective constraints would unlock them"
        )
    if not hints:
        hints.append("no candidate came close — this workload is outside the supported envelope")
    return "; ".join(hints)


def _fmt(value: float | None, unit: str) -> str:
    if value is None:
        return "unknown"
    return f"${value:.2f}" if unit == "usd" else f"{value:.0f} {unit}"
