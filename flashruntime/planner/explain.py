"""Render a PlanReport as human-readable text (the CLI's output).

The explanation is half the product: the selected plan with its arithmetic,
then every other candidate with why it lost, was infeasible, or violated a
constraint. Nothing is hidden — a user who can audit the rejections trusts
the selection.
"""

from __future__ import annotations

from flashruntime.protocol.plan_v1alpha1 import CandidateVerdict, PlanReport, StrategyPlan

_STATUS_LABEL = {
    "selected": "SELECTED",
    "feasible": "feasible (not chosen)",
    "infeasible": "INFEASIBLE",
    "rejected_policy": "rejected (constraint)",
    "rejected_dominated": "rejected (dominated)",
}


def render_report(report: PlanReport) -> str:
    lines: list[str] = []
    if report.selected is not None:
        lines.extend(_render_plan(report.selected))
    else:
        lines.append("NO VALID STRATEGY")
        if report.no_valid_strategy_hint:
            lines.append(f"  hint: {report.no_valid_strategy_hint}")
    lines.append("")
    lines.append(f"Candidates evaluated ({len(report.candidates)}):")
    for c in sorted(report.candidates, key=_candidate_order):
        lines.extend(_render_candidate(c))
    if report.warnings:
        lines.append("")
        lines.append("Notes:")
        lines.extend(f"  - {w}" for w in report.warnings)
    lines.append("")
    lines.append(f"planner {report.planner_version} · request {report.request_digest} · all estimates static unless labeled")
    return "\n".join(lines)


def _render_plan(p: StrategyPlan) -> list[str]:
    lines = [
        f"SELECTED PLAN  {p.plan_id}",
        f"  mode      : {p.workload_mode}",
        f"  strategy  : {p.strategy_family}"
        + (f" + {p.peft}" if p.peft else "")
        + (f" + {p.quantization} quantization" if p.quantization else ""),
        f"  topology  : {p.workers} worker(s)"
        + (f" × {p.gpus_per_worker} GPU" if p.gpus_per_worker else " (CPU)")
        + (", colocated" if p.colocated and p.workers > 1 else ""),
        f"  launcher  : {p.launcher}",
    ]
    knobs = []
    if p.precision:
        knobs.append(f"precision={p.precision}")
    if p.micro_batch_per_gpu:
        knobs.append(f"micro_batch={p.micro_batch_per_gpu}")
    if p.grad_accum:
        knobs.append(f"grad_accum={p.grad_accum}")
    if p.activation_checkpointing is not None:
        knobs.append(f"activation_checkpointing={p.activation_checkpointing}")
    if p.offload != "none":
        knobs.append(f"offload={p.offload}")
    if knobs:
        lines.append(f"  knobs     : {', '.join(knobs)}")
    if p.memory:
        m = p.memory
        lines.append(
            f"  memory/GPU: {m.total_gb} GB  (weights {m.weights_gb} + grads {m.gradients_gb} + "
            f"optimizer {m.optimizer_gb} + activations {m.activations_gb} + transient {m.transient_gb} "
            f"+ overhead {m.overhead_gb})"
            + (f"  [+{m.cpu_offload_gb} GB host RAM]" if m.cpu_offload_gb else "")
        )
    if p.est_time_min:
        lines.append(f"  est. time : {p.est_time_min.value:.0f} min  ({p.est_time_min.note})")
    if p.est_cost_usd:
        lines.append(f"  est. cost : ${p.est_cost_usd.value:.2f}")
    if p.scaling_efficiency is not None and p.workers > 1:
        lines.append(f"  scaling   : {p.scaling_efficiency:.2f} efficiency")
    if p.checkpoint:
        lines.append(f"  checkpoint: {p.checkpoint.backend} every {p.checkpoint.interval_seconds}s")
    if p.profiling_required:
        lines.append("  ⚠ profiling required before launch (memory estimate inside the caution band)")
    lines.append("  libraries :")
    lines.extend(f"    - {ref.name:<28} {ref.role:<10} {ref.purpose}" for ref in p.libraries)
    lines.append("  because   :")
    lines.extend(f"    - {r}" for r in p.selected_because)
    return lines


def _render_candidate(c: CandidateVerdict) -> list[str]:
    head = f"  [{_STATUS_LABEL[c.status]}] {c.name} ({c.workers} worker(s))"
    if c.memory:
        head += f" — {c.memory.total_gb} GB/GPU"
    if c.est_time_min:
        head += f", ~{c.est_time_min.value:.0f} min"
    if c.est_cost_usd:
        head += f", ~${c.est_cost_usd.value:.2f}"
    lines = [head]
    if c.status != "selected":  # the winner's reasons already shown in the plan
        lines.extend(f"      {r}" for r in c.reasons)
    return lines


def _candidate_order(c: CandidateVerdict):
    order = {"selected": 0, "feasible": 1, "rejected_policy": 2, "rejected_dominated": 3, "infeasible": 4}
    return (order[c.status], c.workers, c.name)
