"""FlashRuntime strategy planner — deterministic, explainable, framework-free.

Public entry point:

    from flashruntime.protocol.plan_v1alpha1 import PlanRequest
    from flashruntime.planner import plan

    report = plan(request)          # PlanReport
    print(render(report))           # human-readable explanation

Pipeline (ADR-0003): resolve the request → generate candidates from a
curated menu per workload class → evaluate each with static estimators
(memory, communication, time/cost) → apply hard constraints → rank by the
user's objective → mint a frozen, backend-neutral StrategyPlan and report
every candidate's verdict with its arithmetic.

Hard rule: this package imports **no** ML framework. It reasons about
PyTorch, Ray, Transformers, and DeepSpeed by name and by number — strategy
compilers translate the plan into real configuration at execution time.

Honesty contract: every estimate is labeled `basis: static` until a
profiling run or ledger history replaces it; estimates the inputs can't
support are omitted, never invented; plans inside the memory caution band
are marked `profiling_required`.
"""

from __future__ import annotations

from flashruntime.planner.candidates import generate
from flashruntime.planner.explain import render_report as render
from flashruntime.planner.resolve import request_digest
from flashruntime.planner.selector import select
from flashruntime.protocol.plan_v1alpha1 import PlanReport, PlanRequest

PLANNER_VERSION = "0.1.0"

__all__ = ["plan", "render", "PLANNER_VERSION"]


def plan(request: PlanRequest) -> PlanReport:
    """Evaluate a PlanRequest and return the full PlanReport.

    Deterministic: identical requests (and planner version) produce an
    identical report — the request digest in the output is the identity of
    the decision.
    """
    digest = request_digest(request, PLANNER_VERSION)
    try:
        evaluated, notes = generate(request)
    except ValueError as exc:
        return PlanReport(
            planner_version=PLANNER_VERSION,
            request_digest=digest,
            no_valid_strategy_hint=str(exc),
        )
    return select(request, evaluated, PLANNER_VERSION, digest, notes)
