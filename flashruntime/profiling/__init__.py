"""Profiling: measured numbers that replace the planner's assumptions.

The planner's tier-2 upgrade (FLASHRUNTIME_EVALUATION §F): run a few real
steps of the real workload and replace the `[assumption]` constants in
`planner/catalog.py` (activation bytes, MFU, offload penalty) with
measurements. A `ProfileResult` is born with `basis="profiled"` — the
whole honesty ladder (`static → profiled → ledger`) hangs on never faking
that field.

Isolation invariants (every implementation MUST honor all four — they are
what makes profiling safe to run against production jobs):
1. Separate namespace: profile runs use their own run-id/prefix; they
   never write under a real job's artifact prefix.
2. No commits: a profile run never creates ArtifactRecords, never commits
   checkpoints to the catalog, never counts as accepted work.
3. Independent RNG: profiling must not consume the training run's seed
   stream (the sgd_trainer's step-indexed batching makes this trivial —
   keep that property in future trainers).
4. Bounded cost: hard wall-clock + step budgets; a hung profile is
   cancelled, and "no measurement" is reported honestly rather than a
   fabricated number.

Measurement protocol (defaults from the evaluation): `warmup_steps=3`
(skip compile/cache effects), `measure_steps=20`, peak memory via the
framework's allocator stats plus process RSS, one checkpoint save/restore
cycle — which doubles as free validation of the checkpoint contract
before the real run depends on it.

Status: interface complete (final surface); first concrete implementation
targets the LoRA recipe (SPRINT_PLAN Days 8–10) — profile locally via the
subprocess runner before wiring cluster profiling.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal, Protocol

from pydantic import BaseModel, Field

from flashruntime.protocol.plan_v1alpha1 import PlanRequest, StrategyPlan

__all__ = ["ProfileResult", "ProfileCache", "Profiler", "ProfileError"]


class ProfileError(Exception):
    """The profile run could not produce trustworthy numbers (crashed,
    exceeded budget, measured nonsense). Callers fall back to static
    estimates — with the plan still labeled `basis: static`, never
    upgraded on a failed profile."""


class ProfileResult(BaseModel):
    """What a profile run measured. Every field is optional except the
    bookkeeping — a partial profile is still useful (peak memory without
    throughput beats assumptions), but absent numbers stay absent."""

    basis: Literal["profiled"] = "profiled"
    peak_vram_gb: float | None = Field(default=None, ge=0)
    host_ram_gb: float | None = Field(default=None, ge=0)
    step_time_s: float | None = Field(default=None, gt=0)
    tokens_per_s: float | None = Field(default=None, gt=0)
    dataloader_wait_fraction: float | None = Field(default=None, ge=0, le=1)
    checkpoint_save_s: float | None = Field(default=None, ge=0)
    checkpoint_restore_s: float | None = Field(default=None, ge=0)
    checkpoint_size_gb: float | None = Field(default=None, ge=0)
    measured_steps: int = Field(ge=1)
    warmup_steps: int = Field(ge=0)
    notes: list[str] = Field(default_factory=list)


class ProfileCache(Protocol):
    """Profiles are expensive; identical questions must be answered once.

    The cache key is the tuple that actually determines the numbers:
    (model digest, strategy family + knobs, GPU class, framework versions,
    seq-len bucket, batch bucket) — see FLASHRUNTIME_EVALUATION §F. Real
    runs later back-feed the same table with `basis: ledger` at higher
    trust; the cache interface stays identical.
    """

    def key(self, request: PlanRequest, plan: StrategyPlan) -> str: ...

    def get(self, key: str) -> ProfileResult | None: ...

    def put(self, key: str, result: ProfileResult) -> None: ...


class Profiler(ABC):
    """Run a bounded, isolated measurement of one candidate plan.

    Skip policy lives in the *caller* (the planner), not here: the planner
    skips profiling when a cache hit exists, when static margins are
    comfortable (≤ 0.6·VRAM), or when the profile would cost more than
    2–5% of the job budget. The profiler's only judgment is "did I measure
    something trustworthy" — everything else is planning.
    """

    @abstractmethod
    def profile(
        self,
        request: PlanRequest,
        plan: StrategyPlan,
        budget_seconds: float = 300.0,
    ) -> ProfileResult:
        """Execute warmup + measured steps of `plan` for `request`.

        Inputs: the user's request (data/model identity), the candidate
        plan (strategy + knobs to measure under), a hard wall-clock budget.
        Output: a ProfileResult (partial allowed; see model notes).
        Raises: ProfileError when nothing trustworthy was measured.
        Must honor all four isolation invariants in the module docstring.
        """
