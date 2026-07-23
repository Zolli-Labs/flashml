"""The wire schema for one measured result + the scenario protocol.

A leaf module (imports nothing from the suite) so scenarios can depend on
``ResultRow`` without importing ``registry`` — which imports every scenario —
and thus without a circular import. ``registry`` re-exports both names.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field


class ResultRow(BaseModel):
    """One scenario's measured result (schema element of ``bench_v1``).

    ``median`` is the headline figure in ``unit``; ``p10``/``p90`` bound the
    spread across ``repeats``. ``comparators`` holds the honest side-by-side
    numbers (baseline wall-clock, per-checkpoint cost, setup LOC, …) and
    ``notes`` documents every caveat and skip — an unfavorable number ships
    with its trade-off note, never deleted."""

    scenario: str
    unit: str
    median: float
    p10: float
    p90: float
    repeats: int
    comparators: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


@runtime_checkable
class Scenario(Protocol):
    name: str
    hypothesis: str  # one sentence, shown in the docs beside the table

    def run(self, repeats: int) -> ResultRow: ...
