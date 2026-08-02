"""Resource providers: where machines come from (the fourth axis).

A provider turns money into capacity: list what's available (`offers`),
acquire some (`acquire`), release it (`release`). It NEVER runs workloads
— acquired machines join the system the same way a laptop does (a
FlashNode agent registers and pulls leases) or as a Mode B pool (KubeRay
on the acquired cluster). Keeping acquisition separate from execution is
what lets one job span a rented pool and a volunteer laptop without either
knowing about the other.

Rules distilled from the evaluation (§J) and the master report:
- One adapter first, strict conformance, before any second provider —
  every adapter is a permanent support cost.
- Providers are *quoted*, not trusted: an Offer's price/specs are claims;
  admission benchmarks (flashnode `benchmark/`) verify capability after
  the machine joins, exactly as for volunteered hardware.
- The planner consumes `hourly_usd` and capability fields to price plans;
  the scheduler never talks to providers — acquisition is a deliberate,
  budgeted act by the coordinator/cloud, not a side effect of placement.
- Idempotency: `release` on an unknown/already-released id is a no-op —
  reconciliation loops must be safe to re-run.

Status: interface complete (final surface); first concrete adapter per
demand (SkyPilot as provisioner, or Alibaba ECS for Stage-5 automation —
see HANDBOOK §4).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, ClassVar

from pydantic import BaseModel, Field

__all__ = ["Requirements", "Offer", "AcquiredCapacity", "ResourceProvider", "ProviderError"]


class ProviderError(Exception):
    """Provider API failure (auth, quota, capacity gone). Carries enough
    context to be a ledger event — provider errors are recovery evidence
    (FailureClass.PREEMPTION / capacity loss), never silent retries."""


class Requirements(BaseModel):
    """What the caller needs — the provider filters its inventory by this.
    All fields optional: an empty Requirements means 'show me everything'."""

    gpus: int | None = Field(default=None, ge=0)
    gpu_type: str | None = None
    vram_gb: float | None = Field(default=None, gt=0)
    cpus: int | None = Field(default=None, ge=1)
    memory_gb: float | None = Field(default=None, gt=0)
    region: str | None = None
    max_hourly_usd: float | None = Field(default=None, gt=0)
    spot_ok: bool = Field(
        default=True,
        description="Whether preemptible capacity is acceptable — spot "
        "failure rates feed the planner's expected-failure-cost math",
    )


class Offer(BaseModel):
    """One acquirable configuration, as quoted by the provider now.

    Offers are snapshots: they can expire or be taken. `acquire` must
    re-validate; a stale offer raises ProviderError, never silently
    substitutes different hardware."""

    provider: str
    gpu_type: str | None = None
    gpus: int = Field(default=0, ge=0)
    cpus: int | None = None
    memory_gb: float | None = None
    vram_gb: float | None = None
    hourly_usd: float = Field(gt=0)
    region: str | None = None
    spot: bool = False
    interconnect: str | None = Field(
        default=None, description="Planner link-class name when known (e.g. 'same_host_nvlink')"
    )
    offer_id: str = Field(default="", description="Provider-native id for acquire()")
    expires_at: datetime | None = None
    raw: dict[str, Any] = Field(
        default_factory=dict, description="Provider response verbatim, for audit"
    )


class AcquiredCapacity(BaseModel):
    """A machine (or pool) we are now paying for.

    Contains what the *coordinator* needs to onboard it — never workload
    credentials. `join_hint` describes how it enters the system: a cloud-
    init/user-data script that runs `flashnode work --coordinator ...`
    (Mode A) or a kubeconfig reference for a Mode B pool."""

    capacity_id: str
    provider: str
    offer: Offer
    acquired_at: datetime
    join_hint: dict[str, Any] = Field(default_factory=dict)


class ResourceProvider(ABC):
    """One source of rentable capacity.

    Lifecycle: `healthy()` preflight → `offers()` → `acquire()` →
    (capacity works for a while) → `release()`. Implementations must be
    safe to call from reconciliation loops: every method idempotent or
    read-only, every failure a typed ProviderError.
    """

    #: Registry/ledger name: "alibaba-ecs", "skypilot", "runpod", ...
    name: ClassVar[str]

    def healthy(self) -> tuple[bool, str]:
        """Credentials valid, API reachable? (ok, reason). Called before
        showing this provider's offers to anything that spends money."""
        return True, "no preflight implemented"

    @abstractmethod
    def offers(self, requirements: Requirements) -> list[Offer]:
        """Quote current availability matching `requirements`, cheapest
        first. Read-only; may be cached briefly by the caller. An empty
        list is an answer, not an error."""

    @abstractmethod
    def acquire(self, offer: Offer) -> AcquiredCapacity:
        """Buy it. Either returns capacity we are provably paying for, or
        raises ProviderError with nothing leaked — a half-acquired machine
        the caller doesn't know about is a money leak; implementations
        must clean up their own partial failures."""

    @abstractmethod
    def release(self, capacity_id: str) -> None:
        """Stop paying. Idempotent: unknown/already-released ids are
        no-ops. This is the method cost-safety leans on — it must be the
        most reliable code in the adapter."""
