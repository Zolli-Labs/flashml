"""Backend-neutral execution contract.

A backend receives a validated Job (public JobSpec + runtime identity),
submits it to one execution system, and reports status/events/logs back in
FlashRuntime's vocabulary. Backends never invent job state — they map
observed backend signals onto `JobState` and `Event`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from flashruntime.protocol.v1alpha1 import ArtifactRecord, Event, JobRecord, JobSpec, JobState


class SpecValidationError(ValueError):
    """The JobSpec cannot run on this backend in this deployment profile."""


class BackendUnavailableError(RuntimeError):
    """The backend's execution system cannot be reached."""


@dataclass
class BackendExecution:
    """Identity of a submitted execution inside the backend."""

    execution_id: str  # e.g. RayJob custom-resource name
    backend: str
    submitted_at: datetime
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class BackendStatus:
    """Point-in-time backend view, already mapped to FlashRuntime state."""

    state: JobState
    reason: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ExecutionBackend(Protocol):
    name: str

    async def validate(self, spec: JobSpec) -> None:
        """Raise SpecValidationError if this backend/profile cannot run the spec."""
        ...

    async def submit(self, job: JobRecord) -> BackendExecution: ...

    async def get_status(self, execution_id: str) -> BackendStatus: ...

    def stream_events(self, execution_id: str) -> AsyncIterator[Event]: ...

    async def get_logs(self, execution_id: str) -> str: ...

    async def cancel(self, execution_id: str) -> None: ...

    async def collect_artifacts(self, execution_id: str) -> list[ArtifactRecord]: ...
