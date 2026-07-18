from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

# Task payloads are shipped as JSON over whatever transport a connector uses
# (an HTTP call, a local function call, ...). Big data never goes here — it
# goes through Storage once, at plan time, and tasks reference it by key.
MAX_PAYLOAD_BYTES = 1_000_000


class FlashMLError(Exception):
    """Base class for all FlashML errors."""


class ProvisionError(FlashMLError):
    """Raised when a provider cannot bring up the requested workers, or when
    an unsupported accelerator tier is requested. Provisioning must leave no
    partial state behind when this is raised."""


class TaskError(FlashMLError):
    """Raised by the engine when a task cannot even be dispatched (not for
    task-level failures, which are reported as TaskResult(ok=False))."""


class StorageError(FlashMLError):
    """Raised by a Storage implementation on put/get/exists/delete failures."""


class AuthError(FlashMLError):
    """Raised when a provider's credentials are missing or rejected."""


@dataclass
class ResourceSpec:
    accelerator: str
    workers: int
    region: str | None = None


@dataclass
class Offer:
    provider: str
    accelerator: str
    native_type: str
    price_per_hour: float | None
    available: bool


@dataclass
class WorkerSpec:
    accelerator: str
    entrypoint: str
    dependencies: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Task:
    task_id: str
    job_id: str
    payload: dict[str, Any]
    timeout_s: float = 60.0


@dataclass
class TaskResult:
    task_id: str
    ok: bool
    payload: dict[str, Any] | None
    error: str | None
    duration_ms: int
    worker_id: str


class TaskHandle(ABC):
    @abstractmethod
    def wait(self, timeout_s: float) -> TaskResult: ...


class WorkerPool(ABC):
    @abstractmethod
    def submit(self, task: Task) -> TaskHandle: ...

    @abstractmethod
    def gather(self, handles: list[TaskHandle], timeout_s: float) -> list[TaskResult]:
        """Wait for all handles. A failed task returns TaskResult(ok=False, ...),
        it does not raise -- the caller decides whether that's fatal."""

    @property
    @abstractmethod
    def size(self) -> int: ...


class Storage(ABC):
    """A blob store reachable by the workers this provider provisions."""

    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...

    @abstractmethod
    def get(self, key: str) -> bytes: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def delete_prefix(self, prefix: str) -> None: ...


class Provider(ABC):
    """One instance per Cluster. See docs/PROVIDERS.md for the full connector
    contract new providers must satisfy."""

    name: str

    @abstractmethod
    def offers(self, spec: ResourceSpec) -> list[Offer]: ...

    @abstractmethod
    def provision(self, spec: WorkerSpec, count: int) -> WorkerPool: ...

    @abstractmethod
    def storage(self) -> Storage: ...

    @abstractmethod
    def teardown(self, pool: WorkerPool) -> None: ...
