from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor

from flashruntime.adapters.base import (
    Offer,
    Provider,
    ProvisionError,
    ResourceSpec,
    Storage,
    Task,
    TaskHandle,
    TaskResult,
    WorkerPool,
    WorkerSpec,
)
from flashruntime.adapters.entrypoint import resolve_entrypoint
from flashruntime.adapters.registry import register
from flashruntime.storage.local import LocalStorage


class LocalTaskHandle(TaskHandle):
    def __init__(self, future: Future, task_id: str, worker_id: str):
        self._future = future
        self._task_id = task_id
        self._worker_id = worker_id

    def wait(self, timeout_s: float) -> TaskResult:
        start = time.monotonic()
        try:
            payload = self._future.result(timeout=timeout_s)
            return TaskResult(
                task_id=self._task_id,
                ok=True,
                payload=payload,
                error=None,
                duration_ms=int((time.monotonic() - start) * 1000),
                worker_id=self._worker_id,
            )
        except Exception as exc:  # noqa: BLE001 -- surfaced as TaskResult, not raised
            return TaskResult(
                task_id=self._task_id,
                ok=False,
                payload=None,
                error=str(exc),
                duration_ms=int((time.monotonic() - start) * 1000),
                worker_id=self._worker_id,
            )


class LocalWorkerPool(WorkerPool):
    def __init__(self, count: int, storage: Storage, entrypoint: str):
        self._executor = ThreadPoolExecutor(max_workers=count, thread_name_prefix="flashruntime-local")
        self._count = count
        self._storage = storage
        self._handler = resolve_entrypoint(entrypoint)
        self._next_worker = 0

    def submit(self, task: Task) -> LocalTaskHandle:
        worker_id = f"local-{self._next_worker % self._count}"
        self._next_worker += 1
        future = self._executor.submit(self._handler, task.payload, self._storage)
        return LocalTaskHandle(future, task.task_id, worker_id)

    def gather(self, handles: list[LocalTaskHandle], timeout_s: float) -> list[TaskResult]:
        return [h.wait(timeout_s) for h in handles]

    @property
    def size(self) -> int:
        return self._count

    def shutdown(self) -> None:
        self._executor.shutdown(wait=True)


@register("local")
class LocalProvider(Provider):
    """Zero-setup connector: workers are threads in this process, storage is
    a temp directory. The reference every other connector is tested against."""

    name = "local"

    def __init__(self, storage_root: str | None = None, **_: object):
        self._storage = LocalStorage(root=storage_root)

    def offers(self, spec: ResourceSpec) -> list[Offer]:
        return [
            Offer(
                provider=self.name,
                accelerator=spec.accelerator,
                native_type="cpu-thread",
                price_per_hour=0.0,
                available=True,
            )
        ]

    def provision(self, spec: WorkerSpec, count: int) -> LocalWorkerPool:
        if count < 1:
            raise ProvisionError("worker count must be >= 1")
        return LocalWorkerPool(count, self._storage, spec.entrypoint)

    def storage(self) -> Storage:
        return self._storage

    def teardown(self, pool: WorkerPool) -> None:
        if isinstance(pool, LocalWorkerPool):
            pool.shutdown()
