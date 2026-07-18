from __future__ import annotations

from dataclasses import dataclass, field
from queue import SimpleQueue
from typing import Any, Iterator


@dataclass
class JobEvent:
    job_id: str
    iteration: int
    metrics: dict[str, Any] = field(default_factory=dict)
    worker_states: list[dict[str, Any]] = field(default_factory=list)
    done: bool = False
    error: str | None = None


class Job:
    """Handle returned by Cluster.train(). M0's engine runs the training loop
    synchronously before returning the Job, so stream()/result() simply
    replay the events recorded during that run -- this keeps the public API
    stable for a future background/async engine without committing to one yet.
    """

    def __init__(self, job_id: str):
        self.job_id = job_id
        self._events: SimpleQueue[JobEvent] = SimpleQueue()
        self._result: Any = None
        self._error: BaseException | None = None

    def _emit(self, event: JobEvent) -> None:
        self._events.put(event)

    def _finish(self, result: Any) -> None:
        self._result = result
        self._events.put(JobEvent(job_id=self.job_id, iteration=-1, done=True))

    def _fail(self, exc: BaseException) -> None:
        self._error = exc
        self._events.put(JobEvent(job_id=self.job_id, iteration=-1, done=True, error=str(exc)))

    def stream(self) -> Iterator[JobEvent]:
        while True:
            event = self._events.get()
            yield event
            if event.done:
                return

    def result(self) -> Any:
        # _result/_error are already populated by the time train() returns --
        # the M0 engine runs the loop synchronously before handing back the
        # Job. Read them directly rather than draining the event queue again;
        # stream() may have already consumed it (SimpleQueue.get() has no
        # "peek", so a second drain here would block forever).
        if self._error is not None:
            raise self._error
        return self._result
