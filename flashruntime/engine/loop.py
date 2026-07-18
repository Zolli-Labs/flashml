from __future__ import annotations

import json
import uuid

from flashruntime.algorithms.base import DistributedAlgorithm
from flashruntime.engine.job import Job, JobEvent
from flashruntime.adapters.base import MAX_PAYLOAD_BYTES, Storage, Task, WorkerPool


def _task_id() -> str:
    return uuid.uuid4().hex[:12]


def run_job(
    job: Job,
    algorithm: DistributedAlgorithm,
    dataset: object,
    pool: WorkerPool,
    storage: Storage,
    max_iterations: int,
    convergence_threshold: float,
    task_timeout_s: float,
) -> None:
    """The strategy-agnostic training loop: plan once, then iterate
    make_tasks -> dispatch -> reduce -> check convergence. Populates `job`
    with per-iteration events and a final result or error; never raises --
    failures surface through job.result()."""
    try:
        shards = algorithm.plan(dataset, storage, job.job_id)
        state = algorithm.initialize(shards, storage)

        for iteration in range(max_iterations):
            task_payloads = algorithm.make_tasks(state, shards, iteration)

            handles = []
            for payload in task_payloads:
                encoded = json.dumps(payload).encode()
                if len(encoded) > MAX_PAYLOAD_BYTES:
                    raise ValueError(
                        f"task payload for shard is {len(encoded)} bytes, exceeds "
                        f"{MAX_PAYLOAD_BYTES} -- large data must go through Storage, "
                        "not the task payload"
                    )
                task = Task(task_id=_task_id(), job_id=job.job_id, payload=payload, timeout_s=task_timeout_s)
                handles.append(pool.submit(task))

            results = pool.gather(handles, task_timeout_s)
            failed = [r for r in results if not r.ok]
            if failed:
                raise RuntimeError(f"{len(failed)}/{len(results)} task(s) failed: {failed[0].error}")

            new_state, metrics = algorithm.reduce(state, results)
            job._emit(
                JobEvent(
                    job_id=job.job_id,
                    iteration=iteration,
                    metrics=metrics,
                    worker_states=[
                        {"worker_id": r.worker_id, "duration_ms": r.duration_ms} for r in results
                    ],
                )
            )

            converged = algorithm.converged(state, new_state, metrics, convergence_threshold)
            state = new_state
            if converged:
                break

        final = algorithm.finalize(state, shards, storage)
        job._finish(final)
    except BaseException as exc:  # noqa: BLE001 -- captured on the Job, not propagated here
        job._fail(exc)
