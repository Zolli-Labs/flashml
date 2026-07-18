from __future__ import annotations

import uuid

from flashruntime.algorithms.base import DistributedAlgorithm
from flashruntime.engine.job import Job
from flashruntime.engine.loop import run_job
from flashruntime.adapters.base import WorkerPool, WorkerSpec
from flashruntime.adapters.registry import get_provider


class Cluster:
    """Connects to a provider and runs training jobs against it.

    >>> with flashruntime.Cluster(provider="local", workers=4) as cluster:
    ...     job = cluster.train(
    ...         algorithm=flashruntime.algorithms.KMeans(k=5),
    ...         dataset=embeddings,
    ...     )
    ...     result = job.result()

    Swap provider="local" for provider="runpod" (once that connector lands)
    and the same script runs on real GPUs -- nothing above this line changes.
    """

    def __init__(
        self,
        provider: str,
        workers: int,
        accelerator: str = "cpu-small",
        credentials: dict[str, str] | None = None,
        **provider_kwargs: object,
    ):
        self.provider_name = provider
        self.workers = workers
        self.accelerator = accelerator
        self._provider = get_provider(provider, credentials=credentials, **provider_kwargs)
        self._pools: list[WorkerPool] = []

    def train(
        self,
        algorithm: DistributedAlgorithm,
        dataset: object,
        max_iterations: int = 10,
        convergence_threshold: float = 1e-4,
        task_timeout_s: float = 60.0,
    ) -> Job:
        """Run a training job.

        The distributed loop shape is owned by ``algorithm``; providers only
        execute its entrypoint and never receive a model/strategy selector.
        """
        job_id = uuid.uuid4().hex[:12]
        job = Job(job_id)

        spec = WorkerSpec(accelerator=self.accelerator, entrypoint=algorithm.entrypoint)
        pool = self._provider.provision(spec, self.workers)
        self._pools.append(pool)
        try:
            # Storage acquisition belongs inside the lifecycle guard. A remote
            # connector may provision billable workers successfully and then
            # fail while attaching its shared volume.
            storage = self._provider.storage()
            run_job(
                job=job,
                algorithm=algorithm,
                dataset=dataset,
                pool=pool,
                storage=storage,
                max_iterations=max_iterations,
                convergence_threshold=convergence_threshold,
                task_timeout_s=task_timeout_s,
            )
            return job
        finally:
            try:
                self._provider.teardown(pool)
            finally:
                self._pools.remove(pool)

    def close(self) -> None:
        for pool in list(self._pools):
            self._provider.teardown(pool)
            self._pools.remove(pool)

    def __enter__(self) -> "Cluster":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
