from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flashruntime.adapters.base import TaskResult
from flashruntime.storage.base import Storage

# A shard descriptor is a small, JSON-safe dict: {"shard_id", "storage_key", ...}.
# Model state is likewise small: centroids, weights -- never raw data.
ShardDescriptor = dict[str, Any]
ModelState = dict[str, Any]
Metrics = dict[str, Any]


class DistributedAlgorithm(ABC):
    """One instance per training job. Owns the map/reduce math; knows nothing
    about which provider executes it.

    The same interface covers every strategy the current playground has
    (MapReduce K-Means, Parameter Server / Gradient Sync, Embarrassingly
    Parallel search) -- they differ only in what state gets broadcast and how
    `reduce` combines results, not in the loop shape:

        plan (once) -> initialize -> for each iteration: make_tasks -> reduce -> converged?
    """

    #: dotted "module.path:function" run per-task by the provider's worker.
    entrypoint: str

    @abstractmethod
    def plan(self, dataset: Any, storage: Storage, job_id: str) -> list[ShardDescriptor]:
        """Partition the dataset, writing each shard to storage exactly once.
        Returns small shard descriptors -- never the shard data itself."""

    @abstractmethod
    def initialize(self, shards: list[ShardDescriptor], storage: Storage) -> ModelState:
        """Initial model state (e.g. seed centroids / weights)."""

    @abstractmethod
    def make_tasks(
        self, state: ModelState, shards: list[ShardDescriptor], iteration: int
    ) -> list[dict[str, Any]]:
        """One task payload per shard for this iteration. Must stay small --
        reference shard data by storage key, never inline it."""

    @abstractmethod
    def reduce(
        self, state: ModelState, results: list[TaskResult]
    ) -> tuple[ModelState, Metrics]:
        """Aggregate task results into (new_state, metrics-for-this-iteration)."""

    @abstractmethod
    def converged(
        self, old_state: ModelState, new_state: ModelState, metrics: Metrics, threshold: float
    ) -> bool: ...

    @abstractmethod
    def finalize(
        self, state: ModelState, shards: list[ShardDescriptor], storage: Storage
    ) -> Any:
        """Final result returned from Job.result()."""
