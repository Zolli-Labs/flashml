from __future__ import annotations

from typing import Any

import numpy as np

from flashruntime.algorithms.base import DistributedAlgorithm, Metrics, ModelState, ShardDescriptor
from flashruntime.adapters.base import TaskResult
from flashruntime.storage.base import Storage


def _shard_key(job_id: str, shard_id: int) -> str:
    return f"{job_id}/kmeans_shard_{shard_id}.bin"


class KMeansAlgorithm(DistributedAlgorithm):
    """Distributed Lloyd's-algorithm K-Means.

    scikit-learn supplies the single-machine primitives (k-means++ seeding,
    nearest-centroid assignment); FlashML supplies the broadcast -> map ->
    reduce loop that runs those primitives across shards and workers. This is
    the same MapReduce shape as the original RunPod Flash K-Means worker (see
    TOOLS.md), now provider-agnostic.
    """

    entrypoint = "flashruntime.algorithms.kmeans:map_task"

    def __init__(self, k: int, n_shards: int = 4, random_state: int | None = None):
        self.k = k
        self.n_shards = n_shards
        self.random_state = random_state
        self._seed_dataset: np.ndarray | None = None

    def plan(self, dataset: Any, storage: Storage, job_id: str) -> list[ShardDescriptor]:
        X = np.ascontiguousarray(np.asarray(dataset, dtype=np.float64))
        if X.ndim != 2:
            raise ValueError(f"KMeans dataset must be 2D (n_samples, n_features), got shape {X.shape}")
        if len(X) < self.k:
            raise ValueError(f"need at least k={self.k} samples, got {len(X)}")

        # Kept in-process for centroid seeding only -- never shipped over the
        # wire. Shards below are what workers actually receive, by storage key.
        self._seed_dataset = X

        shard_indices = np.array_split(np.arange(len(X)), self.n_shards)
        shards: list[ShardDescriptor] = []
        for shard_id, idx in enumerate(shard_indices):
            shard_X = X[idx]
            key = _shard_key(job_id, shard_id)
            storage.put(key, shard_X.tobytes())
            shards.append(
                {
                    "shard_id": shard_id,
                    "storage_key": key,
                    "shape": list(shard_X.shape),
                    "n_samples": int(len(idx)),
                }
            )
        return shards

    def initialize(self, shards: list[ShardDescriptor], storage: Storage) -> ModelState:
        from sklearn.cluster import kmeans_plusplus

        assert self._seed_dataset is not None, "plan() must run before initialize()"
        centers, _ = kmeans_plusplus(
            self._seed_dataset, n_clusters=self.k, random_state=self.random_state
        )
        return {"centroids": centers.tolist(), "iteration": 0}

    def make_tasks(
        self, state: ModelState, shards: list[ShardDescriptor], iteration: int
    ) -> list[dict[str, Any]]:
        return [{"shard": shard, "centroids": state["centroids"]} for shard in shards]

    def reduce(self, state: ModelState, results: list[TaskResult]) -> tuple[ModelState, Metrics]:
        k = self.k
        n_features = len(state["centroids"][0])
        sums = np.zeros((k, n_features))
        counts = np.zeros(k)
        total_inertia = 0.0
        total_points = 0

        for r in results:
            payload = r.payload
            assert payload is not None
            for cluster_id, s in payload["partial_sums"].items():
                sums[int(cluster_id)] += np.asarray(s)
            for cluster_id, c in payload["partial_counts"].items():
                counts[int(cluster_id)] += c
            total_inertia += payload["inertia"]
            total_points += payload["n_samples"]

        old_centroids = np.array(state["centroids"])
        new_centroids = old_centroids.copy()
        empty_clusters = [c for c in range(k) if counts[c] == 0]
        for cluster_id in range(k):
            if counts[cluster_id] > 0:
                new_centroids[cluster_id] = sums[cluster_id] / counts[cluster_id]

        metrics: Metrics = {
            "inertia": total_inertia,
            "n_points": total_points,
            "cluster_sizes": counts.astype(int).tolist(),
            "empty_clusters": empty_clusters,
        }
        new_state: ModelState = {
            "centroids": new_centroids.tolist(),
            "iteration": state["iteration"] + 1,
        }
        return new_state, metrics

    def converged(
        self, old_state: ModelState, new_state: ModelState, metrics: Metrics, threshold: float
    ) -> bool:
        old = np.array(old_state["centroids"])
        new = np.array(new_state["centroids"])
        return float(np.linalg.norm(new - old)) < threshold

    def finalize(
        self, state: ModelState, shards: list[ShardDescriptor], storage: Storage
    ) -> Any:
        return {"centroids": state["centroids"], "iterations": state["iteration"]}


def map_task(payload: dict[str, Any], storage: Storage) -> dict[str, Any]:
    """Runs on a worker. Assigns this shard's points to the nearest broadcast
    centroid and returns partial sums/counts per cluster -- never raw points."""
    from sklearn.metrics import pairwise_distances_argmin_min

    shard = payload["shard"]
    X = np.frombuffer(storage.get(shard["storage_key"]), dtype=np.float64).reshape(shard["shape"]).copy()
    centroids = np.asarray(payload["centroids"], dtype=np.float64)

    assignments, dists = pairwise_distances_argmin_min(X, centroids)

    partial_sums: dict[str, list[float]] = {}
    partial_counts: dict[str, int] = {}
    for cluster_id in np.unique(assignments):
        mask = assignments == cluster_id
        partial_sums[str(int(cluster_id))] = X[mask].sum(axis=0).tolist()
        partial_counts[str(int(cluster_id))] = int(mask.sum())

    return {
        "partial_sums": partial_sums,
        "partial_counts": partial_counts,
        "inertia": float((dists**2).sum()),
        "n_samples": int(len(X)),
    }
