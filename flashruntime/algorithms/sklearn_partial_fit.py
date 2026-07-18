from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from flashruntime.algorithms.base import DistributedAlgorithm, Metrics, ModelState, ShardDescriptor
from flashruntime.algorithms.sklearn_registry import get_estimator_class
from flashruntime.adapters.base import TaskResult
from flashruntime.storage.base import Storage


def _shard_keys(job_id: str, shard_id: int) -> tuple[str, str]:
    return f"{job_id}/pf_shard_{shard_id}_X.bin", f"{job_id}/pf_shard_{shard_id}_y.bin"


def _load(storage: Storage, key: str, shape: list[int]) -> np.ndarray:
    return np.frombuffer(storage.get(key), dtype=np.float64).reshape(shape).copy()


class SklearnPartialFitAlgorithm(DistributedAlgorithm):
    """Generic Parameter Server strategy over any scikit-learn estimator that
    exposes partial_fit and holds its fitted state in array-valued attributes
    (coef_/intercept_ for SGDRegressor/Classifier, Perceptron,
    PassiveAggressive*, ...).

    Each iteration: broadcast the current global state -> every worker seeds
    a local estimator with it and takes one partial_fit pass on its shard
    (one local gradient step) -> the server averages the returned state,
    sample-size-weighted (FedAvg). This is an approximation of single-machine
    SGD, not identical to it -- averaging local steps instead of exchanging
    per-sample gradients -- but it is the same trade distributed
    parameter-server training always makes, and it converges for the
    well-behaved convex losses these estimators support.

    Classifiers need `classes` up front (the same requirement sklearn's own
    partial_fit has for streaming classification: the full label set must be
    known before the first call, since a shard may not see every class).
    """

    entrypoint = "flashruntime.algorithms.sklearn_partial_fit:map_task"

    def __init__(
        self,
        estimator_name: str,
        state_attrs: Sequence[str] = ("coef_", "intercept_"),
        estimator_kwargs: dict[str, Any] | None = None,
        classes: Sequence[Any] | None = None,
        n_shards: int = 4,
    ):
        self.estimator_name = estimator_name
        self.state_attrs = tuple(state_attrs)
        self.estimator_kwargs = estimator_kwargs or {}
        self.classes = list(classes) if classes is not None else None
        self.n_shards = n_shards

    def plan(self, dataset: Any, storage: Storage, job_id: str) -> list[ShardDescriptor]:
        X, y = dataset
        X = np.ascontiguousarray(np.asarray(X, dtype=np.float64))
        y = np.ascontiguousarray(np.asarray(y, dtype=np.float64))
        if len(X) != len(y):
            raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
        if len(X) < self.n_shards:
            raise ValueError(f"need at least n_shards={self.n_shards} samples, got {len(X)}")

        shard_indices = np.array_split(np.arange(len(X)), self.n_shards)
        shards: list[ShardDescriptor] = []
        for shard_id, idx in enumerate(shard_indices):
            x_key, y_key = _shard_keys(job_id, shard_id)
            storage.put(x_key, X[idx].tobytes())
            storage.put(y_key, y[idx].tobytes())
            shards.append(
                {
                    "shard_id": shard_id,
                    "x_key": x_key,
                    "y_key": y_key,
                    "x_shape": list(X[idx].shape),
                    "y_shape": list(y[idx].shape),
                    "n_samples": int(len(idx)),
                }
            )
        return shards

    def initialize(self, shards: list[ShardDescriptor], storage: Storage) -> ModelState:
        # Let sklearn compute correctly-shaped/initialized state itself --
        # one partial_fit pass on the first shard -- rather than hand-rolling
        # zero-init logic that has to special-case every estimator's shapes
        # (binary vs. multiclass coef_, etc).
        first = shards[0]
        X = _load(storage, first["x_key"], first["x_shape"])
        y = _load(storage, first["y_key"], first["y_shape"])

        est = get_estimator_class(self.estimator_name)(**self.estimator_kwargs)
        fit_kwargs = {"classes": np.asarray(self.classes)} if self.classes is not None else {}
        est.partial_fit(X, y, **fit_kwargs)

        state: ModelState = {attr: np.asarray(getattr(est, attr)).tolist() for attr in self.state_attrs}
        state["iteration"] = 0
        return state

    def make_tasks(
        self, state: ModelState, shards: list[ShardDescriptor], iteration: int
    ) -> list[dict[str, Any]]:
        return [
            {
                "shard": shard,
                "state": {attr: state[attr] for attr in self.state_attrs},
                "state_attrs": list(self.state_attrs),
                "estimator_name": self.estimator_name,
                "estimator_kwargs": self.estimator_kwargs,
                "classes": self.classes,
            }
            for shard in shards
        ]

    def reduce(self, state: ModelState, results: list[TaskResult]) -> tuple[ModelState, Metrics]:
        total_n = 0
        acc: dict[str, np.ndarray] = {}
        scores = []

        for r in results:
            payload = r.payload
            assert payload is not None
            n = payload["n_samples"]
            for attr in self.state_attrs:
                v = np.asarray(payload["state"][attr]) * n
                acc[attr] = v if attr not in acc else acc[attr] + v
            total_n += n
            scores.append(payload["score"])

        new_state: ModelState = {attr: (acc[attr] / total_n).tolist() for attr in self.state_attrs}
        new_state["iteration"] = state["iteration"] + 1
        metrics: Metrics = {"mean_score": float(np.mean(scores)), "n_samples": total_n}
        return new_state, metrics

    def converged(
        self, old_state: ModelState, new_state: ModelState, metrics: Metrics, threshold: float
    ) -> bool:
        old = np.concatenate([np.asarray(old_state[attr]).ravel() for attr in self.state_attrs])
        new = np.concatenate([np.asarray(new_state[attr]).ravel() for attr in self.state_attrs])
        return float(np.linalg.norm(new - old)) < threshold

    def finalize(self, state: ModelState, shards: list[ShardDescriptor], storage: Storage) -> Any:
        """Returns a real, usable fitted sklearn estimator -- not just a
        dict of numbers -- so the result of a distributed job is exactly what
        `est.fit(X, y)` would have given you locally, ready for `.predict()`."""
        est = get_estimator_class(self.estimator_name)(**self.estimator_kwargs)
        for attr in self.state_attrs:
            value = np.asarray(state[attr])
            # coef_/intercept_-style attributes carry a reference shape from
            # the estimator's own first partial_fit (see initialize()); reuse
            # its dtype/shape rather than guessing.
            setattr(est, attr, value)
        if self.classes is not None:
            est.classes_ = np.asarray(self.classes)
        return est


def map_task(payload: dict[str, Any], storage: Storage) -> dict[str, Any]:
    """Runs on a worker: seed a local estimator with the broadcast global
    state, take one partial_fit pass on this shard, return updated state."""
    shard = payload["shard"]
    X = _load(storage, shard["x_key"], shard["x_shape"])
    y = _load(storage, shard["y_key"], shard["y_shape"])

    estimator_cls = get_estimator_class(payload["estimator_name"])
    est = estimator_cls(**payload["estimator_kwargs"])
    for attr, value in payload["state"].items():
        setattr(est, attr, np.asarray(value, dtype=np.float64))

    classes = payload.get("classes")
    fit_kwargs = {"classes": np.asarray(classes)} if classes is not None else {}
    est.partial_fit(X, y, **fit_kwargs)

    new_state = {attr: np.asarray(getattr(est, attr)).tolist() for attr in payload["state_attrs"]}
    score = float(est.score(X, y))

    return {
        "state": new_state,
        "n_samples": int(len(y)),
        "score": score,
    }
