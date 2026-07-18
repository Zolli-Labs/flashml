from __future__ import annotations

from typing import Any

from flashruntime.algorithms.sklearn_partial_fit import SklearnPartialFitAlgorithm


class LinearRegressionAlgorithm(SklearnPartialFitAlgorithm):
    """Parameter Server / Gradient Sync strategy for linear regression,
    backed by sklearn.linear_model.SGDRegressor via SklearnPartialFitAlgorithm."""

    def __init__(self, n_shards: int = 4, **sgd_kwargs: Any):
        super().__init__(estimator_name="sgd_regressor", estimator_kwargs=sgd_kwargs, n_shards=n_shards)
