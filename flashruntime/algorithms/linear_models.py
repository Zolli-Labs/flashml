from __future__ import annotations

from typing import Any, Sequence

from flashruntime.algorithms.sklearn_partial_fit import SklearnPartialFitAlgorithm


class LogisticRegressionAlgorithm(SklearnPartialFitAlgorithm):
    """Parameter Server strategy for logistic regression, backed by
    sklearn.linear_model.SGDClassifier(loss="log_loss")."""

    def __init__(self, classes: Sequence[Any], n_shards: int = 4, **sgd_kwargs: Any):
        sgd_kwargs.setdefault("loss", "log_loss")
        super().__init__(
            estimator_name="sgd_classifier",
            estimator_kwargs=sgd_kwargs,
            classes=classes,
            n_shards=n_shards,
        )


class PerceptronAlgorithm(SklearnPartialFitAlgorithm):
    """Parameter Server strategy backed by sklearn.linear_model.Perceptron."""

    def __init__(self, classes: Sequence[Any], n_shards: int = 4, **kwargs: Any):
        super().__init__(
            estimator_name="perceptron",
            estimator_kwargs=kwargs,
            classes=classes,
            n_shards=n_shards,
        )


class PassiveAggressiveRegressorAlgorithm(SklearnPartialFitAlgorithm):
    """Parameter Server strategy backed by
    sklearn.linear_model.PassiveAggressiveRegressor."""

    def __init__(self, n_shards: int = 4, **kwargs: Any):
        super().__init__(
            estimator_name="passive_aggressive_regressor",
            estimator_kwargs=kwargs,
            n_shards=n_shards,
        )


class PassiveAggressiveClassifierAlgorithm(SklearnPartialFitAlgorithm):
    """Parameter Server strategy backed by
    sklearn.linear_model.PassiveAggressiveClassifier."""

    def __init__(self, classes: Sequence[Any], n_shards: int = 4, **kwargs: Any):
        super().__init__(
            estimator_name="passive_aggressive_classifier",
            estimator_kwargs=kwargs,
            classes=classes,
            n_shards=n_shards,
        )
