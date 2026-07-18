from __future__ import annotations

from typing import Callable

# Estimator classes are referenced by name (not passed as callables) so that
# task payloads stay JSON-serializable end to end -- the same requirement a
# remote connector's HTTP payload has. Add an entry here to make a new
# scikit-learn partial_fit estimator usable by SklearnPartialFitAlgorithm.


def _sgd_regressor() -> Callable:
    from sklearn.linear_model import SGDRegressor

    return SGDRegressor


def _sgd_classifier() -> Callable:
    from sklearn.linear_model import SGDClassifier

    return SGDClassifier


def _perceptron() -> Callable:
    from sklearn.linear_model import Perceptron

    return Perceptron


def _passive_aggressive_regressor() -> Callable:
    from sklearn.linear_model import PassiveAggressiveRegressor

    return PassiveAggressiveRegressor


def _passive_aggressive_classifier() -> Callable:
    from sklearn.linear_model import PassiveAggressiveClassifier

    return PassiveAggressiveClassifier


ESTIMATOR_REGISTRY: dict[str, Callable[[], type]] = {
    "sgd_regressor": _sgd_regressor,
    "sgd_classifier": _sgd_classifier,
    "perceptron": _perceptron,
    "passive_aggressive_regressor": _passive_aggressive_regressor,
    "passive_aggressive_classifier": _passive_aggressive_classifier,
}


def get_estimator_class(name: str) -> type:
    try:
        loader = ESTIMATOR_REGISTRY[name]
    except KeyError:
        raise ValueError(
            f"Unknown estimator '{name}'. Registered: {sorted(ESTIMATOR_REGISTRY)}"
        ) from None
    return loader()
