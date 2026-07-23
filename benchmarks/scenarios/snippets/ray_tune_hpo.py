# ray.tune grid search — minimal setup for the same 8-trial sweep.
# Source: https://docs.ray.io/en/latest/tune/getting-started.html (Ray 2.x Tune API)
from ray import tune
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


def trainable(config):
    X, y = make_classification(n_samples=600, n_features=12, random_state=0)
    if config["model"] == "logreg":
        est = LogisticRegression(C=config["C"], max_iter=500, random_state=0)
    else:
        est = RandomForestClassifier(n_estimators=config["n_estimators"], random_state=0)
    tune.report({"accuracy_mean": cross_val_score(est, X, y, cv=3).mean()})


tuner = tune.Tuner(
    trainable,
    param_space={
        "model": tune.grid_search(["logreg", "rf"]),
        "C": tune.grid_search([0.1, 1.0]),
        "n_estimators": tune.grid_search([50, 100]),
    },
)
results = tuner.fit()
print(results.get_best_result(metric="accuracy_mean", mode="max").config)
