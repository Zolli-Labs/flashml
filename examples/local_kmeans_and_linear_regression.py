"""End-to-end proof that flashruntime.Cluster(provider="local") runs real
distributed training through the Provider/Storage/Algorithm interfaces --
no RunPod, no network, just the library.

Run: .venv/bin/python examples/local_kmeans_and_linear_regression.py
"""

import numpy as np

import flashruntime


def run_kmeans() -> None:
    print("=== MapReduce: distributed K-Means (flashruntime.algorithms.KMeans) ===")
    rng = np.random.default_rng(0)
    true_centers = np.array([[0, 0], [10, 10], [0, 10]])
    X = np.vstack([rng.normal(loc=c, scale=0.6, size=(200, 2)) for c in true_centers])
    rng.shuffle(X)

    with flashruntime.Cluster(provider="local", workers=4) as cluster:
        job = cluster.train(
            algorithm=flashruntime.algorithms.KMeans(k=3, n_shards=4, random_state=0),
            dataset=X,
            max_iterations=20,
            convergence_threshold=1e-3,
        )
        for event in job.stream():
            if not event.done:
                print(f"  iter {event.iteration}: inertia={event.metrics['inertia']:.2f} "
                      f"sizes={event.metrics['cluster_sizes']}")
        result = job.result()

    found = np.array(result["centroids"])
    print(f"converged in {result['iterations']} iterations")
    print("recovered centroids:\n", found)

    matched = sorted(found.tolist(), key=lambda c: sum(c))
    expected = sorted(true_centers.tolist(), key=lambda c: sum(c))
    max_err = max(np.linalg.norm(np.array(m) - np.array(e)) for m, e in zip(matched, expected))
    print(f"max centroid error vs ground truth: {max_err:.3f}")
    assert max_err < 1.0, "K-Means did not converge close to the true cluster centers"
    print("OK\n")


def run_linear_regression() -> None:
    print("=== Parameter Server: distributed linear regression (flashruntime.algorithms.LinearRegression) ===")
    rng = np.random.default_rng(1)
    n, d = 2000, 5
    true_coef = rng.uniform(-3, 3, size=d)
    true_intercept = 2.0
    X = rng.normal(size=(n, d))
    y = X @ true_coef + true_intercept + rng.normal(scale=0.1, size=n)

    with flashruntime.Cluster(provider="local", workers=4) as cluster:
        job = cluster.train(
            algorithm=flashruntime.algorithms.LinearRegression(
                n_shards=4, max_iter=1, learning_rate="constant", eta0=0.01, random_state=1
            ),
            dataset=(X, y),
            max_iterations=200,
            convergence_threshold=1e-5,
        )
        last_metrics = None
        for event in job.stream():
            if not event.done:
                last_metrics = event.metrics
        result = job.result()

    print(f"final mean R^2={last_metrics['mean_score']:.4f}")
    print("true coef:     ", np.round(true_coef, 3))
    print("recovered coef:", np.round(result.coef_, 3))
    print(f"true intercept={true_intercept:.3f}  recovered intercept={result.intercept_[0]:.3f}")

    coef_err = np.linalg.norm(result.coef_ - true_coef)
    print(f"coef error: {coef_err:.3f}")
    assert coef_err < 0.5, "distributed SGD did not converge close to the true weights"

    # finalize() hands back a real, usable sklearn estimator, not just numbers.
    preds = result.predict(X[:5])
    print(f"predict() on 5 rows works: {np.round(preds, 2)}")
    print("OK\n")


def run_logistic_regression() -> None:
    print("=== Parameter Server: distributed logistic regression (flashruntime.algorithms.LogisticRegression) ===")
    rng = np.random.default_rng(2)
    n, d = 2000, 4
    true_coef = rng.uniform(-3, 3, size=d)
    X = rng.normal(size=(n, d))
    logits = X @ true_coef
    y = (logits > 0).astype(np.float64)

    with flashruntime.Cluster(provider="local", workers=4) as cluster:
        job = cluster.train(
            algorithm=flashruntime.algorithms.LogisticRegression(
                classes=[0.0, 1.0], n_shards=4, max_iter=1, learning_rate="constant", eta0=0.05, random_state=2
            ),
            dataset=(X, y),
            max_iterations=100,
            convergence_threshold=1e-5,
        )
        last_metrics = None
        for event in job.stream():
            if not event.done:
                last_metrics = event.metrics
        result = job.result()

    print(f"final mean accuracy={last_metrics['mean_score']:.4f}")
    preds = result.predict(X)
    accuracy = float((preds == y).mean())
    print(f"accuracy on full set: {accuracy:.4f}")
    assert accuracy > 0.9, "distributed logistic regression did not learn a good decision boundary"
    print("OK\n")


if __name__ == "__main__":
    print(f"registered providers: {flashruntime.registered_providers()}\n")
    run_kmeans()
    run_linear_regression()
    run_logistic_regression()
    print("All local-provider demos passed.")
