# Examples

`local_kmeans_and_linear_regression.py` is the canonical Phase 0 acceptance
example. It uses only the public `flashml` API and proves:

- provider lookup and local provisioning;
- one-time dataset sharding through `Storage`;
- MapReduce K-Means convergence;
- parameter-server/FedAvg-style linear and logistic training;
- final fitted estimators are usable through `predict()`.

Run it from the repository root:

```bash
.venv/bin/python examples/local_kmeans_and_linear_regression.py
```

Future examples belong here only after their phase implementation exists and
the example asserts a meaningful result. Planned APIs should remain in phase
documents rather than executable-looking placeholder scripts.
