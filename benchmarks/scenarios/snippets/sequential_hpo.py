# Hand-rolled sequential sweep — the boilerplate flashruntime replaces.
# You own the grid expansion, the subprocess plumbing, and result collection.
import itertools
import json
import subprocess

grid = {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50, 100]}
keys = sorted(grid)
results = []
for combo in itertools.product(*(grid[k] for k in keys)):
    params = dict(zip(keys, combo))
    argv = ["python", "train.py"]
    for k, v in params.items():
        argv += [f"--{k}", str(v)]
    subprocess.run(argv, cwd="user_sklearn", check=True)
    with open("user_sklearn/metrics.json") as f:
        results.append(json.load(f))
best = max(results, key=lambda r: r["accuracy_mean"])
print(best)
