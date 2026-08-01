# Federated-Averaging Driver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train one model across several independent leased tasks by exchanging weight deltas once per round instead of gradients once per step, so volunteer machines on home internet can jointly train a model.

**Architecture:** A driver submits one Mode A job per round; each task loads the round's weights, trains locally for K steps on its shard, and writes a weight delta. The driver reduces the deltas into new weights and submits the next round. This is `flashml_workloads/kmeans_driver.py`'s control flow with `reduce` replaced — "pipelines are jobs chained by a driver, not a new execution mode." Unlike K-means, the driver aggregates on a **quorum** rather than requiring every shard, so a slow machine degrades participation instead of stalling the round.

**Tech Stack:** Python ≥3.10, pydantic, pure-stdlib workload modules, FastAPI service (existing), pytest.

This plan implements §5.4 of `flashml-cloud/docs/superpowers/specs/2026-07-31-deployed-multi-user-poc-design.md`. It is Plan 1 of 7 for M1.

## Global Constraints

- **Pure stdlib in the driver and the reduce path.** `flashml_workloads/kmeans_shard.py` and `sgd_trainer.py` are both pure stdlib ("runs anywhere, including `--network none` containers"). The driver must import no torch — it runs inside `flashml-api` (spec §5.4.5), which must not carry a torch dependency. Only `fedavg_worker.py` imports torch, and only inside functions.
- **Weights are JSON**, not `torch.save`: `{"<param name>": {"shape": [int, ...], "data": [float, ...]}}`. This keeps the driver torch-free. For the M1 demo model (~100k params) this is ~2 MB per delta per round, which home links handle in seconds. Binary encoding is an M2 optimization; do not add it here.
- **Determinism is the contract.** Per `sgd_trainer.py`: "resuming from step k reproduces the uninterrupted run bit-for-bit — recovery must never silently change the result." Seed every RNG from `spec["params"]["seed"]`.
- **`metrics.json` is the commit artifact.** Per the 2026-07-29 fix, only a root-level `metrics.json` sets the commit hash; the executor uploads `/work/out/` recursively. The delta goes to `/work/out/delta.json`, and `metrics.json` references it.
- **New task modules must be added to `ALLOWED_TASK_MODULES`** (`flashruntime/service/modea.py:51`) or expansion fails closed.
- **Existing suites must stay green.** Measured baseline on this machine
  (2026-07-31, before this plan): **319 passed** on the fast subset.
- **Run tests with the venv on `PATH`, not just the venv's pytest:**

      cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest

  Invoking `.venv/bin/pytest` alone makes
  `test_examples_e2e.py::test_sklearn_sweep_end_to_end` fail with
  `LaunchError: failed to start 'python'` — `LocalLauncher` spawns
  `argv[0] = "python"`, which is not resolvable unless the venv is on `PATH`.
  The same cause skips three torchrun tests. That failure is environmental;
  do not chase it.
- **Inner loop** (the full suite's torchrun DDP tests take many minutes):

      PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q \
        --ignore=tests/test_examples_e2e.py --ignore=tests/test_gpu_e2e.py

  → 319 passed in ~10 s at baseline. Run the full suite once, at Task 6.
- Never run the coordinator with more than one uvicorn worker (`HANDOFF.md` risk #5).

## File Structure

| File | Responsibility |
|---|---|
| `flashml_workloads/fedavg_weights.py` (new) | Weight/delta JSON encoding + the pure reduce function. No torch, no I/O. The only file both the worker and driver import. |
| `flashml_workloads/fedavg_worker.py` (new) | The leased task: load weights, train K local steps, emit delta. Imports torch inside functions only. |
| `flashml_workloads/fedavg_driver.py` (new) | The round loop: submit job, await quorum, reduce, repeat. Pure stdlib. |
| `flashruntime/service/modea.py` (modify) | `_expand_fedavg` + `federated_averaging` workload type + allowlist entry. |
| `tests/test_fedavg_weights.py` (new) | Reduce arithmetic, encoding round-trip, quorum/late-delta rules. |
| `tests/test_fedavg_worker.py` (new) | Task module contract: spec in, delta + metrics out. |
| `tests/test_fedavg_driver.py` (new) | Round loop against a fake coordinator: quorum, deadline, resume. |
| `tests/test_service_fedavg.py` (new) | `_expand_fedavg` produces correct TaskSpecs. |

Splitting `fedavg_weights.py` out of the driver is the decomposition decision that matters: the reduce is the part with real arithmetic to get wrong, and isolating it means it is tested without a coordinator, a job, or torch.

---

### Task 1: Weight encoding and the reduce function

**Files:**
- Create: `flashml_workloads/fedavg_weights.py`
- Test: `tests/test_fedavg_weights.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `encode(state: dict[str, tuple[list[int], list[float]]]) -> dict` — name → `{"shape", "data"}`
  - `decode(blob: dict) -> dict[str, tuple[list[int], list[float]]]`
  - `subtract(new: dict, base: dict) -> dict` — element-wise delta, same JSON shape
  - `apply_delta(base: dict, delta: dict, scale: float = 1.0) -> dict`
  - `reduce_deltas(contributions: list[tuple[dict, int]]) -> dict` — weighted mean of deltas by sample count
  - `WeightShapeMismatch(ValueError)`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fedavg_weights.py
import pytest

from flashml_workloads.fedavg_weights import (
    WeightShapeMismatch,
    apply_delta,
    decode,
    encode,
    reduce_deltas,
    subtract,
)


def _blob(**params):
    """{'w': [1.0, 2.0]} -> encoded blob with shape [len]."""
    return {k: {"shape": [len(v)], "data": list(v)} for k, v in params.items()}


def test_encode_decode_round_trip():
    state = {"w": ([2, 1], [1.5, -2.5]), "b": ([1], [0.25])}
    assert decode(encode(state)) == state


def test_subtract_is_elementwise():
    new, base = _blob(w=[3.0, 5.0]), _blob(w=[1.0, 2.0])
    assert subtract(new, base) == _blob(w=[2.0, 3.0])


def test_apply_delta_adds_scaled_delta():
    base, delta = _blob(w=[1.0, 1.0]), _blob(w=[2.0, 4.0])
    assert apply_delta(base, delta, scale=0.5) == _blob(w=[2.0, 3.0])


def test_reduce_deltas_weights_by_sample_count():
    # 100 samples say +1.0, 300 samples say +5.0 -> (100*1 + 300*5)/400 = 4.0
    got = reduce_deltas([(_blob(w=[1.0]), 100), (_blob(w=[5.0]), 300)])
    assert got["w"]["data"] == [pytest.approx(4.0)]


def test_reduce_deltas_single_contribution_is_identity():
    assert reduce_deltas([(_blob(w=[2.0, -3.0]), 7)]) == _blob(w=[2.0, -3.0])


def test_reduce_deltas_rejects_empty():
    with pytest.raises(ValueError, match="no contributions"):
        reduce_deltas([])


def test_reduce_deltas_rejects_zero_total_samples():
    # Would divide by zero and silently emit garbage weights.
    with pytest.raises(ValueError, match="zero total samples"):
        reduce_deltas([(_blob(w=[1.0]), 0)])


def test_reduce_deltas_rejects_mismatched_shapes():
    with pytest.raises(WeightShapeMismatch):
        reduce_deltas([(_blob(w=[1.0]), 1), (_blob(w=[1.0, 2.0]), 1)])


def test_reduce_deltas_rejects_mismatched_param_names():
    with pytest.raises(WeightShapeMismatch):
        reduce_deltas([(_blob(w=[1.0]), 1), (_blob(bias=[1.0]), 1)])


# --- data length must match the declared shape -------------------------------
# A blob's "shape" is a CLAIM about its "data". Checking only that two blobs
# agree on the claim lets zip() silently truncate and reduce_deltas emit
# garbage — "loads fine, trains to nonsense". Added after the Task 1 review
# found this reachable; do not delete these.

def _lying(n_declared: int, data: list[float]) -> dict:
    return {"w": {"shape": [n_declared], "data": list(data)}}


def test_subtract_rejects_data_length_disagreeing_with_shape():
    with pytest.raises(WeightShapeMismatch, match="carries"):
        subtract(_lying(2, [1.0, 2.0, 3.0]), _blob(w=[5.0, 6.0]))


def test_apply_delta_rejects_data_length_disagreeing_with_shape():
    with pytest.raises(WeightShapeMismatch, match="carries"):
        apply_delta(_blob(w=[1.0, 1.0]), _lying(2, [1.0]))


def test_reduce_deltas_rejects_short_contribution():
    # Previously returned [5.0, 5.0, 0.5] — silently wrong, no error raised.
    with pytest.raises(WeightShapeMismatch, match="carries"):
        reduce_deltas([(_blob(w=[1.0, 1.0, 1.0]), 1), (_lying(3, [9.0, 9.0]), 1)])


def test_reduce_deltas_validates_the_first_contribution_too():
    # reduce_deltas compares contributions[1:] against the first, so the
    # first blob is never the `b` argument and needs its own check.
    with pytest.raises(WeightShapeMismatch, match="carries"):
        reduce_deltas([(_lying(3, [9.0, 9.0]), 1), (_blob(w=[1.0, 1.0, 1.0]), 1)])


def test_scalar_parameter_with_empty_shape_is_accepted():
    """Guard against over-strict validation: shape [] means one element."""
    scalar = {"s": {"shape": [], "data": [4.0]}}
    assert reduce_deltas([(scalar, 1)]) == scalar
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_weights.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashml_workloads.fedavg_weights'`

- [ ] **Step 3: Write minimal implementation**

```python
# flashml_workloads/fedavg_weights.py
"""Weight/delta encoding and the federated-averaging reduce.

Weights cross the wire as JSON so the driver never imports torch: it runs
inside the cloud API (spec §5.4.5), which must stay a light service. Only
`fedavg_worker` needs torch, and it converts at the boundary.

    {"<param>": {"shape": [int, ...], "data": [float, ...]}}

`data` is the flattened tensor in row-major order; `shape` restores it.
Pure stdlib on purpose — same rule as kmeans_shard and sgd_trainer, so
this module runs on any device, including inside a --network none
container.
"""

from __future__ import annotations

import math

__all__ = [
    "WeightShapeMismatch",
    "apply_delta",
    "decode",
    "encode",
    "reduce_deltas",
    "subtract",
]


class WeightShapeMismatch(ValueError):
    """Two weight blobs do not describe the same parameter set.

    Never coerce past this: averaging mismatched blobs would emit weights
    that load fine and train to nonsense.
    """


def encode(state: dict[str, tuple[list[int], list[float]]]) -> dict:
    return {name: {"shape": list(shape), "data": list(data)}
            for name, (shape, data) in state.items()}


def decode(blob: dict) -> dict[str, tuple[list[int], list[float]]]:
    return {name: (list(p["shape"]), list(p["data"])) for name, p in blob.items()}


def _require_well_formed(blob: dict) -> None:
    """Every parameter's data length must match its declared shape.

    Without this, `zip()` in subtract/apply_delta silently truncates to the
    shorter list and reduce_deltas emits garbage — the exact "loads fine,
    trains to nonsense" failure this module exists to prevent. Checking the
    shape FIELD alone is not enough; the field is a claim about the data.
    """
    for name, p in blob.items():
        expected = math.prod(p["shape"])      # math.prod([]) == 1, i.e. a scalar
        if len(p["data"]) != expected:
            raise WeightShapeMismatch(
                f"parameter {name!r} declares shape {p['shape']} "
                f"({expected} elements) but carries {len(p['data'])}"
            )


def _require_same_params(a: dict, b: dict) -> None:
    _require_well_formed(a)
    _require_well_formed(b)
    if a.keys() != b.keys():
        raise WeightShapeMismatch(
            f"parameter names differ: {sorted(a.keys())} vs {sorted(b.keys())}"
        )
    for name in a:
        if a[name]["shape"] != b[name]["shape"]:
            raise WeightShapeMismatch(
                f"parameter {name!r} shape {a[name]['shape']} vs {b[name]['shape']}"
            )


def subtract(new: dict, base: dict) -> dict:
    _require_same_params(new, base)
    return {
        name: {
            "shape": list(new[name]["shape"]),
            "data": [x - y for x, y in zip(new[name]["data"], base[name]["data"])],
        }
        for name in new
    }


def apply_delta(base: dict, delta: dict, scale: float = 1.0) -> dict:
    _require_same_params(base, delta)
    return {
        name: {
            "shape": list(base[name]["shape"]),
            "data": [b + scale * d
                     for b, d in zip(base[name]["data"], delta[name]["data"])],
        }
        for name in base
    }


def reduce_deltas(contributions: list[tuple[dict, int]]) -> dict:
    """Sample-weighted mean of per-worker deltas (FedAvg).

    Weighting by sample count, not by worker, is what keeps the result
    equal to centralized training on the union of the shards when the
    shards are unequal — which they always are once machines differ.
    """
    if not contributions:
        raise ValueError("reduce_deltas: no contributions")
    total = sum(n for _, n in contributions)
    if total <= 0:
        raise ValueError("reduce_deltas: zero total samples")

    first = contributions[0][0]
    _require_well_formed(first)          # contributions[0] is never the `b` arg below
    for blob, _ in contributions[1:]:
        _require_same_params(first, blob)

    out: dict = {}
    for name in first:
        acc = [0.0] * len(first[name]["data"])
        for blob, n in contributions:
            w = n / total
            for i, v in enumerate(blob[name]["data"]):
                acc[i] += w * v
        out[name] = {"shape": list(first[name]["shape"]), "data": acc}
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_weights.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add flashml_workloads/fedavg_weights.py tests/test_fedavg_weights.py
git commit -m "feat(fedavg): weight encoding and sample-weighted delta reduce

Pure stdlib so the driver never imports torch. Rejects mismatched shapes,
empty contributions, and zero total samples rather than emitting weights
that load fine and train to nonsense."
```

---

### Task 2: The worker task module

**Files:**
- Create: `flashml_workloads/fedavg_worker.py`
- Test: `tests/test_fedavg_worker.py`

**Interfaces:**
- Consumes: `fedavg_weights.encode/decode/subtract` (Task 1).
- Produces:
  - CLI `python -m flashml_workloads.fedavg_worker --spec spec.json --out OUTDIR`
  - `build_model(seed: int, in_dim: int, hidden: int, out_dim: int)` — returns a torch module
  - `state_to_blob(model) -> dict` / `blob_to_state(model, blob) -> None`
  - `run_worker(spec: dict, outdir: Path) -> dict` — writes `delta.json` + returns the metrics dict
- Writes `OUTDIR/delta.json` and `OUTDIR/metrics.json`. `metrics.json` carries `{"round", "shard", "samples", "loss", "local_steps", "delta_file": "delta.json"}`.

`spec.json` shape (mirrors `kmeans_shard`'s contract):
```json
{"params": {"round": 0, "shard": 1, "num_shards": 4, "local_steps": 50,
            "lr": 0.05, "batch_size": 32, "seed": 0,
            "in_dim": 16, "hidden": 32, "out_dim": 2, "dataset_size": 512},
 "inputs": {"weights": "/work/inputs/weights.json"}}
```

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fedavg_worker.py
import json

import pytest

torch = pytest.importorskip("torch")

from flashml_workloads import fedavg_worker
from flashml_workloads.fedavg_weights import apply_delta


def _spec(tmp_path, weights_path=None, **overrides):
    params = {
        "round": 0, "shard": 0, "num_shards": 2, "local_steps": 5,
        "lr": 0.05, "batch_size": 8, "seed": 0,
        "in_dim": 4, "hidden": 8, "out_dim": 2, "dataset_size": 64,
    }
    params.update(overrides)
    spec = {"params": params, "inputs": {}}
    if weights_path is not None:
        spec["inputs"]["weights"] = str(weights_path)
    return spec


def _initial_weights(tmp_path, **overrides):
    """Round 0 has no input weights; the worker seeds them deterministically."""
    spec = _spec(tmp_path, **overrides)
    model = fedavg_worker.build_model(
        seed=spec["params"]["seed"], in_dim=spec["params"]["in_dim"],
        hidden=spec["params"]["hidden"], out_dim=spec["params"]["out_dim"],
    )
    return fedavg_worker.state_to_blob(model)


def test_writes_delta_and_metrics(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    metrics = fedavg_worker.run_worker(_spec(tmp_path), out)

    assert (out / "delta.json").exists()
    assert (out / "metrics.json").exists()
    assert metrics["delta_file"] == "delta.json"
    assert metrics["samples"] == 32          # 64 rows / 2 shards
    assert metrics["local_steps"] == 5
    assert isinstance(metrics["loss"], float)


def test_delta_matches_declared_parameter_set(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    fedavg_worker.run_worker(_spec(tmp_path), out)
    delta = json.loads((out / "delta.json").read_text())
    assert delta.keys() == _initial_weights(tmp_path).keys()


def test_delta_is_nonzero_after_training(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    fedavg_worker.run_worker(_spec(tmp_path), out)
    delta = json.loads((out / "delta.json").read_text())
    assert any(abs(v) > 1e-9 for p in delta.values() for v in p["data"])


def test_same_seed_and_shard_is_deterministic(tmp_path):
    outs = []
    for name in ("a", "b"):
        d = tmp_path / name
        d.mkdir()
        fedavg_worker.run_worker(_spec(tmp_path), d)
        outs.append(json.loads((d / "delta.json").read_text()))
    assert outs[0] == outs[1]


def test_different_shards_see_different_data(tmp_path):
    outs = []
    for shard in (0, 1):
        d = tmp_path / f"s{shard}"
        d.mkdir()
        fedavg_worker.run_worker(_spec(tmp_path, shard=shard), d)
        outs.append(json.loads((d / "delta.json").read_text()))
    assert outs[0] != outs[1]


def test_resumes_from_supplied_weights(tmp_path):
    """A round-1 worker must start from the driver's weights, not from seed."""
    base = _initial_weights(tmp_path)
    moved = apply_delta(base, base, scale=1.0)     # deliberately different weights
    wpath = tmp_path / "weights.json"
    wpath.write_text(json.dumps(moved))

    out = tmp_path / "out"
    out.mkdir()
    fedavg_worker.run_worker(_spec(tmp_path, weights_path=wpath, round=1), out)
    delta = json.loads((out / "delta.json").read_text())

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    fedavg_worker.run_worker(_spec(tmp_path), fresh)
    assert delta != json.loads((fresh / "delta.json").read_text())


def test_batches_wrap_and_stay_full_size(tmp_path):
    """batch_size that does not divide the shard evenly must still yield
    full-size batches. Slicing (rather than wrapping) silently produced short
    batches and reweighted those steps' loss."""
    import flashml_workloads.fedavg_worker as w

    sizes = []
    real = w.build_model

    def spy(*a, **k):
        model = real(*a, **k)
        fwd = model.forward

        def wrapped(t):
            sizes.append(int(t.shape[0]))
            return fwd(t)

        model.forward = wrapped
        return model

    # 64 rows / 2 shards = 32 samples; batch 12 does not divide 32.
    w.build_model = spy
    try:
        out = tmp_path / "out"
        out.mkdir()
        w.run_worker(_spec(tmp_path, batch_size=12, local_steps=4), out)
    finally:
        w.build_model = real
    assert sizes == [12, 12, 12, 12], f"short batch produced: {sizes}"


def test_empty_shard_raises_instead_of_dividing_by_zero(tmp_path):
    """num_shards > dataset_size is a legitimate spec that leaves trailing
    shards empty; it must fail loudly, not ZeroDivisionError."""
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(ValueError, match="empty"):
        fedavg_worker.run_worker(
            _spec(tmp_path, dataset_size=2, num_shards=8, shard=7), out
        )


def test_rejects_weights_with_wrong_shapes(tmp_path):
    from flashml_workloads.fedavg_weights import WeightShapeMismatch

    wpath = tmp_path / "weights.json"
    wpath.write_text(json.dumps({"nonsense": {"shape": [1], "data": [0.0]}}))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(WeightShapeMismatch):
        fedavg_worker.run_worker(_spec(tmp_path, weights_path=wpath, round=1), out)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_worker.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashml_workloads.fedavg_worker'`

(If it reports `skipped` instead, torch is not installed in the venv: `uv pip install torch`.)

- [ ] **Step 3: Write minimal implementation**

```python
# flashml_workloads/fedavg_worker.py
"""One federated-averaging round on one shard, runnable as a leased task.

Executor contract (same as kmeans_shard / sgd_trainer):

    python -m flashml_workloads.fedavg_worker --spec spec.json --out OUTDIR

`spec.json`:
    params: round, shard, num_shards, local_steps, lr, batch_size, seed,
            in_dim, hidden, out_dim, dataset_size
    inputs: weights (optional path to the round's weights JSON; absent on
            round 0, where the seed determines the starting point)

Outputs `OUTDIR/delta.json` (this worker's weight change) and
`OUTDIR/metrics.json` (the commit artifact — only a root-level
metrics.json sets the commit hash).

Why a delta and not the new weights: the driver averages contributions,
and averaging deltas keeps the arithmetic correct when a worker joins a
round late with stale weights — its delta is still a valid direction from
the weights it actually saw.

torch is imported inside functions so this module can be inspected (and
the rest of flashml_workloads used) without torch installed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flashml_workloads.fedavg_weights import decode, encode, subtract


def build_model(seed: int, in_dim: int, hidden: int, out_dim: int):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, out_dim)
    )


def state_to_blob(model) -> dict:
    state = {}
    for name, t in model.state_dict().items():
        state[name] = (list(t.shape), [float(x) for x in t.flatten().tolist()])
    return encode(state)


def blob_to_state(model, blob: dict) -> None:
    """Load a weight blob into `model`, failing loudly on any mismatch."""
    import torch

    from flashml_workloads.fedavg_weights import WeightShapeMismatch

    current = state_to_blob(model)
    if current.keys() != blob.keys():
        raise WeightShapeMismatch(
            f"weights name mismatch: expected {sorted(current)}, got {sorted(blob)}"
        )
    new_state = {}
    for name, (shape, data) in decode(blob).items():
        if shape != list(current[name]["shape"]):
            raise WeightShapeMismatch(
                f"parameter {name!r} shape {shape} != {current[name]['shape']}"
            )
        new_state[name] = torch.tensor(data, dtype=torch.float32).reshape(shape)
    model.load_state_dict(new_state)


def _make_shard(params: dict):
    """Deterministic synthetic data, sliced by shard.

    Strided slicing (`x[shard::num_shards]`) rather than contiguous blocks
    so every shard sees the same label distribution — a contiguous split of
    sorted data would give workers disjoint classes and FedAvg would
    diverge for reasons unrelated to the runtime.
    """
    import torch

    g = torch.Generator().manual_seed(params["seed"])
    n, d = params["dataset_size"], params["in_dim"]
    x = torch.randn(n, d, generator=g)
    w = torch.randn(d, 1, generator=g)
    y = ((x @ w).squeeze(1) > 0).long()
    shard, num = params["shard"], params["num_shards"]
    return x[shard::num], y[shard::num]


def run_worker(spec: dict, outdir: Path) -> dict:
    import torch
    from torch import nn

    p = spec["params"]
    outdir = Path(outdir)

    model = build_model(p["seed"], p["in_dim"], p["hidden"], p["out_dim"])
    weights_path = (spec.get("inputs") or {}).get("weights")
    if weights_path:
        blob_to_state(model, json.loads(Path(weights_path).read_text()))
    base = state_to_blob(model)

    x, y = _make_shard(p)
    samples = int(x.shape[0])
    if samples == 0:
        # Reachable from a legitimate spec: num_shards > dataset_size leaves
        # trailing shards empty. Fail loudly here rather than dividing by zero
        # in the batch index below.
        raise ValueError(
            f"shard {p['shard']} of {p['num_shards']} is empty "
            f"(dataset_size={p['dataset_size']}): fewer rows than shards"
        )
    opt = torch.optim.SGD(model.parameters(), lr=p["lr"])
    loss_fn = nn.CrossEntropyLoss()
    batch = p["batch_size"]

    # Batch indices WRAP, exactly as sgd_trainer.py:82-83 does
    # (`idx = [(base + i) % n ...]`), so every step sees a full-size batch even
    # when batch does not divide samples. Slicing `x[start:start+batch]`
    # instead would silently yield short batches and reweight those steps'
    # loss. Indexed by step with no RNG, so a retried attempt reproduces the
    # delta exactly.
    last_loss = 0.0
    for step in range(p["local_steps"]):
        base = (step * batch) % samples
        idx = [(base + i) % samples for i in range(batch)]
        xb, yb = x[idx], y[idx]
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        last_loss = float(loss.item())

    delta = subtract(state_to_blob(model), base)
    (outdir / "delta.json").write_text(json.dumps(delta))

    metrics = {
        "round": p["round"],
        "shard": p["shard"],
        "samples": samples,
        "loss": last_loss,
        "local_steps": p["local_steps"],
        "delta_file": "delta.json",
    }
    # sort_keys matches every sibling task module (sgd_trainer.py:113,
    # kmeans_shard.py:64, sklearn_trial.py:84). metrics.json is the commit
    # artifact whose sha256 the coordinator validates, so byte-stability
    # is load-bearing, not style.
    (outdir / "metrics.json").write_text(json.dumps(metrics, sort_keys=True))
    return metrics


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    run_worker(json.loads(Path(args.spec).read_text()), outdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_worker.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add flashml_workloads/fedavg_worker.py tests/test_fedavg_worker.py
git commit -m "feat(fedavg): per-shard worker task module

Loads the round's weights, trains local_steps on a strided shard, emits a
delta plus the metrics.json commit artifact. Strided (not contiguous)
sharding so every worker sees the same label distribution. torch imported
inside functions only."
```

---

### Task 3: Job expansion for the `federated_averaging` workload type

**Files:**
- Modify: `flashruntime/service/modea.py:51-55` (allowlist), and add `_expand_fedavg` beside `_expand_kmeans` (`modea.py:142`)
- Test: `tests/test_service_fedavg.py`

**Interfaces:**
- Consumes: `flashml_workloads.fedavg_worker` (Task 2) by module name only.
- Produces: `_expand_fedavg(job_id: str, spec: JobSpec) -> list[TaskSpec]`. Task ids are `shard-000`…; `commit_key` is `jobs/{job_id}/{task_id}/metrics.json`.

Workload parameters: `round`, `num_shards`, `weights` (an `artifact://` URI, absent on round 0), plus the worker params passed through (`local_steps`, `lr`, `batch_size`, `seed`, `in_dim`, `hidden`, `out_dim`, `dataset_size`), and `lease_seconds`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_service_fedavg.py
import pytest

from flashruntime.protocol.v1alpha1 import JobSpec
from flashruntime.service.modea import ExpansionError, expand_tasks


def _spec(**params):
    base = {"round": 0, "num_shards": 3, "local_steps": 5, "lr": 0.05,
            "batch_size": 8, "seed": 0, "in_dim": 4, "hidden": 8,
            "out_dim": 2, "dataset_size": 64}
    base.update(params)
    return JobSpec.model_validate({
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": "fedavg"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": "local/tier1", "tag": "dev"},
            "workload": {"type": "federated_averaging", "parameters": base},
        },
    })


def test_expands_one_task_per_shard():
    tasks = expand_tasks("job-1", _spec())
    assert [t.task_id for t in tasks] == ["shard-000", "shard-001", "shard-002"]


def test_each_task_carries_its_shard_index_and_the_worker_module():
    tasks = expand_tasks("job-1", _spec())
    for i, t in enumerate(tasks):
        assert t.payload["module"] == "flashml_workloads.fedavg_worker"
        assert t.payload["params"]["shard"] == i
        assert t.payload["params"]["num_shards"] == 3


def test_commit_key_is_root_metrics_json():
    tasks = expand_tasks("job-1", _spec())
    assert tasks[0].commit_key == "jobs/job-1/shard-000/metrics.json"


def test_round_zero_declares_no_weights_input():
    tasks = expand_tasks("job-1", _spec(round=0))
    assert "weights" not in tasks[0].payload["inputs"]


def test_later_round_declares_the_weights_artifact():
    tasks = expand_tasks("job-1", _spec(round=2, weights="artifact://jobs/j/r1/weights.json"))
    assert tasks[0].payload["inputs"]["weights"] == "artifact://jobs/j/r1/weights.json"
    assert tasks[0].payload["params"]["round"] == 2


def test_weights_must_be_an_artifact_uri():
    with pytest.raises(ExpansionError, match="artifact://"):
        expand_tasks("job-1", _spec(round=1, weights="/etc/passwd"))


def test_isolation_is_stamped_so_placement_can_fail_closed():
    tasks = expand_tasks("job-1", _spec())
    assert "tier" in tasks[0].payload["isolation"]


def test_rejects_zero_shards():
    with pytest.raises(ExpansionError, match="num_shards"):
        expand_tasks("job-1", _spec(num_shards=0))


def test_rejects_a_spec_missing_worker_parameters():
    """fedavg_worker reads every one of these unconditionally. Omitting the
    check would defer the failure to a KeyError inside a container on a
    volunteer's machine, burning an attempt and looking like a node fault."""
    spec = _spec()
    del spec.spec.workload.parameters["lr"]
    with pytest.raises(ExpansionError, match="lr"):
        expand_tasks("job-1", spec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_fedavg.py -v`
Expected: FAIL — `ExpansionError: lease backend supports workload types 'hyperparameter_search' and 'sharded_kmeans', got 'federated_averaging'`

- [ ] **Step 3: Write minimal implementation**

In `flashruntime/service/modea.py`, add to the allowlist at line 51:

```python
ALLOWED_TASK_MODULES = {
    "flashml_workloads.sklearn_trial",
    "flashml_workloads.kmeans_shard",
    "flashml_workloads.sgd_trainer",
    "flashml_workloads.fedavg_worker",
}
```

In `expand_tasks`, beside the existing `sharded_kmeans` branch:

```python
    if workload.type == "federated_averaging":
        return _expand_fedavg(job_id, spec)
```

Update the error message in the same function so it stays truthful:

```python
    if workload.type != "hyperparameter_search":
        raise ExpansionError(
            f"lease backend supports workload types 'hyperparameter_search', "
            f"'sharded_kmeans' and 'federated_averaging', got '{workload.type}'"
        )
```

Add beside `_expand_kmeans`:

```python
def _expand_fedavg(job_id: str, spec: JobSpec) -> list[TaskSpec]:
    """One federated-averaging *round*: one task per shard, each training
    locally from the round's broadcast weights. The driver
    (`flashml_workloads.fedavg_driver`) reduces the deltas and submits the
    next round — same stage-composition pattern as `_expand_kmeans`.
    """
    p = spec.spec.workload.parameters
    num_shards = int(p.get("num_shards", 0))
    if num_shards < 1:
        raise ExpansionError(f"federated_averaging needs num_shards >= 1, got {num_shards}")

    inputs: dict[str, str] = {}
    weights = p.get("weights")
    if weights is not None:
        if not str(weights).startswith("artifact://"):
            raise ExpansionError(
                f"input 'weights' must be an artifact:// URI, got {weights!r}"
            )
        inputs["weights"] = weights

    isolation = {
        "tier": spec.spec.isolation.tier,
        "allowFallback": spec.spec.isolation.allowFallback,
    }
    # Every one of these is read unconditionally by fedavg_worker. Dropping a
    # missing key here would defer the failure to a KeyError inside a container
    # on a volunteer's machine, where it burns an attempt and reads as a node
    # fault rather than a bad submission. Fail at expansion instead.
    worker_keys = ("local_steps", "lr", "batch_size", "seed",
                   "in_dim", "hidden", "out_dim", "dataset_size")
    missing = [k for k in worker_keys if k not in p]
    if missing:
        raise ExpansionError(
            f"federated_averaging is missing required parameters: {sorted(missing)}"
        )

    tasks = []
    for shard in range(num_shards):
        task_id = f"shard-{shard:03d}"
        params = {k: p[k] for k in worker_keys}
        params.update({"round": int(p.get("round", 0)),
                       "shard": shard, "num_shards": num_shards})
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_id=job_id,
                commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                lease_seconds=float(p.get("lease_seconds", 120.0)),
                payload={
                    "module": "flashml_workloads.fedavg_worker",
                    "params": params,
                    "inputs": inputs,
                    "output_prefix": f"jobs/{job_id}/{task_id}/",
                    "task_id": task_id,
                    "image": spec.spec.image.reference,
                    "isolation": isolation,
                },
            )
        )
    return tasks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_fedavg.py tests/test_service_modea.py -v`
Expected: 9 new passed; `test_service_modea.py` unchanged and green

- [ ] **Step 5: Commit**

```bash
git add flashruntime/service/modea.py tests/test_service_fedavg.py
git commit -m "feat(fedavg): expand federated_averaging jobs into per-shard tasks

One task per shard per round, mirroring _expand_kmeans. Weights arrive as
an artifact:// input (absent on round 0). Adds fedavg_worker to
ALLOWED_TASK_MODULES so expansion does not fail closed."
```

---

### Task 4: Driver round loop with quorum

**Files:**
- Create: `flashml_workloads/fedavg_driver.py`
- Test: `tests/test_fedavg_driver.py`

**Interfaces:**
- Consumes: `fedavg_weights.reduce_deltas/apply_delta/encode` (Task 1); the `federated_averaging` workload type (Task 3).
- Produces:
  - `run_fedavg(base_url, *, rounds, num_shards, min_participants, worker_params, round_timeout_s=600.0, poll_seconds=1.0, on_round=None) -> dict`
    returning `{"weights": blob, "history": [RoundResult, ...], "job_ids": [str, ...]}`
  - `RoundResult` TypedDict: `{"round": int, "participants": int, "mean_loss": float, "job_id": str}`
  - `QuorumNotMet(RuntimeError)`

Quorum semantics, exactly:
- The driver polls the round's job. Once **`min_participants`** shard tasks have committed, it aggregates immediately and does **not** wait for the rest.
- Deltas that commit after aggregation are discarded — never applied to a later round (they were computed from weights that no longer exist).
- If the round deadline passes with fewer than `min_participants` commits, raise `QuorumNotMet`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fedavg_driver.py
import json

import pytest

from flashml_workloads.fedavg_driver import ArtifactNotFound, QuorumNotMet, run_fedavg


class FakeCoordinator:
    """Stands in for the HTTP coordinator.

    `commits[r]` lists (shard, delta_value, samples) that appear for round r.
    A shard absent from a round never commits — that is how a straggler or a
    closed laptop is simulated.
    """

    def __init__(self, commits, param_name="w"):
        self.commits = commits
        self.param_name = param_name
        self.submitted = []
        self.uploaded = {}
        self._round = -1

    def submit(self, body):
        self._round = body["spec"]["workload"]["parameters"]["round"]
        job_id = f"job-r{self._round}"
        self.submitted.append((job_id, body))
        return {"job_id": job_id}

    def job_state(self, job_id):
        return "RUNNING"

    def artifacts(self, job_id):
        r = int(job_id.split("r")[1])
        out = []
        for shard, _, _ in self.commits.get(r, []):
            out.append({"key": f"jobs/{job_id}/shard-{shard:03d}/metrics.json"})
            out.append({"key": f"jobs/{job_id}/shard-{shard:03d}/delta.json"})
        return out

    def get_artifact(self, key):
        # Anything previously PUT wins. Without this the fake would
        # re-derive a *delta* for a weights key and a resume test could pass
        # against entirely the wrong data path.
        if key in self.uploaded:
            return self.uploaded[key]
        parts = key.split("/")
        if len(parts) < 3 or not parts[2].startswith("shard-"):
            raise ArtifactNotFound(key)
        r = int(parts[1].split("r")[1])
        shard = int(parts[2].split("-")[1])
        match = [(v, n) for s, v, n in self.commits.get(r, []) if s == shard]
        if not match:
            raise ArtifactNotFound(key)
        value, samples = match[0]
        if key.endswith("metrics.json"):
            return {"round": r, "shard": shard, "samples": samples,
                    "loss": 1.0 / (r + 1), "local_steps": 5,
                    "delta_file": "delta.json"}
        return {self.param_name: {"shape": [1], "data": [value]}}

    def put_artifact(self, key, body):
        self.uploaded[key] = body


def _params():
    return {"local_steps": 5, "lr": 0.05, "batch_size": 8, "seed": 0,
            "in_dim": 4, "hidden": 8, "out_dim": 2, "dataset_size": 64}


def test_runs_requested_number_of_rounds():
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(3)})
    result = run_fedavg(fake, rounds=3, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert len(result["history"]) == 3
    assert [h["round"] for h in result["history"]] == [0, 1, 2]


def test_weights_accumulate_reduced_deltas():
    # Every round both shards report +1.0 -> weights walk 0 -> 1 -> 2 -> 3.
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(3)})
    result = run_fedavg(fake, rounds=3, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["weights"]["w"]["data"] == [pytest.approx(3.0)]


def test_aggregates_on_quorum_without_waiting_for_stragglers():
    # 3 shards, only 2 ever commit; quorum of 2 must still complete the round.
    fake = FakeCoordinator({0: [(0, 2.0, 10), (2, 4.0, 10)]})
    result = run_fedavg(fake, rounds=1, num_shards=3, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["history"][0]["participants"] == 2
    assert result["weights"]["w"]["data"] == [pytest.approx(3.0)]


def test_quorum_uses_sample_weighting():
    fake = FakeCoordinator({0: [(0, 1.0, 100), (1, 5.0, 300)]})
    result = run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["weights"]["w"]["data"] == [pytest.approx(4.0)]


def test_raises_when_quorum_never_met():
    fake = FakeCoordinator({0: [(0, 1.0, 10)]})
    with pytest.raises(QuorumNotMet, match="1 of 2"):
        run_fedavg(fake, rounds=1, num_shards=3, min_participants=2,
                   worker_params=_params(), round_timeout_s=0.2, poll_seconds=0.01,
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_each_round_declares_previous_round_weights_as_input():
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(2)})
    run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
               worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    first = fake.submitted[0][1]["spec"]["workload"]["parameters"]
    second = fake.submitted[1][1]["spec"]["workload"]["parameters"]
    assert "weights" not in first
    assert second["weights"].startswith("artifact://")


def test_on_round_callback_receives_progress():
    seen = []
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(2)})
    run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
               worker_params=_params(), on_round=seen.append,
               initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert [s["round"] for s in seen] == [0, 1]
    assert all("mean_loss" in s for s in seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_driver.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashml_workloads.fedavg_driver'`

- [ ] **Step 3: Write minimal implementation**

```python
# flashml_workloads/fedavg_driver.py
"""Federated averaging as a sequence of lease jobs.

One round = one Mode A job (N independent shard tasks); the driver reduces
the shard deltas into new weights and submits the next round. Same
stage-composition pattern as `kmeans_driver` — "pipelines are jobs chained
by a driver, not a new execution mode" — so a dead worker costs one shard
retry and a dead driver resumes from the last completed round.

The one deliberate difference from kmeans_driver: it required *every*
shard (`if len(partials) != len(shard_uris): raise`). This driver
aggregates on a QUORUM. Volunteer machines are unequal and unreliable by
definition; requiring all of them would let one closed laptop stall every
participant's round. Deltas arriving after aggregation are DISCARDED, never
carried into a later round — they were computed against weights that no
longer exist, and applying them would silently corrupt the average.

Pure stdlib: this runs inside the cloud API, which must not carry torch.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Protocol, TypedDict

from flashml_workloads.fedavg_weights import apply_delta, reduce_deltas

__all__ = ["ArtifactNotFound", "Coordinator", "QuorumNotMet", "RoundResult",
           "run_fedavg"]


class QuorumNotMet(RuntimeError):
    """A round's deadline passed with too few committed shards."""


class ArtifactNotFound(LookupError):
    """No artifact exists at that key.

    A named exception, not a bare `Exception` catch: `resume_state` must
    distinguish "this round never completed" (expected, keep looking) from
    "the coordinator is unreachable" (fatal, must not look like round 0).
    """


class RoundResult(TypedDict):
    round: int
    participants: int
    mean_loss: float
    job_id: str


class Coordinator(Protocol):
    """The coordinator operations the driver needs.

    Declared as a Protocol so tests substitute a fake without HTTP, and so
    the cloud API can pass an implementation that adds auth headers.
    """

    def submit(self, body: dict) -> dict: ...
    def job_state(self, job_id: str) -> str: ...
    def artifacts(self, job_id: str) -> list[dict]: ...
    def get_artifact(self, key: str) -> Any: ...
    def put_artifact(self, key: str, body: Any) -> None: ...


def _round_body(round_idx: int, num_shards: int, worker_params: dict,
                weights_uri: str | None, lease_seconds: float) -> dict:
    params: dict[str, Any] = dict(worker_params)
    params.update({"round": round_idx, "num_shards": num_shards,
                   "lease_seconds": lease_seconds})
    if weights_uri is not None:
        params["weights"] = weights_uri
    return {
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": f"fedavg-r{round_idx:03d}"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": "local/tier1", "tag": "dev"},
            "workload": {"type": "federated_averaging", "parameters": params},
        },
    }


def _collect(coord: Coordinator, job_id: str) -> list[tuple[dict, int, float]]:
    """Committed (delta, samples, loss) triples for a round, so far."""
    out = []
    for a in coord.artifacts(job_id):
        key = a["key"]
        if not key.endswith("metrics.json"):
            continue
        metrics = coord.get_artifact(key)
        delta_key = key.rsplit("/", 1)[0] + "/" + metrics.get("delta_file", "delta.json")
        out.append((coord.get_artifact(delta_key),
                    int(metrics["samples"]), float(metrics["loss"])))
    return out


def run_fedavg(
    coord: Coordinator,
    *,
    rounds: int,
    num_shards: int,
    min_participants: int,
    worker_params: dict,
    initial_weights: dict,
    round_timeout_s: float = 600.0,
    poll_seconds: float = 1.0,
    lease_seconds: float = 120.0,
    on_round: Callable[[RoundResult], None] | None = None,
) -> dict:
    if min_participants < 1:
        raise ValueError("min_participants must be >= 1")
    if min_participants > num_shards:
        raise ValueError(
            f"min_participants {min_participants} exceeds num_shards {num_shards}"
        )

    weights = initial_weights
    weights_uri: str | None = None
    history: list[RoundResult] = []
    job_ids: list[str] = []

    for r in range(rounds):
        job_id = coord.submit(
            _round_body(r, num_shards, worker_params, weights_uri, lease_seconds)
        )["job_id"]
        job_ids.append(job_id)

        deadline = time.monotonic() + round_timeout_s
        collected: list[tuple[dict, int, float]] = []
        while True:
            collected = _collect(coord, job_id)
            if len(collected) >= min_participants:
                break
            state = coord.job_state(job_id)
            if state in ("FAILED", "CANCELLED"):
                raise QuorumNotMet(
                    f"round {r}: job {job_id} ended {state} with "
                    f"{len(collected)} of {min_participants} needed"
                )
            if time.monotonic() > deadline:
                raise QuorumNotMet(
                    f"round {r}: timed out with {len(collected)} of "
                    f"{min_participants} needed ({num_shards} shards dispatched)"
                )
            time.sleep(poll_seconds)

        # Anything committing from here on is discarded by construction: we
        # never re-read this job after aggregating.
        reduced = reduce_deltas([(d, n) for d, n, _ in collected])
        weights = apply_delta(weights, reduced)

        weights_key = f"jobs/{job_id}/round-{r:03d}/weights.json"
        coord.put_artifact(weights_key, weights)
        weights_uri = f"artifact://{weights_key}"

        result: RoundResult = {
            "round": r,
            "participants": len(collected),
            "mean_loss": sum(loss for _, _, loss in collected) / len(collected),
            "job_id": job_id,
        }
        history.append(result)
        if on_round is not None:
            on_round(result)

    return {"weights": weights, "history": history, "job_ids": job_ids}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_driver.py -v`
Expected: 9 passed

- [ ] **Step 5: Commit**

```bash
git add flashml_workloads/fedavg_driver.py tests/test_fedavg_driver.py
git commit -m "feat(fedavg): round-loop driver with quorum aggregation

kmeans_driver's control flow with reduce swapped for a sample-weighted
delta mean. Aggregates once min_participants shards commit instead of
requiring all of them, so one closed laptop cannot stall a round. Deltas
arriving after aggregation are discarded, never applied to a later round.
Pure stdlib — the driver runs inside the cloud API, which carries no torch."
```

---

### Task 5: HTTP coordinator adapter and driver resume

**Files:**
- Modify: `flashml_workloads/fedavg_driver.py`
- Test: `tests/test_fedavg_driver.py` (append)

**Interfaces:**
- Consumes: `Coordinator` protocol (Task 4).
- Produces:
  - `HttpCoordinator(base_url: str, headers: dict[str, str] | None = None)` implementing `Coordinator`
  - `resume_state(coord: Coordinator, job_ids: list[str]) -> tuple[int, dict, str | None]` returning `(next_round, weights, weights_uri)`
  - `run_fedavg(..., start_round: int = 0)` — resumes rather than restarting

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_fedavg_driver.py
from flashml_workloads.fedavg_driver import HttpCoordinator, resume_state


def test_http_coordinator_sends_auth_headers(monkeypatch):
    captured = {}

    def fake_request(method, url, data=None, headers=None, timeout=None):
        captured.update({"method": method, "url": url, "headers": headers or {}})
        return {"job_id": "job-r0"}

    coord = HttpCoordinator("http://c:8100", headers={"Authorization": "Bearer t"})
    monkeypatch.setattr(coord, "_request", fake_request)
    coord.submit({"spec": {}})
    assert captured["headers"]["Authorization"] == "Bearer t"
    assert captured["url"] == "http://c:8100/v1alpha1/jobs"


def test_resume_state_finds_last_completed_round():
    # 42.0 is deliberately unlike any delta value in `commits` — if the fake
    # ever re-derives a delta for this key instead of returning what was PUT,
    # this assertion must fail rather than coincide.
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {
        "w": {"shape": [1], "data": [42.0]}
    }
    next_round, weights, uri = resume_state(fake, ["job-r0"])
    assert next_round == 1
    assert weights["w"]["data"] == [42.0]
    assert uri == "artifact://jobs/job-r0/round-000/weights.json"


def test_resume_state_picks_the_newest_round_not_the_first():
    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {"w": {"shape": [1], "data": [1.0]}}
    fake.uploaded["jobs/job-r1/round-001/weights.json"] = {"w": {"shape": [1], "data": [2.0]}}
    next_round, weights, _ = resume_state(fake, ["job-r0", "job-r1"])
    assert next_round == 2
    assert weights["w"]["data"] == [2.0]


def test_resume_state_skips_a_round_that_never_aggregated():
    # Round 1 was submitted but crashed before writing weights: resume at 1.
    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {"w": {"shape": [1], "data": [7.0]}}
    next_round, weights, _ = resume_state(fake, ["job-r0", "job-r1"])
    assert next_round == 1
    assert weights["w"]["data"] == [7.0]


def test_resume_state_propagates_transport_errors():
    """An unreachable coordinator must NOT look like 'no rounds completed' —
    that would silently restart a finished run from scratch."""
    class Unreachable(FakeCoordinator):
        def get_artifact(self, key):
            raise ConnectionError("coordinator unreachable")

    with pytest.raises(ConnectionError):
        resume_state(Unreachable({}), ["job-r0"])


def test_resume_state_on_empty_history_starts_at_zero():
    fake = FakeCoordinator({})
    assert resume_state(fake, []) == (0, {}, None)


def test_run_fedavg_resumes_from_start_round():
    fake = FakeCoordinator({1: [(0, 1.0, 10), (1, 1.0, 10)]})
    result = run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
                        worker_params=_params(), start_round=1,
                        initial_weights={"w": {"shape": [1], "data": [5.0]}},
                        weights_uri="artifact://jobs/job-r0/round-000/weights.json")
    # Only round 1 runs; weights walk 5.0 -> 6.0
    assert [h["round"] for h in result["history"]] == [1]
    assert result["weights"]["w"]["data"] == [pytest.approx(6.0)]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_driver.py -v`
Expected: FAIL — `ImportError: cannot import name 'HttpCoordinator'`

- [ ] **Step 3: Write minimal implementation**

Add to `flashml_workloads/fedavg_driver.py`:

```python
import json
import urllib.error
import urllib.request


class HttpCoordinator:
    """`Coordinator` over the coordinator's HTTP API.

    `headers` carries the caller's credentials — the cloud API passes the
    machine/service token here rather than the driver knowing anything
    about auth.
    """

    def __init__(self, base_url: str, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})

    def _request(self, method: str, url: str, data: bytes | None = None,
                 headers: dict | None = None, timeout: float | None = 60.0):
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def submit(self, body: dict) -> dict:
        return self._request("POST", f"{self.base_url}/v1alpha1/jobs",
                             data=json.dumps(body).encode(), headers=self.headers)

    def job_state(self, job_id: str) -> str:
        return self._request("GET", f"{self.base_url}/v1alpha1/jobs/{job_id}",
                             headers=self.headers)["state"]

    def artifacts(self, job_id: str) -> list[dict]:
        return self._request("GET", f"{self.base_url}/v1alpha1/jobs/{job_id}/artifacts",
                             headers=self.headers)

    def get_artifact(self, key: str):
        try:
            return self._request("GET", f"{self.base_url}/v1alpha1/artifacts/{key}",
                                 headers=self.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ArtifactNotFound(key) from None
            raise   # 5xx / auth failures are NOT "round never completed"

    def put_artifact(self, key: str, body) -> None:
        self._request("PUT", f"{self.base_url}/v1alpha1/artifacts/{key}",
                      data=json.dumps(body).encode(), headers=self.headers)


def resume_state(coord: Coordinator, job_ids: list[str]) -> tuple[int, dict, str | None]:
    """Where to restart after a driver crash.

    `job_ids[r]` is the job submitted for round r — they are appended in
    order, so the round index and the list index are the same thing. Rounds
    are idempotent: the weights artifact is written only AFTER a round
    aggregates, so the newest one that exists names the last round that
    fully completed.

    Only ArtifactNotFound is swallowed. A transport error must propagate:
    silently treating an unreachable coordinator as "no rounds done" would
    restart a finished run from scratch.
    """
    for r in range(len(job_ids) - 1, -1, -1):
        key = f"jobs/{job_ids[r]}/round-{r:03d}/weights.json"
        try:
            weights = coord.get_artifact(key)
        except ArtifactNotFound:
            continue
        if weights:
            return r + 1, weights, f"artifact://{key}"
    return 0, {}, None
```

Change `run_fedavg`'s signature to accept `start_round: int = 0` and
`weights_uri: str | None = None`, and replace the loop header:

```python
    weights = initial_weights
    history: list[RoundResult] = []
    job_ids: list[str] = []

    for r in range(start_round, rounds):
```

(delete the local `weights_uri: str | None = None` initialization — it now
comes from the parameter).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_fedavg_driver.py -v`
Expected: 14 passed

- [ ] **Step 5: Commit**

```bash
git add flashml_workloads/fedavg_driver.py tests/test_fedavg_driver.py
git commit -m "feat(fedavg): HTTP coordinator adapter and driver resume

HttpCoordinator takes caller-supplied headers so the cloud API injects
credentials without the driver knowing about auth. resume_state finds the
newest committed round-weights artifact; rounds are idempotent because
weights are written only after aggregation."
```

---

### Task 6: Local two-agent convergence demo

**Files:**
- Create: `scripts/fedavg_local_demo.py`
- Test: `tests/integration/test_fedavg_e2e.py`
- Modify: `docs/site/guides/` — add `federated-averaging.md`; register it if `scripts/build_docs.py --check` requires an index entry

**Interfaces:**
- Consumes: everything above.
- Produces: `scripts/fedavg_local_demo.py` — starts a coordinator, launches 2 subprocess-tier agents, runs 5 rounds, prints per-round participants and loss, asserts loss decreased.

This is the task that proves the premise. Everything before it is unit-tested arithmetic.

- [ ] **Step 1: Write the failing test**

```python
# tests/integration/test_fedavg_e2e.py
"""Federated averaging across two real agent processes.

Marked integration because it spawns a coordinator and two agents over real
sockets. The point is convergence across process boundaries — the unit
tests already pin the arithmetic.
"""
import pytest

pytestmark = pytest.mark.integration

torch = pytest.importorskip("torch")


def test_five_rounds_reduce_loss_across_two_agents(fedavg_cluster):
    """fedavg_cluster fixture: coordinator + 2 agents, torn down after."""
    result = fedavg_cluster.run(rounds=5, num_shards=2, min_participants=2)

    losses = [h["mean_loss"] for h in result["history"]]
    assert len(losses) == 5
    assert losses[-1] < losses[0], f"loss did not decrease: {losses}"
    assert all(h["participants"] == 2 for h in result["history"])


def test_round_completes_on_quorum_when_one_agent_dies(fedavg_cluster):
    """The volunteer-network case: a machine leaves mid-run."""
    result = fedavg_cluster.run(rounds=3, num_shards=2, min_participants=1,
                                kill_agent_after_round=1)

    assert len(result["history"]) == 3
    assert result["history"][-1]["participants"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/integration/test_fedavg_e2e.py -m integration -v`
Expected: FAIL — `fixture 'fedavg_cluster' not found`

- [ ] **Step 3: Write the fixture, demo script, and guide**

Build `fedavg_cluster` in `tests/integration/conftest.py` following the existing
lease-test pattern. Three environment rules that have already cost debugging
time and will silently break this if ignored:

1. **Set a neutral `cwd` for every subprocess.** Launching from the workspace
   root makes `flashruntime/` resolve as a namespace package and fail
   strangely (`HANDOFF.md` §3). `e2e/run_demo.py` shows the pattern.
2. **Do not use pytest `tmp_path` for agent workdirs** — under colima it
   bind-mounts as an EMPTY directory. Use `FLASHNODE_WORKDIR` under `$HOME`
   (2026-07-29 gotcha (b)).
3. **Run subprocesses with `python -u`** — uvicorn and pipes buffer stdout, so
   the demo appears to hang (`HANDOFF.md` §3).

`scripts/fedavg_local_demo.py` should print a line per round:

```
round 0  participants 2/2  mean_loss 0.7031
round 1  participants 2/2  mean_loss 0.6402
...
converged: 0.7031 -> 0.4118 over 5 rounds
```

Write `docs/site/guides/federated-averaging.md` covering: why rounds and not
DDP (the `--network none` rendezvous constraint), the quorum rule, the
`flashml.yaml` shape, and an explicit statement that this proves collaborative
training rather than faster training (spec §10).

- [ ] **Step 4: Run the demo and the integration test**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -u scripts/fedavg_local_demo.py`
Expected: 5 rounds print, final loss below the first

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/integration/test_fedavg_e2e.py -m integration -v`
Expected: 2 passed

- [ ] **Step 5: Run the full suite and docs check**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q && .venv/bin/python scripts/build_docs.py --check && ./scripts/audit_secrets.sh`
Expected: ≥319 pre-existing passed plus the ~40 added by this plan; docs check OK; secrets CLEAN

- [ ] **Step 6: Commit**

```bash
git add scripts/fedavg_local_demo.py tests/integration/test_fedavg_e2e.py \
        tests/integration/conftest.py docs/site/guides/federated-averaging.md
git commit -m "feat(fedavg): two-agent convergence demo and integration test

Proves the premise across real process boundaries: 5 rounds across 2 agents
reduce loss, and a round still completes on quorum when one agent dies
mid-run. Guide states plainly that this proves collaborative training, not
faster training."
```

---

### Task 7: Log the slice in PROGRESS.md

**Files:**
- Modify: `../PROGRESS.md` (workspace root)

Required by the workspace logging protocol: "Evidence or it didn't happen."

- [ ] **Step 1: Add the entry**

Newest-first under `## Entries`, following the template. It must carry: real
test counts per suite, the demo command run and its actual output (the loss
numbers), and — per rule 4 — the reasoning behind the quorum decision so nobody
"harmonizes" it back to kmeans_driver's all-shards rule later.

State honestly what is **not** proven: no real Docker daemon run, no
cross-internet run, no heterogeneous hardware. Those arrive in Plans 6 and 7.

- [ ] **Step 2: Commit**

```bash
cd .. && git add PROGRESS.md
git commit -m "docs(progress): federated-averaging driver (fedavg slice, Plan 1 of M1)"
```

---

## Self-Review

**Spec coverage.** §5.4.1 (why rounds not DDP) → Task 6 guide. §5.4.2 (reuse the
driver pattern) → Tasks 3–4. §5.4.3 (round quorum) → Task 4. §5.4.4 (data
placement) → Task 2 `_make_shard`; the baked-in-image variant belongs to Plan 7,
which builds the curated images. §5.4.5 (driver runs in flashml-api) → Task 5's
`Coordinator` protocol and header injection, which is the seam Plan 3 consumes.
§5.4 repo boundary (driver lives in flashruntime) → the whole plan is in
flashruntime. §7 error rows for round quorum → Tasks 4–5. §8 FedAvg test rows →
Tasks 1, 4, 5.

**Deliberately not covered here:** the `flashml.yaml` → `federated_averaging`
compilation (Plan 4), and the MNIST-vs-CIFAR baked dataset (Plan 7, open
question §11.3). Task 2 uses synthetic data so this plan has no dataset
dependency at all.

**Type consistency.** `reduce_deltas(list[tuple[dict, int]])` is called in Task 4
as `reduce_deltas([(d, n) for d, n, _ in collected])` — matches. `apply_delta(base,
delta, scale=1.0)` called as `apply_delta(weights, reduced)` — matches the
default. `state_to_blob`/`blob_to_state` used consistently in Task 2. `Coordinator`
protocol methods (`submit`, `job_state`, `artifacts`, `get_artifact`,
`put_artifact`) match `FakeCoordinator` and `HttpCoordinator` exactly.
`metrics["delta_file"]` written in Task 2, read in Task 4's `_collect`.

**Known rough edge:** `resume_state` in Task 5 has a redundant no-op loop over
`coord.artifacts(job_id)` and a nested scan that is more complex than the
problem needs. Its tests pin the behavior; simplify it during implementation if
the tests stay green.
