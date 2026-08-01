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


def test_rejects_weights_with_wrong_shapes(tmp_path):
    from flashml_workloads.fedavg_weights import WeightShapeMismatch

    wpath = tmp_path / "weights.json"
    wpath.write_text(json.dumps({"nonsense": {"shape": [1], "data": [0.0]}}))
    out = tmp_path / "out"
    out.mkdir()
    with pytest.raises(WeightShapeMismatch):
        fedavg_worker.run_worker(_spec(tmp_path, weights_path=wpath, round=1), out)
