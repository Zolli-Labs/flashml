"""Checkpointable trainer — written before the implementation (TDD).

The Stage-7 contract in its purest form: a training task that writes
periodic checkpoints, dies mid-run only on a *fresh* start (so retries are
deterministic), and — the load-bearing property — **resuming from a
checkpoint reproduces the uninterrupted run exactly**. If resume weren't
bit-identical, recovery would silently change results, and fault tolerance
that returns different answers is corruption with extra steps.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from flashml_workloads.sgd_trainer import run_trainer

DATASET = "\n".join(
    f"{x:.1f},{y:.1f},{1 if x + y > 4 else 0}"
    for x, y in [(1, 1), (2, 1), (3, 3), (4, 2), (0, 1), (5, 4), (2, 4), (1, 0)]
)


@pytest.fixture()
def dataset(tmp_path):
    p = tmp_path / "train.csv"
    p.write_text(DATASET + "\n")
    return p


def _spec(dataset, out, *, steps=10, every=4, kill_at=None, resume=None):
    params = {"steps": steps, "checkpoint_every": every, "lr": 0.1, "seed": 7}
    if kill_at is not None:
        params["kill_at_step"] = kill_at
    inputs = {"dataset": str(dataset)}
    if resume is not None:
        inputs["resume"] = str(resume)
    return {"task_id": "t1", "params": params, "inputs": inputs}, Path(out)


def test_uninterrupted_run_writes_periodic_checkpoints_and_metrics(dataset, tmp_path):
    spec, out = _spec(dataset, tmp_path / "out")
    metrics = run_trainer(spec, out)
    ckpts = sorted(p.name for p in (out / "ckpt").iterdir())
    assert ckpts == ["step-000004.json", "step-000008.json"]
    assert metrics["steps"] == 10
    assert metrics["started_from_step"] == 0
    assert metrics["resumed"] is False
    assert 0.0 < metrics["final_loss"] < 1.0


def test_fresh_run_dies_at_kill_step_leaving_checkpoints(dataset, tmp_path):
    spec, out = _spec(dataset, tmp_path / "out", kill_at=6)
    with pytest.raises(SystemExit):
        run_trainer(spec, out)
    assert (out / "ckpt" / "step-000004.json").is_file()
    assert not (out / "metrics.json").exists()  # died before finishing


def test_resume_reproduces_the_uninterrupted_run_exactly(dataset, tmp_path):
    # baseline: 10 uninterrupted steps
    spec_a, out_a = _spec(dataset, tmp_path / "a")
    baseline = run_trainer(spec_a, out_a)

    # crash at step 6 (checkpoint exists at 4), then resume to 10
    spec_b, out_b = _spec(dataset, tmp_path / "b", kill_at=6)
    with pytest.raises(SystemExit):
        run_trainer(spec_b, out_b)
    spec_c, out_c = _spec(
        dataset, tmp_path / "c", kill_at=6, resume=out_b / "ckpt" / "step-000004.json"
    )
    resumed = run_trainer(spec_c, out_c)

    assert resumed["resumed"] is True
    assert resumed["started_from_step"] == 4
    assert resumed["steps"] == 10
    # bit-identical recovery: same weights, same loss as never having died
    assert resumed["weights"] == baseline["weights"]
    assert resumed["final_loss"] == baseline["final_loss"]
    # kill_at_step must NOT fire on a resumed run (else retries loop forever)


def test_checkpoint_file_carries_step_and_weights(dataset, tmp_path):
    spec, out = _spec(dataset, tmp_path / "out")
    run_trainer(spec, out)
    ckpt = json.loads((out / "ckpt" / "step-000008.json").read_text())
    assert ckpt["step"] == 8
    assert isinstance(ckpt["weights"], list) and len(ckpt["weights"]) == 2
    assert "bias" in ckpt
