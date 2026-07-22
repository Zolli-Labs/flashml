"""Acceptance tests (spec §9): real user code, operated end to end.
Auto-skip per missing dependency; they run on any dev laptop with the
extras installed."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).parent.parent / "examples"


def test_sklearn_sweep_end_to_end(tmp_path):
    pytest.importorskip("sklearn")
    import flashruntime as flash
    from flashruntime.integrations import sklearn as fr_sklearn

    run = flash.submit(
        fr_sklearn.hpo(
            "train.py",
            {"model": ["logreg", "rf"], "C": [0.1], "n_estimators": [30]},
            source=str(EXAMPLES / "user_sklearn"),
        ),
        output_dir=tmp_path,
    )
    assert run.state.value == "SUCCEEDED"
    assert len(run.trials) == 2
    best = run.best_trial()
    assert 0.5 < best["accuracy_mean"] <= 1.0


@pytest.mark.parametrize("example", ["user_pytorch", "user_pytorch_vanilla"])
def test_ddp_two_processes_on_cpu(tmp_path, example):
    pytest.importorskip("torch")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    import flashruntime as flash
    from flashruntime.integrations import pytorch as fr_torch

    args = "--steps 60 --checkpoint-every 20" if example == "user_pytorch" else ""
    run = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / example),
            nproc_per_node=2,
            script_args=args,
        ),
        output_dir=tmp_path,
    )
    assert run.state.value == "SUCCEEDED", run.logs()
    assert run.trials, run.logs()


def test_kill_and_resume_reproduces_uninterrupted_result(tmp_path):
    """Spec §9 criterion 3b: crash mid-training, resubmit, resume from the
    newest valid manifest — and land on the same final loss."""
    pytest.importorskip("torch")
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    import flashruntime as flash
    from flashruntime.integrations import pytorch as fr_torch

    def ddp(extra: str = ""):
        return fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args=f"--steps 80 --checkpoint-every 20 {extra}".strip(),
        )

    baseline = flash.submit(ddp(), output_dir=tmp_path / "baseline")
    assert baseline.state.value == "SUCCEEDED", baseline.logs()

    workdir = tmp_path / "crashy"
    crashed = flash.submit(ddp("--kill-at-step 40"), output_dir=workdir)
    assert crashed.state.value == "FAILED"

    resumed = flash.submit(ddp("--kill-at-step 40"), output_dir=workdir)
    assert resumed.state.value == "SUCCEEDED", resumed.logs()
    assert resumed.trials[0]["resumed_from"] == 40
    assert resumed.trials[0]["final_loss"] == pytest.approx(
        baseline.trials[0]["final_loss"], abs=1e-6
    )
