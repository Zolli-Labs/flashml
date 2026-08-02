"""End-to-end demo: FlashRuntime operating YOUR code.

    python examples/bring_your_code_demo.py

Runs (1) an sklearn hyperparameter sweep, and — when torch+torchrun are
installed — (2) a 2-process CPU DDP training run, then (3) the
kill-and-resume story: crash mid-training, resubmit, watch it resume from
the last valid checkpoint manifest.
"""
import shutil
import sys
import tempfile
from pathlib import Path

import flashruntime as flash
from flashruntime.integrations import pytorch as fr_torch
from flashruntime.integrations import sklearn as fr_sklearn

EXAMPLES = Path(__file__).parent


def sklearn_sweep() -> None:
    print("=== 1. sklearn hyperparameter sweep (Mode A shape, local) ===")
    run = flash.submit(
        fr_sklearn.hpo(
            "train.py",
            {"model": ["logreg", "rf"], "C": [0.1, 1.0], "n_estimators": [50]},
            source=str(EXAMPLES / "user_sklearn"),
        )
    )
    print(f"state={run.state.value}  trials={len(run.trials)}")
    print("best:", run.best_trial())


def pytorch_ddp() -> None:
    print("\n=== 2. PyTorch DDP, 2 processes on CPU (gloo) ===")
    run = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args="--steps 120 --checkpoint-every 40",
        )
    )
    print(f"state={run.state.value}  metrics={run.trials}")

    print("\n=== 3. kill at step 60, then resume from the last valid checkpoint ===")
    workdir = Path(tempfile.mkdtemp(prefix="flashruntime-demo-"))
    crash = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args="--steps 120 --checkpoint-every 40 --kill-at-step 60",
        ),
        output_dir=workdir,
    )
    print(f"crashed run: state={crash.state.value} (expected FAILED)")
    resume = flash.submit(
        fr_torch.ddp(
            "train.py",
            source=str(EXAMPLES / "user_pytorch"),
            nproc_per_node=2,
            script_args="--steps 120 --checkpoint-every 40 --kill-at-step 60",
        ),
        output_dir=workdir,  # same dir ⇒ same job ⇒ same checkpoint tree
    )
    print(f"resumed run: state={resume.state.value}  metrics={resume.trials}")
    if resume.trials:
        print(f"resumed_from step {resume.trials[0].get('resumed_from')} — recovery, not a restart")


def main() -> None:
    sklearn_sweep()
    try:
        import torch  # noqa: F401
    except ImportError:
        print("\n(torch not installed — skipping the PyTorch demos: pip install torch)")
        return
    if shutil.which("torchrun") is None:
        print("\n(torchrun not on PATH — skipping the DDP demos)")
        return
    pytorch_ddp()


if __name__ == "__main__":
    sys.exit(main())
