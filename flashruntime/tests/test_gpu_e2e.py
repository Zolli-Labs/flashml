"""Real-GPU acceptance test (Task 12) — the CUDA-path evidence the CPU suite
structurally cannot produce.

`tests/test_examples_e2e.py` proves DDP end-to-end, but only over gloo on the
CPU box every developer has; `flashruntime.torch._resolve_device` and the nccl
branch of `prepare()` are unit-tested via a *pure* helper because no GPU is
present. This module closes that gap on real hardware: it is skipped wholesale
unless CUDA is present with >=2 visible GPUs, and is driven on a rented 2-GPU
RunPod box by `scripts/runpod_gpu_e2e.py`.

It asserts, on the actual GPUs, exactly the three things the CPU suite can only
approximate:
  1. a 2-process **nccl** DDP run of `examples/user_pytorch` completes and
     reports `backend == "nccl"` (the metrics key added for this);
  2. the model trained on **cuda** — rank 0 reports `device == "cuda:0"`
     (`_resolve_device` exercised for real, model params placed on the GPU);
  3. **GPU kill-and-resume** via `max_restarts=1` resumes from the step-40
     checkpoint (`resumed_from == 40`) and lands on the uninterrupted loss —
     recovery does not change the math, even across a CUDA<->CPU checkpoint
     round-trip.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

if not torch.cuda.is_available():  # the whole point is real CUDA
    pytest.skip("CUDA not available — GPU e2e runs only on a GPU box", allow_module_level=True)

EXAMPLES = Path(__file__).parent.parent / "examples"


@pytest.fixture(scope="module", autouse=True)
def _require_two_gpus():
    """2-process nccl DDP needs one visible GPU per rank; skip (do not fail)
    on a smaller box so a single-GPU run of this file is honest, not red."""
    if shutil.which("torchrun") is None:
        pytest.skip("torchrun not on PATH")
    count = torch.cuda.device_count()
    if count < 2:
        pytest.skip(f"need >=2 GPUs for 2-proc nccl DDP; found {count}")


def _ddp(extra: str = ""):
    from flashruntime.integrations import pytorch as fr_torch

    # steps/checkpoint/kill all on the 8-batch/rank/epoch boundary (512/32/2)
    # so a resumed run replays identical batches — see train.py docstring.
    args = f"--steps 80 --checkpoint-every 20 {extra}".strip()
    return fr_torch.ddp(
        "train.py",
        source=str(EXAMPLES / "user_pytorch"),
        nproc_per_node=2,
        script_args=args,
    )


def test_two_proc_nccl_ddp_completes_on_gpu(tmp_path):
    import flashruntime as flash

    run = flash.submit(_ddp(), output_dir=tmp_path)
    assert run.state.value == "SUCCEEDED", run.logs()
    metrics = run.trials[0]
    # the CUDA claims the CPU suite cannot make:
    assert metrics["backend"] == "nccl", metrics  # torch.distributed over nccl
    assert metrics["device"] == "cuda:0", metrics  # rank 0 model on GPU 0


def test_gpu_kill_and_resume_reproduces_uninterrupted_result(tmp_path):
    import flashruntime as flash

    baseline = flash.submit(_ddp(), output_dir=tmp_path / "baseline")
    assert baseline.state.value == "SUCCEEDED", baseline.logs()

    resumed = flash.submit(
        _ddp("--kill-at-step 40"), output_dir=tmp_path / "crashy", max_restarts=1
    )
    assert resumed.state.value == "SUCCEEDED", resumed.logs()
    assert resumed.trials[0]["resumed_from"] == 40, resumed.trials[0]
    assert resumed.trials[0]["device"] == "cuda:0", resumed.trials[0]
    # Recovery must not change the math. Tolerance is looser than the CPU
    # suite's 1e-6 only because GPU matmul reductions carry more FP noise than
    # CPU's; the checkpoint round-trip itself (CUDA->CPU save, CPU->CUDA load)
    # is bit-exact for float32, so this stays a tight, meaningful bound.
    assert resumed.trials[0]["final_loss"] == pytest.approx(
        baseline.trials[0]["final_loss"], abs=1e-3
    )
    types = [e["type"] for e in resumed.events]
    assert "FAILURE_CLASSIFIED" in types and "RECOVERY_ACTION_SELECTED" in types
