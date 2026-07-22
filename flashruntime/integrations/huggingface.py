"""Hugging Face adapter: HF Trainer already wraps DDP/FSDP internally when
launched by torchrun, so launching is the pytorch adapter's job. What HF
adds is its callback seam — `flashruntime_callback()` commits Trainer
checkpoints as verified manifests. transformers is imported only inside
that factory, in the user's training process.
"""

from __future__ import annotations

import os
from pathlib import Path

from flashruntime.checkpoint.local import latest_valid_manifest
from flashruntime.integrations.pytorch import ddp
from flashruntime.workloads.command import CommandWorkload


def trainer(
    script: str, *, source: str = ".", nproc_per_node: int = 1, script_args: str = ""
) -> CommandWorkload:
    return ddp(script, source=source, nproc_per_node=nproc_per_node, script_args=script_args)


def latest_checkpoint(output_dir: str | Path) -> str | None:
    """Newest Trainer checkpoint dir with a VALID manifest — pass as
    `trainer.train(resume_from_checkpoint=...)`. None means fresh start."""
    manifest = latest_valid_manifest(Path(output_dir), pattern="checkpoint-*")
    return None if manifest is None else manifest.storage_prefix


def flashruntime_callback():
    """Build the TrainerCallback (transformers import paid here, in the
    user's process only): on_save commits a manifest, on_log relays metrics."""
    from transformers import TrainerCallback  # noqa: PLC0415 — user process only

    from flashruntime.checkpoint.local import write_manifest

    class FlashRuntimeCallback(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                step_dir = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                if step_dir.is_dir():
                    write_manifest(
                        step_dir,
                        job_id=os.environ.get("FLASHML_JOB_ID", "local"),
                        attempt_id=os.environ.get("FLASHML_ATTEMPT_ID", "local"),
                        step=state.global_step,
                        framework="transformers",
                    )
            return control

        def on_log(self, args, state, control, logs=None, **kwargs):
            if logs and state.is_world_process_zero:
                from flashruntime.torch import log_metrics

                log_metrics({**logs, "step": state.global_step})
            return control

    return FlashRuntimeCallback()
