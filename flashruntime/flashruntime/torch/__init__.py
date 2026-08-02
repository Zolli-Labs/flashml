"""In-training-script helper: one import makes a PyTorch script
launch-anywhere and fault-tolerant.

    import flashruntime.torch as ft
    model, opt, loader = ft.prepare(model, opt, loader)
    ...
    ft.checkpoint(model, opt, step=step, every=100)
    ft.log_metrics({"loss": float(loss)})

Launched by torchrun (WORLD_SIZE>1): prepare() wires torch's OWN DDP +
DistributedSampler and restores the newest VALID checkpoint manifest.
Launched as plain `python train.py`: prepare() is a no-op passthrough.

GUARDRAIL (ADR-0003 — do not rebuild Accelerate): this module wires
torch's own primitives and REPORTS launch facts — nothing more (that
reporting includes a per-rank heartbeat file the run viewer reads). The
surface is prepare / checkpoint / log_metrics plus read-only accessors
(rank / world_size / is_main / start_step / device / backend). The
boundary is capability, not count: no FSDP policies, no autocast, no
DeepSpeed config, no strategy knobs — ever. Users wanting those use the
real framework features, which the launcher still launches correctly.

torch is imported inside functions only: flashruntime's core never
depends on it.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

__all__ = [
    "prepare",
    "checkpoint",
    "log_metrics",
    "rank",
    "world_size",
    "is_main",
    "start_step",
    "device",
    "backend",
]

_restored_step = 0
# Set by prepare() from launch facts so a script can REPORT what it ran on
# (device string + torch.distributed backend) without re-deriving it. Both
# stay at their launched-single-process defaults until prepare() runs.
_device = "cpu"
_backend: str | None = None

# Heartbeat state: the run viewer draws machine → worker → rank from a small
# per-rank JSON each process mirrors to ranks/rank-N.json. Throttled so a
# tight training loop calling log_metrics()/checkpoint() every step costs at
# most one small atomic write per second per rank.
_last_beat = 0.0
_last_step: int | None = None
_BEAT_MIN_INTERVAL_S = 1.0


def _write_heartbeat(step: int | None = None, force: bool = False) -> None:
    """Mirror this rank's identity + progress to ranks/rank-<N>.json.

    Atomic (tmp + os.replace, the run.json idiom) so the viewer never reads
    a torn file. Best-effort by contract: observability must never be able
    to crash training, so every failure is swallowed. `step=None` reuses the
    last step this process reported (a refresh must not erase progress)."""
    global _last_beat, _last_step
    if step is not None:
        _last_step = step
    now = time.time()
    if not force and now - _last_beat < _BEAT_MIN_INTERVAL_S:
        return
    try:
        beat = {
            "rank": rank(),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "pid": os.getpid(),
            "device": _device,
            "backend": _backend,
            "world_size": world_size(),
            "step": _last_step,
            "ts": now,
        }
        ranks_dir = _output_dir() / "ranks"
        ranks_dir.mkdir(parents=True, exist_ok=True)
        tmp = ranks_dir / f".rank-{rank()}.{os.getpid()}.tmp"
        tmp.write_text(json.dumps(beat))
        os.replace(tmp, ranks_dir / f"rank-{rank()}.json")
        _last_beat = now
    except Exception:  # noqa: BLE001 — by contract, swallow everything
        pass


def _resolve_device(world_size: int, cuda_available: bool, local_rank: int) -> str:
    """Decide the training device from launch facts alone — returns ``"cpu"``
    or ``f"cuda:{local_rank}"``.

    Kept pure (no torch import, no globals) so the device *decision* is
    testable on this CPU-only box: the CUDA code paths in ``prepare`` that
    consume the result cannot execute here, but the choice they depend on
    can. ``world_size`` is part of the launch-facts signature; the device
    string itself depends only on whether CUDA is present and which local
    rank this process owns.
    """
    if cuda_available:
        return f"cuda:{local_rank}"
    return "cpu"


def world_size() -> int:
    return int(os.environ.get("WORLD_SIZE", "1"))


def rank() -> int:
    return int(os.environ.get("RANK", "0"))


def is_main() -> bool:
    return rank() == 0


def start_step() -> int:
    """First step the training loop should run: 0 fresh, >0 after a
    resume (set by prepare() when it restores a checkpoint)."""
    return _restored_step


def device() -> str:
    """The device the last ``prepare()`` placed this rank's model on —
    ``"cpu"`` or ``f"cuda:{local_rank}"``. Lets a script report where it
    trained (e.g. into ``metrics.json``) without re-deriving the choice."""
    return _device


def backend() -> str | None:
    """The ``torch.distributed`` backend the last ``prepare()`` initialized
    (``"nccl"`` on GPU, ``"gloo"`` on CPU), or ``None`` when launched
    single-process (no process group)."""
    return _backend


def _output_dir() -> Path:
    return Path(os.environ.get("FLASHML_OUTPUT_DIR", "."))


def _ckpt_root() -> Path:
    # job-scoped (NOT attempt-scoped): a restarted attempt must find its
    # predecessor's manifests — the launcher exports FLASHML_CKPT_DIR
    root = os.environ.get("FLASHML_CKPT_DIR")
    return Path(root) if root else _output_dir() / "ckpt"


def prepare(model, optimizer=None, dataloader=None):
    """Wire distributed execution (when launched distributed) and restore
    the newest valid checkpoint (when one exists). Returns the possibly
    wrapped/rebuilt (model, optimizer, dataloader) triple."""
    global _restored_step, _device, _backend
    import torch

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = _resolve_device(world_size(), torch.cuda.is_available(), local_rank)
    on_cuda = device != "cpu"
    _device = device  # remember for device()/metrics reporting

    if on_cuda:
        # place the model on this rank's GPU BEFORE the DDP wrap (and for a
        # single-process GPU box too — the "just works on a GPU" path). This
        # branch is CUDA-only and never runs on the CPU-only test machine.
        torch.cuda.set_device(local_rank)
        model = model.to(device)

    if world_size() > 1:
        import torch.distributed as dist

        _backend = "nccl" if torch.cuda.is_available() else "gloo"
        if not dist.is_initialized():
            dist.init_process_group(backend=_backend)
        if on_cuda:
            # bind DDP to the device; gloo/CPU must keep the no-args wrap.
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank], output_device=local_rank
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(model)
        if dataloader is not None:
            from torch.utils.data import DataLoader
            from torch.utils.data.distributed import DistributedSampler

            dataloader = DataLoader(
                dataloader.dataset,
                batch_size=dataloader.batch_size,
                sampler=DistributedSampler(dataloader.dataset),
                collate_fn=dataloader.collate_fn,
                num_workers=dataloader.num_workers,
                drop_last=dataloader.drop_last,
            )

    from flashruntime.checkpoint.local import latest_valid_manifest

    manifest = latest_valid_manifest(_ckpt_root())
    if manifest is not None:
        step_dir = Path(manifest.storage_prefix)
        target = model.module if hasattr(model, "module") else model
        # load straight onto the live device: CPU-saved (device-agnostic)
        # parts map onto "cpu" (unchanged) or this rank's "cuda:N".
        target.load_state_dict(torch.load(step_dir / "model.pt", map_location=device))
        if optimizer is not None and (step_dir / "optimizer.pt").is_file():
            optimizer.load_state_dict(torch.load(step_dir / "optimizer.pt", map_location=device))
        _restored_step = manifest.step

    # announce this rank to the run viewer the moment it is wired
    _write_heartbeat(step=_restored_step, force=True)
    return model, optimizer, dataloader


def checkpoint(model, optimizer=None, *, step: int, every: int | None = None) -> None:
    """Write a resumable checkpoint under the manifest contract (parts
    first, manifest last). rank 0 writes; every rank synchronizes on the
    barrier so no one races past a half-written checkpoint.

    `every=None` (default) checkpoints unconditionally; `every=N` gates to
    every Nth step; `every<=0` means "no periodic checkpointing" and is a
    no-op — a helper whose contract is fault tolerance must never itself
    crash training over a config value (found by the benchmark suite:
    `every=0` used to raise ZeroDivisionError)."""
    _write_heartbeat(step=step)  # progress signal — fires even on gated calls
    if every is not None and (every <= 0 or step == 0 or step % every != 0):
        return
    import torch

    if is_main():
        from flashruntime.checkpoint.local import write_manifest

        step_dir = _ckpt_root() / f"step-{step:06d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        target = model.module if hasattr(model, "module") else model
        # detach + move to CPU so the saved parts are device-agnostic: a
        # checkpoint trained on cuda:3 must restore onto cpu or a box with
        # fewer GPUs (map_location handles the rest at restore time).
        model_state = {k: v.detach().cpu() for k, v in target.state_dict().items()}
        torch.save(model_state, step_dir / "model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
        write_manifest(
            step_dir,
            job_id=os.environ.get("FLASHML_JOB_ID", "local"),
            attempt_id=os.environ.get("FLASHML_ATTEMPT_ID", "local"),
            step=step,
            world_size=world_size(),
            framework=f"pytorch-{torch.__version__.split('+')[0]}",
        )
    if world_size() > 1:
        import torch.distributed as dist

        dist.barrier()


def log_metrics(metrics: dict) -> None:
    """Append one JSON record to FLASHML_OUTPUT_DIR/metrics.jsonl (rank 0
    only). Never raises — metrics must never kill training."""
    # every rank beats (identity/progress); only rank 0 appends metrics below
    step = metrics.get("step") if isinstance(metrics, dict) else None
    _write_heartbeat(step=step if isinstance(step, int) else None)
    if not is_main():
        return
    try:
        path = _output_dir() / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(metrics) + "\n")
    except Exception:  # noqa: BLE001 — by contract, swallow everything
        pass
