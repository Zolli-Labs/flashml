"""Adapters build workloads; they never import their frameworks at module
level (four-axes rule: framework code runs in the user's process only)."""

from __future__ import annotations

import sys

import pytest


def test_importing_adapters_pulls_no_frameworks():
    preloaded = {m for m in ("torch", "sklearn", "transformers") if m in sys.modules}
    import flashruntime.integrations.huggingface  # noqa: F401
    import flashruntime.integrations.pytorch  # noqa: F401
    import flashruntime.integrations.sklearn  # noqa: F401

    for mod in ("torch", "sklearn", "transformers"):
        if mod not in preloaded:
            assert mod not in sys.modules, f"adapter import pulled in {mod}"


def test_sklearn_hpo_builds_cartesian_fanout():
    from flashruntime.integrations import sklearn as fr_sklearn

    wl = fr_sklearn.hpo("train.py", {"model": ["logreg", "rf"], "C": [0.1, 1.0]}, source="/proj")
    assert wl.resolved_mode() == "independent_tasks"
    assert len(wl.task_params) == 4
    # flags for every grid key, sorted, placeholder-form
    assert wl.command == ["python", "train.py", "--C", "{C}", "--model", "{model}"]
    assert wl.outputs.primary_metric == "accuracy_mean"
    assert wl.source.path == "/proj"


def test_pytorch_ddp_builds_torchrun_command():
    from flashruntime.integrations import pytorch as fr_torch

    wl = fr_torch.ddp("train.py", source="/proj", nproc_per_node=4, script_args="--steps 100")
    assert wl.command[:5] == [
        "torchrun",
        "--nproc-per-node=4",
        "--nnodes=1",
        "--standalone",
        # Single-node by definition, so pin the rendezvous address: torchrun
        # otherwise advertises socket.getfqdn(), which macOS can resolve to an
        # unresolvable ip6.arpa name — workers then retry DNS forever.
        "--local-addr=127.0.0.1",
    ]
    assert wl.command[5:] == ["train.py", "--steps", "100"]
    assert wl.resolved_mode() == "coordinated"
    assert wl.checkpoint is not None


def test_pytorch_multinode_not_yet():
    from flashruntime.integrations import pytorch as fr_torch

    with pytest.raises(NotImplementedError, match="multi-node"):
        fr_torch.ddp("train.py", nnodes=2)


def test_hf_trainer_is_torchrun_shaped():
    from flashruntime.integrations import huggingface as fr_hf

    wl = fr_hf.trainer("finetune.py", nproc_per_node=2)
    assert wl.command[0] == "torchrun"
    assert wl.resolved_mode() == "coordinated"


def test_hf_latest_checkpoint_reads_manifests(tmp_path):
    from flashruntime.checkpoint.local import write_manifest
    from flashruntime.integrations import huggingface as fr_hf

    d = tmp_path / "checkpoint-40"
    d.mkdir()
    (d / "model.safetensors").write_bytes(b"w")
    write_manifest(d, job_id="j", attempt_id="a", step=40)
    assert fr_hf.latest_checkpoint(tmp_path) == str(d)
    assert fr_hf.latest_checkpoint(tmp_path / "none") is None
