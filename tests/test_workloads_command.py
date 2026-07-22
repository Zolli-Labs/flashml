"""CommandWorkload: the user-facing 'bring your own code' description."""

from __future__ import annotations

import pytest


def _wl(**over):
    from flashruntime.workloads.command import CommandWorkload

    base = dict(command="python train.py")
    base.update(over)
    return CommandWorkload(**base)


def test_argv_from_string_and_list():
    assert _wl(command="python train.py --lr 0.1").argv() == [
        "python", "train.py", "--lr", "0.1"
    ]
    assert _wl(command=["python", "train.py"]).argv() == ["python", "train.py"]


def test_argv_substitutes_task_params():
    wl = _wl(command="python train.py --lr {lr}")
    assert wl.argv({"lr": 0.01}) == ["python", "train.py", "--lr", "0.01"]


def test_argv_missing_placeholder_raises():
    with pytest.raises(KeyError):
        _wl(command="python train.py --lr {lr}").argv({"seed": 1})


def test_inputs_must_be_artifact_uris():
    with pytest.raises(ValueError, match="artifact://"):
        _wl(inputs={"dataset": "/tmp/data.csv"})
    wl = _wl(inputs={"dataset": "artifact://jobs/x/data.csv"})
    assert wl.inputs["dataset"].startswith("artifact://")


def test_resolved_mode_auto_rules():
    assert _wl().resolved_mode() == "local"
    assert _wl(task_params=[{"lr": 0.1}]).resolved_mode() == "independent_tasks"
    assert _wl(command="torchrun --nproc-per-node=2 train.py").resolved_mode() == "coordinated"
    # explicit pin always wins
    assert _wl(task_params=[{"lr": 0.1}], mode="local").resolved_mode() == "local"


def test_to_jobspec_requires_image_and_carries_isolation():
    from flashruntime.protocol.v1alpha1 import ImageSpec, IsolationSpec
    from flashruntime.workloads.command import to_jobspec

    wl = _wl(
        command="python train.py --lr {lr}",
        task_params=[{"lr": 0.1}, {"lr": 0.01}],
        isolation=IsolationSpec(tier="sandboxed"),
    )
    with pytest.raises(ValueError, match="image"):
        to_jobspec(wl, name="sweep")

    spec = to_jobspec(wl, name="sweep", image=ImageSpec(repository="ghcr.io/me/x", tag="1.0"))
    assert spec.spec.execution.backend == "leases"
    assert spec.spec.workload.type == "command"
    assert spec.spec.workload.parameters["command"] == ["python", "train.py", "--lr", "{lr}"]
    assert spec.spec.workload.parameters["task_params"] == [{"lr": 0.1}, {"lr": 0.01}]
    assert spec.spec.isolation.tier == "sandboxed"
