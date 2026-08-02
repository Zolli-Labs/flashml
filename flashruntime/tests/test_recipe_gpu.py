"""`resources.gpuPerTask` must survive the JobSpec -> task payload hop.

This is the same hop that broke `local_inputs`, and the exposure is worse.
`CommandRecipe.expand` builds each payload from a fixed key list read out of
`spec.spec.workload.parameters`. `gpuPerTask` does not live there — it lives on
`spec.spec.resources`, a branch of the spec `expand` does not currently read at
all. There is no "unrecognised key" to notice going missing.

The failure mode is the dangerous direction, exactly as before. The gate sees no
`gpus` requirement in the payload, so it does not fail closed and refuse the
CPU-only node — it fails OPEN, and the job lands anywhere. Unit tests at either
end pass throughout: `test_protocol_gpu.py` asserts the field exists on the
spec, `test_placement_gpu.py` builds a payload containing `gpus` by hand. Each
end is correct about itself. This file is about the hop between them.
"""

from __future__ import annotations

from flashruntime.protocol.v1alpha1 import JobSpec
from flashruntime.recipes.command import CommandRecipe


def _spec(resources=None, **params) -> JobSpec:
    return JobSpec.model_validate({
        "metadata": {"name": "gpu-job"},
        "spec": {
            "image": {"repository": "ghcr.io/zolli-labs/flashml-pytorch-cuda",
                      "tag": "2026.08.1"},
            "workload": {
                "type": "command",
                "parameters": {"command": ["python", "train.py"], **params},
            },
            "resources": {"minimumWorkers": 1, "maximumWorkers": 1,
                          **(resources or {})},
            "isolation": {"tier": "sandboxed", "allowFallback": False},
        },
    })


def test_gpu_per_task_reaches_the_task_payload():
    """Where the placement gate actually reads it."""
    tasks = CommandRecipe().expand("job-1", _spec({"gpuPerTask": 2}))

    assert tasks, "expected at least one task"
    for task in tasks:
        assert task.payload.get("gpus") == 2, (
            "the placement gate reads task.payload['gpus']; without this "
            "forward the gate fails OPEN and GPU work is placed on any node"
        )


def test_a_single_gpu_request_reaches_the_task_payload():
    tasks = CommandRecipe().expand("job-1", _spec({"gpuPerTask": 1}))
    assert tasks[0].payload["gpus"] == 1


def test_the_default_leaves_gpus_absent_from_the_payload():
    """Absent, never 0 — the same convention as `unpack_inputs` and
    `local_inputs` beside it.

    An omitted key is the branch every job that exists today takes, and the
    one the gate's "no requirement" path is written against. Emitting `0`
    would mean the same thing to the gate today while quietly ceasing to
    exercise that branch."""
    for task in CommandRecipe().expand("job-1", _spec()):
        assert "gpus" not in task.payload


def test_an_explicit_zero_also_leaves_gpus_absent():
    for task in CommandRecipe().expand("job-1", _spec({"gpuPerTask": 0})):
        assert "gpus" not in task.payload


def test_every_task_of_a_sweep_carries_the_requirement():
    """`gpuPerTask` is per task, so a multi-task sweep must stamp all of
    them — one unstamped task is one task placed on a CPU-only host."""
    spec = _spec(
        {"gpuPerTask": 1},
        task_params=[{"lr": 0.1}, {"lr": 0.2}, {"lr": 0.3}],
        command=["python", "train.py", "--lr", "{lr}"],
    )
    tasks = CommandRecipe().expand("job-1", spec)

    assert len(tasks) == 3
    assert [t.payload["gpus"] for t in tasks] == [1, 1, 1]


def test_the_gpu_forward_does_not_disturb_the_existing_payload_keys():
    """The no-GPU payload must stay byte-for-byte what it is today: the whole
    reason absent stays absent."""
    plain = CommandRecipe().expand("job-1", _spec())[0].payload
    with_gpu = CommandRecipe().expand("job-1", _spec({"gpuPerTask": 1}))[0].payload

    assert set(with_gpu) - set(plain) == {"gpus"}
    for key, value in plain.items():
        assert with_gpu[key] == value


def test_the_forwarded_requirement_passes_the_placement_gate():
    """The two hops joined: what `expand` writes is what the gate reads.

    Neither side is asserted against a hand-built intermediate here — that is
    the whole point.
    """
    from flashruntime.scheduler import IsolationAwarePlacement

    task = CommandRecipe().expand("job-1", _spec({"gpuPerTask": 1}))[0]
    policy = IsolationAwarePlacement()
    base = {"sandbox_capable": True, "argv_capable": True}

    cpu_node = {"node_id": "cpu", **base, "capabilities": {"gpus": []}}
    gpu_node = {"node_id": "gpu", **base, "capabilities": {"gpus": [{"index": 0}]}}

    assert policy.eligible(task, cpu_node) is False
    assert policy.eligible(task, gpu_node) is True


def test_a_job_with_no_gpu_requirement_is_placed_on_both():
    from flashruntime.scheduler import IsolationAwarePlacement

    task = CommandRecipe().expand("job-1", _spec())[0]
    policy = IsolationAwarePlacement()
    base = {"sandbox_capable": True, "argv_capable": True}

    cpu_node = {"node_id": "cpu", **base, "capabilities": {"gpus": []}}
    gpu_node = {"node_id": "gpu", **base, "capabilities": {"gpus": [{"index": 0}]}}

    assert policy.eligible(task, cpu_node) is True
    assert policy.eligible(task, gpu_node) is True
