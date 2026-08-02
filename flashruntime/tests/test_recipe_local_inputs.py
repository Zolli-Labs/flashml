"""`local_inputs` must survive the JobSpec -> task payload hop.

The feature is built at both ends of this hop and was broken in the middle.
`flashml.yaml` puts `local_inputs` into the workload parameters, and
`IsolationAwarePlacement` reads it from `task.payload` — but `CommandRecipe`
builds each payload from a fixed key list and dropped every parameter it did
not recognise, `local_inputs` among them.

The failure mode is the dangerous direction. The gate sees no requirement, so
it does not fail closed and refuse the node — it fails OPEN, places the task
anywhere, and flashnode mounts nothing. The task then runs without the data it
asked for.

Nothing caught it because both ends' tests construct the intermediate value
directly: the compile tests assert the parameter lands in the JobSpec, and the
placement tests build a payload containing `local_inputs` by hand. Each end was
correct about itself. This test is about the hop between them.
"""
from __future__ import annotations

from flashruntime.protocol.v1alpha1 import JobSpec
from flashruntime.recipes.command import CommandRecipe


def _spec(**params) -> JobSpec:
    return JobSpec.model_validate({
        "metadata": {"name": "local-data-job"},
        "spec": {
            "image": {"repository": "ghcr.io/zolli-labs/flashml-python-slim",
                      "tag": "2026.08.1"},
            "workload": {
                "type": "command",
                "parameters": {"command": ["python", "train.py"], **params},
            },
            "resources": {"minimumWorkers": 1, "maximumWorkers": 1},
            "isolation": {"tier": "sandboxed", "allowFallback": False},
        },
    })


def test_local_inputs_reaches_the_task_payload():
    """Where the placement gate actually reads it."""
    tasks = CommandRecipe().expand("job-1", _spec(local_inputs=["patients"]))

    assert tasks, "expected at least one task"
    for task in tasks:
        assert task.payload.get("local_inputs") == ["patients"], (
            "the placement gate reads task.payload['local_inputs']; without it "
            "the gate fails OPEN and the task is placed on a node with no data"
        )


def test_absent_local_inputs_stays_absent():
    """Never an empty list — same reasoning as `unpack_inputs` beside it.

    An omitted key is the path where nothing about placement changes, and
    emitting `[]` would mean the same thing today while quietly ceasing to
    exercise that path."""
    for task in CommandRecipe().expand("job-1", _spec()):
        assert "local_inputs" not in task.payload


def test_local_inputs_is_copied_not_aliased():
    """A payload must not share a mutable list with the caller's spec."""
    names = ["patients"]
    tasks = CommandRecipe().expand("job-1", _spec(local_inputs=names))
    names.append("labs")
    assert tasks[0].payload["local_inputs"] == ["patients"]
