"""`GpuInfo` and `ResourcesSpec.gpuPerTask` — the two wire additions GPU
placement needs.

These are protocol tests, not scheduler tests: they pin the *wire* behaviour
the fifth placement gate is later allowed to trust, exactly as
`test_protocol_local_datasets.py` does for `local_datasets`.

The typing change matters more than it looks. `NodeCapabilities.gpus` was
`list[dict[str, Any]]`; it becomes `list[GpuInfo]`. Agents already deployed on
machines we cannot reach send plain JSON objects, so a raw dict must still
coerce — a model that only accepted `GpuInfo` instances would be a breaking
change dressed up as a type annotation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from flashruntime.protocol.v1alpha1 import GpuInfo, NodeCapabilities, ResourcesSpec


# ---------------------------------------------------------------------------
# GpuInfo
# ---------------------------------------------------------------------------


def test_gpu_info_round_trips_with_only_index_set():
    """Every field but `index` is optional: a probe that cannot read a value
    says nothing rather than guessing, and the omission must survive JSON."""
    gpu = GpuInfo(index=0)
    restored = GpuInfo.model_validate_json(gpu.model_dump_json())

    assert restored.index == 0
    assert restored.name == ""
    assert restored.memory_total_mb is None
    assert restored.driver_version == ""
    assert restored.compute_capability == ""


def test_gpu_info_round_trips_a_fully_populated_device():
    gpu = GpuInfo(
        index=1,
        name="NVIDIA GeForce RTX 4090",
        memory_total_mb=24564,
        driver_version="550.54.14",
        compute_capability="8.9",
    )
    restored = GpuInfo.model_validate_json(gpu.model_dump_json())

    assert restored == gpu


def test_gpu_info_requires_an_index():
    """The one field a probe can always report. Without it there is no way to
    tell two devices apart, so it is not defaulted."""
    with pytest.raises(ValidationError):
        GpuInfo.model_validate({"name": "NVIDIA GeForce RTX 4090"})


# ---------------------------------------------------------------------------
# NodeCapabilities.gpus
# ---------------------------------------------------------------------------


def test_node_capabilities_accepts_gpu_info_instances():
    caps = NodeCapabilities(gpus=[GpuInfo(index=0)])

    assert len(caps.gpus) == 1
    assert caps.gpus[0].index == 0


def test_node_capabilities_coerces_raw_dicts_from_the_wire():
    """Deployed agents send JSON objects, not `GpuInfo` instances. The typing
    change must be additive on the wire or every already-enrolled machine
    fails to register the moment the coordinator upgrades."""
    caps = NodeCapabilities.model_validate(
        {
            "cpu_cores": 8,
            "gpus": [
                {"index": 0, "name": "NVIDIA A10G", "memory_total_mb": 22731},
                {"index": 1},
            ],
        }
    )

    assert [g.index for g in caps.gpus] == [0, 1]
    assert isinstance(caps.gpus[0], GpuInfo)
    assert caps.gpus[0].name == "NVIDIA A10G"
    assert caps.gpus[1].name == ""


def test_node_capabilities_ignores_fields_a_newer_probe_might_add():
    """Forward compatibility in the other direction: a newer agent reporting a
    field this coordinator does not know must not fail registration."""
    caps = NodeCapabilities.model_validate(
        {"gpus": [{"index": 0, "some_future_field": "ignored"}]}
    )

    assert caps.gpus[0].index == 0


def test_node_capabilities_defaults_to_no_gpus():
    """Absent means "no GPU", never "unknown, assume yes" — the gate reads
    this list and its length is the whole matching rule in v1."""
    assert NodeCapabilities().gpus == []
    assert NodeCapabilities.model_validate({"cpu_cores": 4}).gpus == []


def test_node_capabilities_gpu_default_is_not_shared_between_instances():
    # `default_factory` and not a bare `[]`: one shared list would let a
    # single node's devices leak into every other node's capabilities.
    first = NodeCapabilities()
    second = NodeCapabilities()

    assert first.gpus is not second.gpus
    first.gpus.append(GpuInfo(index=0))
    assert second.gpus == []


# ---------------------------------------------------------------------------
# ResourcesSpec.gpuPerTask
# ---------------------------------------------------------------------------


def test_gpu_per_task_defaults_to_zero():
    """0 means "no GPU required", which is every job that exists today."""
    assert ResourcesSpec().gpuPerTask == 0


def test_gpu_per_task_accepts_a_positive_count():
    assert ResourcesSpec(gpuPerTask=2).gpuPerTask == 2


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="fractional"),
        pytest.param("one", id="not-a-number"),
    ],
)
def test_gpu_per_task_rejects_values_that_are_not_a_device_count(value):
    with pytest.raises(ValidationError):
        ResourcesSpec(gpuPerTask=value)


def test_gpu_per_task_round_trips_through_json():
    spec = ResourcesSpec(gpuPerTask=1)
    restored = ResourcesSpec.model_validate_json(spec.model_dump_json())

    assert restored.gpuPerTask == 1
