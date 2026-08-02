"""`NodeRegistration.local_datasets` — the names of on-host datasets a node
advertises, and the fail-closed defaulting that governs them.

These are protocol tests, not scheduler tests: they pin the *wire* behaviour
that the placement gate is later allowed to trust. The gate can only fail
closed if the field itself does.
"""

from flashruntime.protocol.v1alpha1 import NodeRegistration


def make_registration(**overrides):
    base = {
        "node_id": "node-a",
        "kubernetes_node": "worker-0",
        "hostname": "worker-0.local",
        "capabilities": {"cpu_cores": 4, "os": "linux", "architecture": "amd64"},
    }
    base.update(overrides)
    return base


def test_local_datasets_defaults_empty_when_absent_from_the_wire():
    # An agent deployed before this field existed sends a registration with no
    # `local_datasets` key at all. It must decode as advertising nothing, so it
    # is never eligible for local-data work until it is upgraded and opts in
    # explicitly (AGENTS.md rule 3: security-relevant fields fail closed).
    reg = NodeRegistration.model_validate(make_registration())
    assert reg.local_datasets == []


def test_local_datasets_round_trips_through_json():
    reg = NodeRegistration.model_validate(
        make_registration(local_datasets=["patients", "labs"])
    )
    restored = NodeRegistration.model_validate_json(reg.model_dump_json())
    assert restored.local_datasets == ["patients", "labs"]


def test_local_datasets_default_is_not_shared_between_instances():
    # A bare `[]` default would be one list shared by every NodeRegistration
    # ever constructed: one node appending to it would silently advertise its
    # datasets on behalf of every other node in the pool. `default_factory`
    # is what stops that, and this test is what stops someone "simplifying"
    # it back.
    first = NodeRegistration.model_validate(make_registration(node_id="node-a"))
    second = NodeRegistration.model_validate(make_registration(node_id="node-b"))

    assert first.local_datasets is not second.local_datasets

    first.local_datasets.append("patients")
    assert second.local_datasets == []
