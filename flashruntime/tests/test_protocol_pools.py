"""Wire fields for team pools (AGENTS.md rule 3: security fields fail closed).

`pools` (plural, on capabilities) names TEAM pools minted by the cloud
control plane and is stamped server-side by its API proxy. It is not
`NodeRegistration.pool` (singular), a deployment-profile label ("local")
that predates teams and is only ever logged.
"""

from flashruntime.protocol.v1alpha1 import (
    NodeCapabilities,
    NodeHeartbeat,
    NodeRegistration,
    PlacementSpec,
)


def test_capabilities_pools_defaults_empty_and_round_trips():
    assert NodeCapabilities().pools == []
    caps = NodeCapabilities.model_validate({"pools": ["p-1", "p-2"]})
    assert caps.model_dump()["pools"] == ["p-1", "p-2"]


def test_capabilities_pools_default_is_not_shared_between_instances():
    a, b = NodeCapabilities(), NodeCapabilities()
    a.pools.append("p-1")
    assert b.pools == []


def test_registration_unsandboxed_argv_capable_defaults_false():
    reg = NodeRegistration(
        node_id="n", kubernetes_node="", hostname="h",
        capabilities=NodeCapabilities(),
    )
    assert reg.unsandboxed_argv_capable is False


def test_heartbeat_pools_defaults_none_meaning_no_statement():
    hb = NodeHeartbeat.model_validate({"node_id": "n"})
    assert hb.pools is None
    hb2 = NodeHeartbeat.model_validate({"node_id": "n", "pools": []})
    assert hb2.pools == []


def test_placement_pool_accepts_a_team_pool_id():
    spec = PlacementSpec.model_validate({"pool": "3f2a7b1e-team"})
    assert spec.pool == "3f2a7b1e-team"
    assert PlacementSpec().pool == "any"


def test_old_wire_shapes_still_validate():
    """An agent on 0.3.3 sends none of the new fields; nothing may break."""
    reg = NodeRegistration.model_validate(
        {"node_id": "n", "kubernetes_node": "", "hostname": "h",
         "capabilities": {"cpu_cores": 4}}
    )
    assert reg.capabilities.pools == []
    assert reg.unsandboxed_argv_capable is False
