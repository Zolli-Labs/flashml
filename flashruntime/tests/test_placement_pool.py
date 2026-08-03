"""Seventh placement gate: team pools (AGENTS.md rule 3 — fail closed).

A pool-scoped task must land only on a node whose stamped capabilities list
that pool. The failure mode this prevents is the design's worst one: pool
jobs carry allowFallback, so a task escaping the pool boundary would run
UNSANDBOXED on a stranger's machine.
"""

import pytest


def _task(pool=None, **payload_extra):
    """`pool` is passed RAW so tests can poison it (the `_gpu_task` trick).
    None means the key is absent entirely."""
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {"module": "flashml_workloads.sklearn_trial"}
    if pool is not None:
        payload["pool"] = pool
    payload.update(payload_extra)
    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload=payload,
    )


def _node(pools=None, **extra):
    """Node view shaped like the claim endpoint's: pools lives under
    `capabilities`, which is NodeCapabilities.model_dump() — plain dicts."""
    capabilities = {"cpu_cores": 8}
    if pools is not None:
        capabilities["pools"] = pools
    node = {"node_id": "n1", "capabilities": capabilities}
    node.update(extra)
    return node


def test_task_without_pool_places_anywhere():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_task(), _node()) is True


def test_pool_task_places_on_a_member_node():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(
        _task("p-1"), _node(pools=["p-1", "p-2"])
    ) is True


def test_pool_task_refuses_a_non_member_node():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(
        _task("p-1"), _node(pools=["p-2"])
    ) is False


def test_pool_task_refuses_a_node_with_no_pools_at_all():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_task("p-1"), _node()) is False


@pytest.mark.parametrize(
    "advertised",
    [None, "p-1", {"p-1": True}, 1, [None], [1], ["p-1", None], b"p-1"],
)
def test_type_confused_advertisement_fails_closed(advertised):
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(
        _task("p-1"), _node(pools=advertised)
    ) is False


@pytest.mark.parametrize("required", ["", 1, True, ["p-1"], {"id": "p-1"}])
def test_type_confused_requirement_fails_closed(required):
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(
        _task(required), _node(pools=["p-1"])
    ) is False


def test_capabilities_absent_or_confused_fails_closed():
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task("p-1"), {"node_id": "n1"}) is False
    assert policy.eligible(
        _task("p-1"), {"node_id": "n1", "capabilities": "confused"}
    ) is False


def test_allow_fallback_does_not_waive_the_pool_gate():
    """The waiver and the boundary must never trade places: allowFallback is
    what pool jobs CARRY, so it waiving this gate would unsandbox strangers."""
    from flashruntime.scheduler import IsolationAwarePlacement

    task = _task("p-1", isolation={"tier": "sandboxed", "allowFallback": True})
    assert IsolationAwarePlacement().eligible(task, _node(pools=["p-2"])) is False


def test_claim_endpoint_confines_a_pool_task_to_member_nodes():
    """End to end over the claim endpoint (the hop that broke local_datasets
    three times): capabilities.pools must survive register → node view →
    gate. Model on test_placement_gpu's claim test."""
    import pathlib

    import fastapi
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.service.modea import ModeAState, build_router

    state = ModeAState(LeaseManager(), artifacts_dir=pathlib.Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    state.manager.add_task(_task("p-1"))

    def register(node_id: str, pools: list[str]):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={"node_id": node_id, "kubernetes_node": "", "hostname": node_id,
                  "capabilities": {"cpu_cores": 8, "pools": pools}},
        )
        assert r.status_code == 200

    register("outsider", [])
    register("member", ["p-1"])

    assert client.post("/v1alpha1/leases/claim", json={"node_id": "outsider"}).status_code == 204
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "member"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000"


def test_heartbeat_pools_refresh_reaches_placement():
    """Membership change without an agent restart: a heartbeat carrying
    pools=[] must strip eligibility on the next claim. Read the heartbeat
    handler in service/modea.py first for the exact route shape."""
    import pathlib

    import fastapi
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.service.modea import ModeAState, build_router

    state = ModeAState(LeaseManager(), artifacts_dir=pathlib.Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    state.manager.add_task(_task("p-1"))

    r = client.post(
        "/v1alpha1/nodes/register",
        json={"node_id": "member", "kubernetes_node": "", "hostname": "member",
              "capabilities": {"cpu_cores": 8, "pools": ["p-1"]}},
    )
    assert r.status_code == 200

    hb = client.post("/v1alpha1/nodes/member/heartbeat",
                     json={"node_id": "member", "pools": []})
    assert hb.status_code == 200

    assert client.post("/v1alpha1/leases/claim", json={"node_id": "member"}).status_code == 204
