"""Trusted-pool argv placement (AGENTS.md rule 3 — every leg fails closed).

An argv payload normally requires the containerised argv contract
(argv_capable). Inside a team pool the host's OPERATOR may opt into running
pool argv work unsandboxed. Three legs, all required: the task is
pool-scoped, its submitter waived the tier (allowFallback), and the node
opted in. Any one alone must place nothing — `CommandRecipe` now accepts the
waiver only when it is pool-scoped (`allowFallback` iff `placement.pool` is
set), so nothing upstream should ever produce a waiver without a pool
either; this gate must not rely on that upstream coupling regardless of its
current shape.
"""

import pytest


def _argv_task(pool=None, allow_fallback=True, **payload_extra):
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {
        "argv": ["python", "/work/inputs/code/train.py"],
        "isolation": {"tier": "sandboxed", "allowFallback": allow_fallback},
    }
    if pool is not None:
        payload["pool"] = pool
    payload.update(payload_extra)
    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload=payload,
    )


def _node(pools=None, **extra):
    capabilities = {"cpu_cores": 8}
    if pools is not None:
        capabilities["pools"] = pools
    node = {"node_id": "n1", "capabilities": capabilities}
    node.update(extra)
    return node


def test_docker_argv_node_still_takes_pool_argv_work():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(pools=["p-1"], argv_capable=True, sandbox_capable=True)
    assert IsolationAwarePlacement().eligible(_argv_task("p-1"), node) is True


def test_trusted_node_takes_pool_argv_work_without_docker():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(pools=["p-1"], unsandboxed_argv_capable=True)
    assert IsolationAwarePlacement().eligible(_argv_task("p-1"), node) is True


def test_trusted_node_never_takes_a_public_sandboxed_argv_job():
    """The leg that keeps strangers' code off trusting hosts: no pool, no
    trusted placement — even though the node opted in."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(unsandboxed_argv_capable=True)
    assert IsolationAwarePlacement().eligible(
        _argv_task(None, allow_fallback=False), node
    ) is False


def test_pool_without_waiver_does_not_unlock_trusted_argv():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(pools=["p-1"], unsandboxed_argv_capable=True)
    assert IsolationAwarePlacement().eligible(
        _argv_task("p-1", allow_fallback=False), node
    ) is False


def test_node_that_did_not_opt_in_is_refused():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(pools=["p-1"])  # member, subprocess-only, no opt-in
    assert IsolationAwarePlacement().eligible(_argv_task("p-1"), node) is False


@pytest.mark.parametrize("optin", [1, "true", None, [True]])
def test_type_confused_opt_in_fails_closed(optin):
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(pools=["p-1"], unsandboxed_argv_capable=optin)
    assert IsolationAwarePlacement().eligible(_argv_task("p-1"), node) is False


def test_trusted_placement_still_respects_the_pool_gate():
    """Both gates apply: a trusted opted-in node OUTSIDE the pool refuses."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = _node(pools=["p-2"], unsandboxed_argv_capable=True)
    assert IsolationAwarePlacement().eligible(_argv_task("p-1"), node) is False


def test_claim_endpoint_forwards_the_trusted_opt_in_to_the_gate():
    """End to end over the claim endpoint (the hop that broke local_datasets
    three times, per test_placement_pool's own claim test): registration's
    unsandboxed_argv_capable must survive register -> node view -> gate.
    Every eligible()-level test above hand-builds a flat node dict and so
    cannot see this hop at all — deleting the forwarding line in
    service/modea.py's claim node_view makes `node.get(...)` read None, the
    gate's `is True` check fails closed, and the opted-in node below would
    wrongly stay ineligible. This test is what catches that line going
    missing; the unit tests above cannot."""
    import pathlib

    import fastapi
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.service.modea import ModeAState, build_router

    state = ModeAState(LeaseManager(), artifacts_dir=pathlib.Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    state.manager.add_task(_argv_task("p-1"))

    def register(node_id: str, opted_in: bool):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={
                "node_id": node_id, "kubernetes_node": "", "hostname": node_id,
                "unsandboxed_argv_capable": opted_in,
                "capabilities": {"cpu_cores": 8, "pools": ["p-1"]},
            },
        )
        assert r.status_code == 200

    register("not-opted-in", False)
    register("trusted", True)

    assert client.post(
        "/v1alpha1/leases/claim", json={"node_id": "not-opted-in"}
    ).status_code == 204
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "trusted"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000"
