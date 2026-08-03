"""The sixth fail-closed placement gate: a task whose payload lists
`exclude_nodes` may not be leased to a node named there (AGENTS.md rule 3).

Why it exists. Upfront redundant assignment
(`2026-08-03-result-verification-design.md` §4.2) expands one checked task
into two ordinary tasks with the same payload and mutually exclusive
placement. Nothing in the runtime could say "anywhere but there", and with a
fleet of two machines the twin lands on the same node half the time —
verifying nothing while costing double. The rest of that feature (pair
expansion, dispatch, comparison) is deliberately not built here; this is the
one runtime change it cannot be built without.

Polarity: **fail closed**, like `argv_capable`, `local_datasets` and `gpus`,
and deliberately not like `module_capable`. A task that ends up placeable
nowhere simply never runs, and the redundancy slice records `unknown` — never
`pass`. Placing the twin on the excluded node is the failure that is not
detectable afterwards: the pair agrees, the verdict is recorded as a match,
and nothing was verified. An unrunnable task announces itself; a fake
verification does not.
"""

from __future__ import annotations

import pytest


def _task(excluded=None, **payload_extra):
    """A task carrying `exclude_nodes` RAW, so tests can poison it with
    type-confused values the coordinator would never emit (the same trick
    `_gpu_task` plays on `gpus`).

    `None` means the key is absent entirely. Callers pass the list they mean:
    a default of `"node-a"` here made three tests pass for the wrong reason —
    the bare string is itself refused as type-confused, so they never
    exercised the exclusion match at all.
    """
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {"module": "flashml_workloads.sklearn_trial"}
    if excluded is not None:
        payload["exclude_nodes"] = excluded
    payload.update(payload_extra)
    return TaskSpec(
        task_id="task-000~v", job_id="job-a", commit_key="job-a/task-000~v/m.json",
        payload=payload,
    )


def _node(node_id="node-a", **extra):
    node = {"node_id": node_id, "capabilities": {"cpu_cores": 8}}
    node.update(extra)
    return node


# ---------------------------------------------------------------------------
# The two placement outcomes
# ---------------------------------------------------------------------------


def test_the_named_node_is_refused():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_task(["node-a"]), _node("node-a")) is False


def test_any_other_node_still_gets_it():
    """The twin has to land SOMEWHERE or the verification never happens."""
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_task(["node-a"]), _node("node-b")) is True


def test_several_nodes_can_be_excluded_at_once():
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    task = _task(["node-a", "node-b"])
    assert policy.eligible(task, _node("node-a")) is False
    assert policy.eligible(task, _node("node-b")) is False
    assert policy.eligible(task, _node("node-c")) is True


def test_a_task_with_no_exclusion_runs_anywhere():
    """The branch every job that exists today takes. The key must be ABSENT,
    not empty."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task(None), _node("node-a")) is True
    assert policy.eligible(_task(None), _node("node-z")) is True


def test_an_empty_exclusion_list_excludes_nobody():
    """`[]` is what the FIRST member of a verification pair carries: it is
    dispatched before anyone has claimed the other, so it excludes nothing
    yet. It must run anywhere, like `gpus: 0` and an empty `local_inputs`."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task([]), _node("node-a")) is True
    assert policy.eligible(_task([]), {"node_id": "node-a"}) is True


def test_exclusion_matches_a_whole_name_and_never_a_fragment():
    """`local_datasets` was nearly broken by `"patients" in "patients-2024"`.
    List membership is exact, and this test is what keeps anyone from
    "improving" it into a prefix or substring check."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task(["node-a"]), _node("node-a-2")) is True
    assert policy.eligible(_task(["node-a-2"]), _node("node-a")) is True


# ---------------------------------------------------------------------------
# Fail-closed: the requirement itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "excluded",
    [
        pytest.param("node-a", id="bare-string"),
        pytest.param({"node-a"}, id="set"),
        pytest.param(("node-a",), id="tuple"),
        pytest.param({"node_id": "node-a"}, id="dict"),
        pytest.param(1, id="int"),
        pytest.param(True, id="boolean"),
    ],
)
def test_a_type_confused_exclusion_is_ineligible_everywhere(excluded):
    """Fail closed on EVERY node, including ones nobody meant to exclude, and
    without crashing the predicate — exactly as a non-list `local_inputs`
    does.

    The bare string is the one that has to be said out loud. `"node-a" in
    "node-a"` is True in Python but `"node-a" in "node-alpha"` is True as
    well, so a string that happened to work would silently exclude every host
    whose name contains another's. Refusing the shape outright is the only
    reading that does not depend on what the names happen to be.
    """
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task(excluded), _node("node-a")) is False
    assert policy.eligible(_task(excluded), _node("node-z")) is False


@pytest.mark.parametrize(
    "excluded",
    [
        pytest.param(["node-a", None], id="null-member"),
        pytest.param(["node-a", 7], id="int-member"),
        pytest.param([["node-b"]], id="nested-list-member"),
        pytest.param([{"node_id": "node-b"}], id="dict-member"),
    ],
)
def test_a_list_holding_anything_but_names_is_ineligible_everywhere(excluded):
    """A member that is not a name means the exclusion was built wrong, and
    the node it meant to exclude is exactly the one a membership test would
    now let through. Refusing the whole task costs a task; trusting it costs
    the verification that task existed for."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task(excluded), _node("node-a")) is False
    assert policy.eligible(_task(excluded), _node("node-b")) is False


# ---------------------------------------------------------------------------
# Fail-closed: the node's own identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        pytest.param({}, id="no-node-id-key"),
        pytest.param({"node_id": None}, id="explicit-none"),
        pytest.param({"node_id": ""}, id="empty-string"),
        pytest.param({"node_id": 7}, id="int"),
        pytest.param({"node_id": ["node-b"]}, id="list"),
    ],
)
def test_a_node_that_cannot_be_identified_is_refused(node):
    """The gate answers "is this the node we must avoid?". A view with no
    readable identity cannot answer it, and "we could not tell" must not
    resolve to "go ahead" — that is the failure the whole gate exists to
    prevent, arrived at by a different route.

    It costs nothing real: every node view the claim endpoint builds carries
    the node_id it just authenticated.
    """
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_task(["node-a"]), node) is False


def test_an_unidentifiable_node_still_gets_work_that_excludes_nobody():
    """Scoped to tasks that actually exclude something. An empty or absent
    exclusion asks no question, so there is nothing to fail closed about."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_task(None), {}) is True
    assert policy.eligible(_task([]), {}) is True


# ---------------------------------------------------------------------------
# Interaction with the existing gates
# ---------------------------------------------------------------------------


def test_allow_fallback_cannot_bypass_the_exclusion_gate():
    """`allowFallback` waives the sandbox-TIER requirement and nothing else.
    It is the submitter's statement about their own isolation posture; it has
    nothing to say about which machine already holds the other half of a
    verification pair."""
    from flashruntime.scheduler import IsolationAwarePlacement

    task = _task(["node-a"], isolation={"tier": "sandboxed", "allowFallback": True})
    assert IsolationAwarePlacement().eligible(task, _node("node-a")) is False


def test_the_exclusion_gate_adds_a_requirement_and_satisfies_none():
    """A node that is not excluded still has to pass everything else."""
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.scheduler import IsolationAwarePlacement

    task = TaskSpec(
        task_id="task-000~v", job_id="job-a", commit_key="job-a/task-000~v/m.json",
        payload={"argv": ["python", "train.py"], "exclude_nodes": ["node-a"]},
    )
    policy = IsolationAwarePlacement()
    assert policy.eligible(task, _node("node-b")) is False  # no argv runner
    assert policy.eligible(task, _node("node-b", argv_capable=True)) is True


def test_claim_over_a_mixed_queue_serves_the_excluded_node_something_else():
    """An excluded task at the head of the queue must not block the machine
    it excludes — the next clean task is still leased, and the twin is never
    mis-placed."""
    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.scheduler import IsolationAwarePlacement

    mgr = LeaseManager()
    mgr.add_task(_task(["node-a"]))  # head of queue, must avoid node-a
    mgr.add_task(
        TaskSpec(task_id="task-001", job_id="job-a", commit_key="job-a/task-001/m.json")
    )
    policy = IsolationAwarePlacement()

    lease = mgr.claim("node-a", policy=policy, node=_node("node-a"))
    assert lease is not None and lease.task_id == "task-001"

    lease2 = mgr.claim("node-b", policy=policy, node=_node("node-b"))
    assert lease2 is not None and lease2.task_id == "task-000~v"


def test_claim_endpoint_keeps_the_twin_off_the_node_that_took_the_original():
    """End to end over the claim endpoint, the way the redundancy slice will
    use it: the node view the registry builds must carry `node_id`, or the
    gate reads an unidentifiable node on EVERY claim and an excluded task
    becomes unplaceable everywhere.

    This pins the same hop `test_placement_gpu` pins for `capabilities.gpus`.
    """
    import pathlib

    import fastapi
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.service.modea import ModeAState, build_router

    state = ModeAState(LeaseManager(), artifacts_dir=pathlib.Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    state.manager.add_task(_task(["node-a"]))

    for node_id in ("node-a", "node-b"):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={
                "node_id": node_id, "kubernetes_node": "", "hostname": node_id,
                "capabilities": {"cpu_cores": 8},
            },
        )
        assert r.status_code == 200

    assert client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).status_code == 204
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "node-b"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000~v"
