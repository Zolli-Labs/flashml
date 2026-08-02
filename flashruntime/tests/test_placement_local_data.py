"""The fourth fail-closed placement gate: a task that names `local_inputs` may
only be leased to a node advertising every one of those names in its
`local_datasets` capability (AGENTS.md rule 3).

The data never leaves the host, so placement is the only thing standing between
a task and a machine that cannot serve it. An absent, None, or wrongly-typed
capability counts as NOT capable, exactly like `sandbox_capable`/`argv_capable`.
"""

from __future__ import annotations

import pytest


def _local_task(local_inputs=("patients",), **payload_extra):
    """A task requiring local datasets. `local_inputs` is passed through RAW so
    tests can poison it with type-confused values the protocol would never
    emit (the same trick `_task_raw` plays on the isolation payload)."""
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {"module": "flashml_workloads.sklearn_trial"}
    if local_inputs is not None:
        payload["local_inputs"] = local_inputs
    payload.update(payload_extra)
    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload=payload,
    )


def test_node_advertising_a_superset_is_eligible():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "local_datasets": ["patients", "labs"]}
    assert IsolationAwarePlacement().eligible(_local_task(["patients"]), node) is True


def test_node_advertising_every_required_name_is_eligible():
    """Two requirements, both advertised — the gate is per-name, not per-task."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "local_datasets": ["labs", "patients"]}
    task = _local_task(["patients", "labs"])
    assert IsolationAwarePlacement().eligible(task, node) is True


def test_node_advertising_only_some_required_names_is_ineligible():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "local_datasets": ["patients"]}
    task = _local_task(["patients", "labs"])
    assert IsolationAwarePlacement().eligible(task, node) is False


def test_node_advertising_a_different_dataset_is_ineligible():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "local_datasets": ["labs"]}
    assert IsolationAwarePlacement().eligible(_local_task(["patients"]), node) is False


@pytest.mark.parametrize(
    "advertised",
    [
        pytest.param([], id="empty-list"),
        pytest.param(None, id="explicit-none"),
        pytest.param("patients", id="bare-string-type-confusion"),
        pytest.param({"patients": "/srv/data"}, id="dict-type-confusion"),
        pytest.param(("patients",), id="tuple-type-confusion"),
        pytest.param(True, id="truthy-boolean"),
    ],
)
def test_ineligible_unless_the_capability_is_a_list_containing_the_name(advertised):
    """Fail closed on absence AND on type confusion. The bare string
    "patients" must not count: `"patients" in "patients"` is True in Python,
    which is exactly the substring trap `sandbox_capable is True` avoids for
    booleans. Only a genuine list of names advertises anything."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "local_datasets": advertised}
    assert IsolationAwarePlacement().eligible(_local_task(["patients"]), node) is False


def test_node_with_no_local_datasets_field_at_all_is_ineligible():
    """An already-deployed agent whose registration predates the field
    advertises nothing — absent must read as "offers nothing", never as
    "offers anything"."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1"}  # no local_datasets key at all
    assert IsolationAwarePlacement().eligible(_local_task(["patients"]), node) is False


def test_allow_fallback_cannot_bypass_the_local_data_gate():
    """allowFallback waives the sandbox-tier requirement only. It must not also
    waive the local-data gate — the same reasoning that stops it waiving the
    argv gate: a submitter cannot grant themselves data a host never offered."""
    from flashruntime.scheduler import IsolationAwarePlacement

    task = _local_task(
        ["patients"], isolation={"tier": "sandboxed", "allowFallback": True}
    )
    node = {"node_id": "n1", "sandbox_capable": False, "local_datasets": []}
    assert IsolationAwarePlacement().eligible(task, node) is False


def test_allow_fallback_does_not_bypass_it_on_a_node_missing_the_field():
    from flashruntime.scheduler import IsolationAwarePlacement

    task = _local_task(
        ["patients"], isolation={"tier": "sandboxed", "allowFallback": True}
    )
    assert IsolationAwarePlacement().eligible(task, {"node_id": "n1"}) is False


def test_type_confused_local_inputs_is_ineligible_everywhere():
    """A poisoned requirement (present but not a list, e.g. the bare string
    "patients") must fail closed on every node — including one that advertises
    the name — and must never crash the predicate."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    node = {"node_id": "n1", "local_datasets": ["patients"]}
    assert policy.eligible(_local_task("patients"), node) is False
    assert policy.eligible(_local_task({"patients": 1}), node) is False


def test_tasks_without_local_inputs_are_unaffected():
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    task = _local_task(None)  # no local_inputs key in the payload at all
    assert policy.eligible(task, {"node_id": "n1"}) is True
    assert policy.eligible(task, {"node_id": "n2", "local_datasets": ["labs"]}) is True


def test_empty_local_inputs_list_requires_nothing():
    """`local_inputs: []` is what a job that opted out of the feature compiles
    to — it demands nothing, so it runs anywhere, like tier "standard"."""
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_local_task([]), {"node_id": "n1"}) is True


def test_claim_over_a_local_data_queue_serves_the_clean_task():
    """A local-data task at the head of the queue must not block a node that
    cannot serve it — the next clean task is still leased, and the local-data
    task is never mis-leased."""
    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.scheduler import IsolationAwarePlacement

    mgr = LeaseManager()
    mgr.add_task(_local_task(["patients"]))  # head of queue, needs data
    mgr.add_task(
        TaskSpec(task_id="task-001", job_id="job-a", commit_key="job-a/task-001/m.json")
    )
    policy = IsolationAwarePlacement()

    lease = mgr.claim("n1", policy=policy, node={"node_id": "n1"})
    assert lease is not None
    assert lease.task_id == "task-001"

    holder = {"node_id": "n2", "local_datasets": ["patients"]}
    lease2 = mgr.claim("n2", policy=policy, node=holder)
    assert lease2 is not None
    assert lease2.task_id == "task-000"


def test_claim_endpoint_leases_local_data_work_only_to_the_holder():
    """End to end over the claim endpoint: the registry's claim-time node view
    must forward the advertised names, or the gate reads an absent capability
    on EVERY node and a local-data task becomes unplaceable everywhere — fail
    closed, but also permanently stuck."""
    import pathlib

    import fastapi
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.service.modea import ModeAState, build_router

    state = ModeAState(LeaseManager(), artifacts_dir=pathlib.Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)
    state.manager.add_task(_local_task(["patients"]))

    def register(node_id: str, local_datasets: list[str]):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={
                "node_id": node_id, "kubernetes_node": "", "hostname": node_id,
                "capabilities": {}, "local_datasets": local_datasets,
            },
        )
        assert r.status_code == 200

    register("no-data-node", [])
    register("holder-node", ["patients", "labs"])

    assert client.post("/v1alpha1/leases/claim", json={"node_id": "no-data-node"}).status_code == 204
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "holder-node"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000"
