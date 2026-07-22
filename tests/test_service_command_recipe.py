"""CommandRecipe: JobSpec{type: command} → lease tasks, isolation stamped,
dispatched through the recipe registry by expand_tasks."""

from __future__ import annotations

import pytest


def _jobspec(**over):
    from flashruntime.protocol.v1alpha1 import ImageSpec, IsolationSpec
    from flashruntime.workloads.command import CommandWorkload, to_jobspec

    defaults = dict(command="python train.py --lr {lr}", task_params=[{"lr": 0.1}, {"lr": 0.01}])
    defaults.update(over)
    wl = CommandWorkload(**defaults)
    return to_jobspec(wl, name="cmd-job", image=ImageSpec(repository="ghcr.io/me/img", tag="1.0"))


def test_expand_substitutes_params_and_stamps_isolation():
    from flashruntime.recipes.command import CommandRecipe
    from flashruntime.protocol.v1alpha1 import IsolationSpec

    spec = _jobspec(isolation=IsolationSpec(tier="sandboxed", allowFallback=False))
    tasks = CommandRecipe().expand("job1", spec)
    assert [t.task_id for t in tasks] == ["task-000", "task-001"]
    assert tasks[0].payload["argv"] == ["python", "train.py", "--lr", "0.1"]
    assert tasks[1].payload["argv"] == ["python", "train.py", "--lr", "0.01"]
    assert tasks[0].payload["isolation"] == {"tier": "sandboxed", "allowFallback": False}
    assert tasks[0].payload["image"] == "ghcr.io/me/img:1.0"
    assert tasks[0].commit_key == "jobs/job1/task-000/metrics.json"
    assert tasks[0].max_attempts == spec.spec.retryPolicy.maxTaskAttempts


def test_expand_single_task_when_no_fanout():
    from flashruntime.recipes.command import CommandRecipe

    tasks = CommandRecipe().expand("job1", _jobspec(command="python eval.py", task_params=None))
    assert len(tasks) == 1
    assert tasks[0].payload["argv"] == ["python", "eval.py"]


def test_expand_rejects_bad_params():
    from flashruntime.recipes.command import CommandRecipe

    spec = _jobspec()
    spec.spec.workload.parameters["command"] = "not-a-list"
    with pytest.raises(ValueError, match="argv"):
        CommandRecipe().expand("job1", spec)

    spec2 = _jobspec()
    spec2.spec.workload.parameters["task_params"] = [{"seed": 1}]  # {lr} unfilled
    with pytest.raises(ValueError, match="placeholder"):
        CommandRecipe().expand("job1", spec2)


def test_expand_tasks_dispatches_command_type_via_registry():
    from flashruntime.service import modea

    tasks = modea.expand_tasks("job1", _jobspec())
    assert len(tasks) == 2
    assert tasks[0].payload["argv"][0] == "python"


def test_legacy_expansions_still_work():
    from flashruntime.protocol.v1alpha1 import (
        ExecutionSpec, ImageSpec, JobMetadata, JobSpec, JobSpecInner, WorkloadSpec,
    )
    from flashruntime.service import modea

    spec = JobSpec(
        metadata=JobMetadata(name="sweep"),
        spec=JobSpecInner(
            execution=ExecutionSpec(backend="leases"),
            image=ImageSpec(repository="r", tag="1"),
            workload=WorkloadSpec(
                type="hyperparameter_search",
                parameters={"trials": [{"model": "logreg", "C": 0.1}]},
            ),
        ),
    )
    tasks = modea.expand_tasks("job2", spec)
    assert tasks[0].payload["module"] == "flashml_workloads.sklearn_trial"


def test_claim_endpoint_fails_closed_for_sandboxed_tasks():
    """Full HTTP path: a sandboxed command task is never leased to a
    non-sandbox node. Mirrors tests/test_service_modea.py conventions."""
    fastapi = pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import IsolationSpec
    from flashruntime.service.modea import ModeAState, build_router, expand_tasks

    state = ModeAState(LeaseManager(), artifacts_dir=__import__("pathlib").Path("/tmp"))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    client = TestClient(app)

    for task in expand_tasks("job1", _jobspec(isolation=IsolationSpec(tier="sandboxed"))):
        state.manager.add_task(task)

    def register(node_id: str, sandbox: bool):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={
                "node_id": node_id,
                "kubernetes_node": "",
                "hostname": node_id,
                "capabilities": {},
                "sandbox_capable": sandbox,
            },
        )
        assert r.status_code == 200

    register("plain-node", sandbox=False)
    register("sandbox-node", sandbox=True)

    # fail closed: plain node gets nothing
    assert client.post("/v1alpha1/leases/claim", json={"node_id": "plain-node"}).status_code == 204
    # sandbox-capable node gets the task
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "sandbox-node"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000"


# -- F1: legacy expansions must also stamp the isolation requirement -----------


def _legacy_spec(workload_type: str, parameters: dict, **iso):
    from flashruntime.protocol.v1alpha1 import (
        ExecutionSpec,
        ImageSpec,
        IsolationSpec,
        JobMetadata,
        JobSpec,
        JobSpecInner,
        WorkloadSpec,
    )

    return JobSpec(
        metadata=JobMetadata(name="sweep"),
        spec=JobSpecInner(
            execution=ExecutionSpec(backend="leases"),
            image=ImageSpec(repository="r", tag="1"),
            isolation=IsolationSpec(**iso) if iso else IsolationSpec(),
            workload=WorkloadSpec(type=workload_type, parameters=parameters),
        ),
    )


def test_legacy_hyperparameter_search_stamps_isolation():
    """A sandboxed hyperparameter_search job must stamp the isolation tier
    into every task payload — else the placement gate can't fail closed."""
    from flashruntime.service import modea

    spec = _legacy_spec(
        "hyperparameter_search",
        {"trials": [{"model": "logreg", "C": 0.1}]},
        tier="sandboxed",
        allowFallback=False,
    )
    tasks = modea.expand_tasks("job2", spec)
    assert tasks[0].payload["isolation"] == {"tier": "sandboxed", "allowFallback": False}


def test_legacy_kmeans_stamps_isolation():
    from flashruntime.service import modea

    spec = _legacy_spec(
        "sharded_kmeans",
        {"shards": ["artifact://s0", "artifact://s1"], "centroids": [[0.0], [1.0]]},
        tier="sandboxed",
        allowFallback=True,
    )
    tasks = modea.expand_tasks("jobK", spec)
    assert tasks[0].payload["isolation"] == {"tier": "sandboxed", "allowFallback": True}


def test_legacy_sandboxed_task_not_leased_to_incapable_node():
    """End-to-end of the stamp: a sandboxed legacy task sits unclaimed on a
    non-sandbox node and is only leased to a sandbox-capable one."""
    from flashruntime.leases import LeaseManager
    from flashruntime.scheduler import IsolationAwarePlacement
    from flashruntime.service import modea

    spec = _legacy_spec(
        "hyperparameter_search",
        {"trials": [{"model": "logreg", "C": 0.1}]},
        tier="sandboxed",
        allowFallback=False,
    )
    mgr = LeaseManager()
    for task in modea.expand_tasks("jobX", spec):
        mgr.add_task(task)
    policy = IsolationAwarePlacement()

    # fail closed: incapable node gets nothing
    assert mgr.claim("n2", policy=policy, node={"node_id": "n2", "sandbox_capable": False}) is None
    # capable node gets the task
    lease = mgr.claim("n1", policy=policy, node={"node_id": "n1", "sandbox_capable": True})
    assert lease is not None


# -- F2: literal / positional braces must be a 422, never a 500 ----------------


def test_expand_literal_braces_raise_valueerror_not_indexerror():
    """`{}` / `{0}` in argv are auto/positional fields str.format cannot fill
    from a params dict — they must surface as ValueError (→422), not
    IndexError (→500)."""
    from flashruntime.recipes.command import CommandRecipe

    spec = _jobspec(command=["echo", "{}"], task_params=[{"lr": 0.1}])
    with pytest.raises(ValueError, match="task 0"):
        CommandRecipe().expand("job1", spec)

    spec2 = _jobspec(command=["echo", "{0}"], task_params=[{"lr": 0.1}])
    with pytest.raises(ValueError, match="task 0"):
        CommandRecipe().expand("job1", spec2)
