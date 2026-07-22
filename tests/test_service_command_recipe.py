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
