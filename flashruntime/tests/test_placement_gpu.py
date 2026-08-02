"""The fifth fail-closed placement gate: a task whose payload asks for `gpus: N`
may only be leased to a node whose `capabilities.gpus` is a list of at least N
entries (AGENTS.md rule 3).

Polarity follows `argv_capable` / `local_datasets`, not `module_capable`. A
misplaced module task only wastes retry attempts; a CUDA job on a CPU-only box
either crashes on `torch.cuda.is_available()` or silently falls back to the CPU
and runs two orders of magnitude slower while reporting success. Neither is
something to discover from a bill.

The gate is one-directional: a GPU node still receives CPU work.
"""

from __future__ import annotations

import pytest


def _gpu_task(gpus=1, **payload_extra):
    """A task requiring GPUs. `gpus` is passed through RAW so tests can poison
    it with type-confused values the protocol would never emit (the same trick
    `_local_task` plays on `local_inputs`)."""
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {"module": "flashml_workloads.sklearn_trial"}
    if gpus is not None:
        payload["gpus"] = gpus
    payload.update(payload_extra)
    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload=payload,
    )


def _node(gpu_count=0, **extra):
    """A node view shaped like the claim endpoint's: the GPU list lives under
    `capabilities`, which is `NodeCapabilities.model_dump()` — plain dicts by
    the time placement sees them, not GpuInfo instances."""
    node = {
        "node_id": "n1",
        "capabilities": {
            "cpu_cores": 8,
            "gpus": [{"index": i, "name": "NVIDIA A10G"} for i in range(gpu_count)],
        },
    }
    node.update(extra)
    return node


# ---------------------------------------------------------------------------
# DoD 3–6: the four placement outcomes
# ---------------------------------------------------------------------------


def test_one_gpu_job_is_refused_on_a_node_advertising_none():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(1), _node(0)) is False


def test_one_gpu_job_is_placed_on_a_node_advertising_one():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(1), _node(1)) is True


def test_two_gpu_job_is_refused_on_a_one_gpu_node():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(2), _node(1)) is False


def test_two_gpu_job_is_placed_on_a_two_gpu_node():
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(2), _node(2)) is True


def test_a_job_with_no_gpu_requirement_is_placed_on_both():
    """The key must be ABSENT, not 0 — this is the branch every job that
    exists today takes."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    task = _gpu_task(None)  # no "gpus" key in the payload at all
    assert policy.eligible(task, _node(0)) is True
    assert policy.eligible(task, _node(2)) is True


def test_gpus_zero_requires_nothing_and_runs_anywhere():
    """`gpus: 0` is what an explicit opt-out compiles to. It demands nothing,
    like tier "standard" and an empty `local_inputs`."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_gpu_task(0), _node(0)) is True
    assert policy.eligible(_gpu_task(0), {"node_id": "n1"}) is True


def test_a_gpu_node_still_receives_cpu_work():
    """The gate is one-directional. Reserving GPU hosts for GPU jobs is a
    separate scheduling decision; making it a gate here would idle the
    scarcest hardware on the network."""
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(None), _node(4)) is True


# ---------------------------------------------------------------------------
# Fail-closed: the advertised capability
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "advertised",
    [
        pytest.param([], id="empty-list"),
        pytest.param(None, id="explicit-none"),
        pytest.param("gpu", id="bare-string-type-confusion"),
        pytest.param({}, id="empty-dict-type-confusion"),
        pytest.param({"0": "NVIDIA A10G"}, id="dict-type-confusion"),
        pytest.param(({"index": 0},), id="tuple-type-confusion"),
        pytest.param(True, id="truthy-boolean"),
        pytest.param(1, id="bare-count-int"),
    ],
)
def test_ineligible_unless_the_capability_is_a_list_long_enough(advertised):
    """Fail closed on absence AND on type confusion. A bare `1` is the most
    tempting wrong answer — it reads like "one GPU" and has no length, so
    accepting it would mean writing a second, looser matching rule beside the
    one the probe actually feeds."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "capabilities": {"gpus": advertised}}
    assert IsolationAwarePlacement().eligible(_gpu_task(1), node) is False


def test_node_with_no_capabilities_key_at_all_is_ineligible():
    """An already-deployed agent whose registration predates GPU probing
    advertises nothing — absent must read as "has none", never "unknown, try
    it and see"."""
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(1), {"node_id": "n1"}) is False


def test_node_with_capabilities_none_is_ineligible():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "capabilities": None}
    assert IsolationAwarePlacement().eligible(_gpu_task(1), node) is False


def test_node_with_type_confused_capabilities_is_ineligible_without_crashing():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "capabilities": "8 cores, 1 gpu"}
    assert IsolationAwarePlacement().eligible(_gpu_task(1), node) is False


def test_node_capabilities_without_a_gpus_key_is_ineligible():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "capabilities": {"cpu_cores": 8}}
    assert IsolationAwarePlacement().eligible(_gpu_task(1), node) is False


# ---------------------------------------------------------------------------
# Fail-closed: the requirement itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "required",
    [
        pytest.param("1", id="numeric-string"),
        pytest.param(-1, id="negative"),
        pytest.param(1.5, id="fractional"),
        pytest.param(1.0, id="whole-float"),
        pytest.param([1], id="list"),
        pytest.param({"count": 1}, id="dict"),
        pytest.param(True, id="boolean-true"),
    ],
)
def test_type_confused_requirement_is_ineligible_everywhere(required):
    """A poisoned requirement must fail closed on EVERY node — including one
    with plenty of GPUs — and must never crash the predicate.

    `True` is the one that needs saying out loud: `bool` is a subclass of
    `int` in Python, so `isinstance(True, int)` is True and `True >= 1`.
    Without the explicit bool exclusion, `gpus: true` would silently mean
    "one GPU" — a JSON typo that placed real work.
    """
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    assert policy.eligible(_gpu_task(required), _node(0)) is False
    assert policy.eligible(_gpu_task(required), _node(8)) is False


def test_boolean_false_is_not_read_as_zero_gpus():
    """The mirror of the above: `False` is not "no GPU required" either. It is
    a type-confused requirement and fails closed like every other."""
    from flashruntime.scheduler import IsolationAwarePlacement

    assert IsolationAwarePlacement().eligible(_gpu_task(False), _node(8)) is False


# ---------------------------------------------------------------------------
# Interaction with the existing gates
# ---------------------------------------------------------------------------


def test_allow_fallback_cannot_bypass_the_gpu_gate():
    """allowFallback waives the sandbox-tier requirement only. It is the
    submitter's statement about their own isolation posture, and has nothing
    to say about hardware that either exists or does not."""
    from flashruntime.scheduler import IsolationAwarePlacement

    task = _gpu_task(1, isolation={"tier": "sandboxed", "allowFallback": True})
    node = _node(0, sandbox_capable=True)
    assert IsolationAwarePlacement().eligible(task, node) is False


def test_a_gpu_node_still_fails_the_argv_gate_it_does_not_pass():
    """The GPU gate adds a requirement; it never satisfies another one."""
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.scheduler import IsolationAwarePlacement

    task = TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload={"argv": ["python", "train.py"], "gpus": 1},
    )
    assert IsolationAwarePlacement().eligible(task, _node(1)) is False
    assert IsolationAwarePlacement().eligible(task, _node(1, argv_capable=True)) is True


def test_claim_over_a_gpu_queue_serves_the_clean_task():
    """A GPU task at the head of the queue must not block a CPU-only node —
    the next clean task is still leased, and the GPU task is never
    mis-leased."""
    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.scheduler import IsolationAwarePlacement

    mgr = LeaseManager()
    mgr.add_task(_gpu_task(1))  # head of queue, needs a GPU
    mgr.add_task(
        TaskSpec(task_id="task-001", job_id="job-a", commit_key="job-a/task-001/m.json")
    )
    policy = IsolationAwarePlacement()

    lease = mgr.claim("n1", policy=policy, node=_node(0))
    assert lease is not None
    assert lease.task_id == "task-001"

    lease2 = mgr.claim("n2", policy=policy, node=_node(1, node_id="n2"))
    assert lease2 is not None
    assert lease2.task_id == "task-000"


def test_claim_endpoint_leases_gpu_work_only_to_a_gpu_host():
    """End to end over the claim endpoint: the registry's claim-time node view
    must forward `capabilities.gpus`, or the gate reads an absent capability on
    EVERY node and a GPU task becomes unplaceable everywhere.

    This pins hop 1 of the three that broke the `local_datasets` work — the
    node view. It is believed correct today (`capabilities` is forwarded whole
    as `model_dump()`), and this test is what keeps it that way.
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
    state.manager.add_task(_gpu_task(1))

    def register(node_id: str, gpus: list[dict]):
        r = client.post(
            "/v1alpha1/nodes/register",
            json={
                "node_id": node_id, "kubernetes_node": "", "hostname": node_id,
                "capabilities": {"cpu_cores": 8, "gpus": gpus},
            },
        )
        assert r.status_code == 200

    register("cpu-node", [])
    register("gpu-node", [{"index": 0, "name": "NVIDIA A10G", "memory_total_mb": 22731}])

    assert client.post("/v1alpha1/leases/claim", json={"node_id": "cpu-node"}).status_code == 204
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "gpu-node"})
    assert r.status_code == 200
    assert r.json()["task_id"] == "task-000"
