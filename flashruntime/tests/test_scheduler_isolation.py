"""The one fail-closed placement rule the isolation contract requires:
sandboxed tasks go only to sandbox_capable nodes (AGENTS.md rule 3)."""

from __future__ import annotations

import pytest


def _task(task_id: str, tier: str | None = None, allow_fallback: bool = False):
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {}
    if tier is not None:
        payload["isolation"] = {"tier": tier, "allowFallback": allow_fallback}
    return TaskSpec(task_id=task_id, job_id="j1", commit_key=f"j1/{task_id}", payload=payload)


def _task_raw(task_id: str, isolation):
    """Build a task with a *raw* isolation payload (any JSON type) so tests can
    poison it with type-confused values the protocol would never emit."""
    from flashruntime.protocol.v1alpha1 import TaskSpec

    return TaskSpec(
        task_id=task_id,
        job_id="j1",
        commit_key=f"j1/{task_id}",
        payload={"isolation": isolation},
    )


def test_eligibility_matrix():
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    capable = {"node_id": "n1", "sandbox_capable": True}
    incapable = {"node_id": "n2", "sandbox_capable": False}
    unknown = {"node_id": "n3"}  # missing key ⇒ NOT capable (fail closed)

    assert policy.eligible(_task("t", tier="standard"), incapable)
    assert policy.eligible(_task("t"), incapable)  # no isolation payload ⇒ standard
    assert policy.eligible(_task("t", tier="sandboxed"), capable)
    assert not policy.eligible(_task("t", tier="sandboxed"), incapable)
    assert not policy.eligible(_task("t", tier="sandboxed"), unknown)
    assert policy.eligible(_task("t", tier="sandboxed", allow_fallback=True), incapable)


def test_claim_with_policy_fails_closed_and_preserves_fifo():
    from flashruntime.leases import LeaseManager
    from flashruntime.scheduler import IsolationAwarePlacement

    mgr = LeaseManager()
    mgr.add_task(_task("t-sandboxed", tier="sandboxed"))
    mgr.add_task(_task("t-standard"))
    policy = IsolationAwarePlacement()

    # incapable node: must skip the sandboxed head-of-queue and get the standard task
    lease = mgr.claim("n2", policy=policy, node={"node_id": "n2", "sandbox_capable": False})
    assert lease.task_id == "t-standard"

    # capable node: gets the sandboxed task
    lease2 = mgr.claim("n1", policy=policy, node={"node_id": "n1", "sandbox_capable": True})
    assert lease2.task_id == "t-sandboxed"

    # nothing left ⇒ None, and the sandboxed task was never mis-leased
    assert mgr.claim("n2", policy=policy, node={"node_id": "n2"}) is None


def test_claim_without_policy_is_unchanged():
    from flashruntime.leases import LeaseManager

    mgr = LeaseManager()
    mgr.add_task(_task("first"))
    mgr.add_task(_task("second"))
    assert mgr.claim("n1").task_id == "first"  # FIFO, exactly as before


# -- fail-closed hardening (type-confusion): security-review fix round 1 -------


def test_allow_fallback_json_string_false_does_not_waive():
    """A JSON-string 'false' is truthy in Python; it must NOT waive the gate.
    Only a genuine boolean True waives (AGENTS.md rule 3, fail closed)."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    incapable = {"node_id": "n2", "sandbox_capable": False}
    task = _task_raw("t", {"tier": "sandboxed", "allowFallback": "false"})
    assert not policy.eligible(task, incapable)


def test_sandbox_capable_json_string_false_is_not_capable():
    """A node advertising the string 'false' is truthy in Python but must
    read as NOT capable — only a genuine boolean True counts as capable."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    node_str_false = {"node_id": "n2", "sandbox_capable": "false"}
    assert not policy.eligible(_task("t", tier="sandboxed"), node_str_false)


def test_non_dict_isolation_is_ineligible_everywhere():
    """A type-confused isolation payload (present but not a dict) must fail
    closed — ineligible on every node, and never crash the predicate."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    capable = {"node_id": "n1", "sandbox_capable": True}
    incapable = {"node_id": "n2", "sandbox_capable": False}
    poisoned = _task_raw("t", "sandboxed")  # a bare string, not a dict
    assert not policy.eligible(poisoned, capable)
    assert not policy.eligible(poisoned, incapable)


def test_claim_over_poisoned_queue_serves_clean_task():
    """A poisoned task (non-dict isolation) must not crash choose() and block
    the whole queue — the other clean pending task is still served."""
    from flashruntime.leases import LeaseManager
    from flashruntime.scheduler import IsolationAwarePlacement

    mgr = LeaseManager()
    mgr.add_task(_task_raw("t-poison", "sandboxed"))  # head-of-queue, poisoned
    mgr.add_task(_task("t-clean"))  # standard ⇒ eligible everywhere
    policy = IsolationAwarePlacement()

    lease = mgr.claim("n1", policy=policy, node={"node_id": "n1", "sandbox_capable": True})
    assert lease is not None
    assert lease.task_id == "t-clean"


def test_unknown_tier_string_requires_capability():
    """An unknown/typo'd tier (e.g. 'Sandboxed') must be treated like the
    sandboxed gate, not silently downgraded to run-anywhere."""
    from flashruntime.scheduler import IsolationAwarePlacement

    policy = IsolationAwarePlacement()
    capable = {"node_id": "n1", "sandbox_capable": True}
    incapable = {"node_id": "n2", "sandbox_capable": False}
    task = _task("t", tier="Sandboxed")  # capitalised typo ⇒ NOT run-anywhere
    assert policy.eligible(task, capable)
    assert not policy.eligible(task, incapable)


def test_claim_with_foreign_spec_policy_returns_none():
    """A custom policy returning a spec that is not among the pending records
    must yield a clean None, not a StopIteration crash."""
    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import TaskSpec

    class ForeignSpecPolicy:
        def choose(self, pending, node):
            return TaskSpec(task_id="ghost", job_id="j1", commit_key="j1/ghost")

    mgr = LeaseManager()
    mgr.add_task(_task("real"))
    assert mgr.claim("n1", policy=ForeignSpecPolicy(), node={"node_id": "n1"}) is None


# -- argv gate: volunteer compute pool (untrusted argv execution) -------------


def _argv_task():
    from flashruntime.protocol.v1alpha1 import TaskSpec

    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload={"argv": ["python", "train.py"],
                 "isolation": {"tier": "sandboxed", "allowFallback": False}},
    )


@pytest.mark.parametrize("value", [None, False, "true", 1, "yes"])
def test_argv_task_ineligible_without_genuine_true(value):
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "sandbox_capable": True}
    if value is not None:
        node["argv_capable"] = value
    assert IsolationAwarePlacement().eligible(_argv_task(), node) is False


def test_argv_task_eligible_on_capable_node():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "sandbox_capable": True, "argv_capable": True}
    assert IsolationAwarePlacement().eligible(_argv_task(), node) is True


def test_allow_fallback_cannot_bypass_the_argv_gate():
    """allowFallback waives the sandbox capability requirement. It must not
    also waive argv capability, or a submitter could land arbitrary argv on
    a node with no argv runner at all."""
    from flashruntime.scheduler import IsolationAwarePlacement

    task = _argv_task()
    task.payload["isolation"]["allowFallback"] = True
    node = {"node_id": "n1", "sandbox_capable": False}
    assert IsolationAwarePlacement().eligible(task, node) is False


def test_non_argv_tasks_are_unaffected():
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.scheduler import IsolationAwarePlacement

    task = TaskSpec(task_id="t", job_id="j", commit_key="j/t/m.json",
                    payload={"module": "flashml_workloads.sklearn_trial"})
    assert IsolationAwarePlacement().eligible(task, {"node_id": "n1"}) is True


# -- F1: module gate — fail-OPEN, the mirror of the argv gate above -----------
#
# An argv-only node poisons every module job in the pool otherwise: it
# claims a module task, ArgvDockerRunner rejects the payload, the attempt
# fails, and the task requeues within poll_seconds until maxTaskAttempts is
# exhausted — job FAILED. The gate must exclude only a node that EXPLICITLY
# says module_capable: false; absence/None must still mean capable, or every
# already-deployed node (whose registration predates this field) silently
# loses all module work the moment it ships.


def _module_task():
    from flashruntime.protocol.v1alpha1 import TaskSpec

    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload={"module": "flashml_workloads.sklearn_trial", "params": {}},
    )


def test_argv_only_node_ineligible_for_module_task():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "argv_capable": True, "module_capable": False}
    assert IsolationAwarePlacement().eligible(_module_task(), node) is False


def test_subprocess_or_docker_node_eligible_for_module_task():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "argv_capable": False, "module_capable": True}
    assert IsolationAwarePlacement().eligible(_module_task(), node) is True


def test_node_missing_module_capable_field_is_eligible_for_module_task():
    """Old agent: registration predates module_capable entirely. Fail-open
    on availability means it must keep receiving module work."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1"}  # no module_capable key at all
    assert IsolationAwarePlacement().eligible(_module_task(), node) is True


def test_node_with_module_capable_none_is_eligible_for_module_task():
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "module_capable": None}
    assert IsolationAwarePlacement().eligible(_module_task(), node) is True


def test_argv_gate_fail_closed_behaviour_is_unchanged_by_the_module_gate():
    """The module gate must not loosen the pre-existing argv gate: a node
    with module_capable True but argv_capable absent is still ineligible
    for argv work."""
    from flashruntime.scheduler import IsolationAwarePlacement

    node = {"node_id": "n1", "sandbox_capable": True, "module_capable": True}
    assert IsolationAwarePlacement().eligible(_argv_task(), node) is False
