"""The one fail-closed placement rule the isolation contract requires:
sandboxed tasks go only to sandbox_capable nodes (AGENTS.md rule 3)."""

from __future__ import annotations


def _task(task_id: str, tier: str | None = None, allow_fallback: bool = False):
    from flashruntime.protocol.v1alpha1 import TaskSpec

    payload = {}
    if tier is not None:
        payload["isolation"] = {"tier": tier, "allowFallback": allow_fallback}
    return TaskSpec(task_id=task_id, job_id="j1", commit_key=f"j1/{task_id}", payload=payload)


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
