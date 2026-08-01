"""Guard against DEFAULT_ALLOWED_MODULES drifting behind the coordinator's
own allowlist.

Background: flashnode enforces its *own* copy of the task-module allowlist
client-side (fail-closed, by design — see runner.py's module docstring).
The coordinator enforces a separate copy in
flashruntime/flashruntime/service/modea.py (`ALLOWED_TASK_MODULES`). Nothing
keeps the two in sync automatically: a workload can be added to the
coordinator's list, pass the coordinator's own tests, and still be refused
by every real agent, because the agent's list was never touched. That is
exactly what happened with `flashml_workloads.fedavg_worker` — each repo's
allowlist looked correct in isolation, and only a cross-repo e2e test
(flashruntime/../e2e/test_fedavg_loop.py) caught the mismatch.

Ideally this test would import `flashruntime.service.modea.ALLOWED_TASK_MODULES`
directly and compare against it, making the guard self-maintaining. Two
things rule that out here:

  1. AGENTS.md (hard rule 2) scopes flashnode's dependency on flashruntime to
     the versioned *protocol* package ("wire messages come from
     flashruntime.protocol"); `flashruntime.service` is the coordinator's
     internal implementation, not part of that contract, and pulling it into
     flashnode (even just in tests) blurs a boundary the docs draw on
     purpose.
  2. It is not even importable in flashnode's own test environment:
     `flashruntime.service.modea` pulls in FastAPI, which is a coordinator
     dependency flashnode does not install (`ModuleNotFoundError: No module
     named 'fastapi'`).

So this uses the documented fallback: an explicit mirror of
`ALLOWED_TASK_MODULES`, with `modea.py` named as the source of truth. When a
workload module is added there, this test fails until the same string is
added to flashnode's DEFAULT_ALLOWED_MODULES (and, in the other direction,
would fail if flashnode ever allowed a module the coordinator does not).
"""

from __future__ import annotations

from flashnode.executor.runner import DEFAULT_ALLOWED_MODULES

# Mirrors ALLOWED_TASK_MODULES in
# flashruntime/flashruntime/service/modea.py — that file is the source of
# truth. Update this set whenever that one changes.
COORDINATOR_ALLOWED_TASK_MODULES = frozenset(
    {
        "flashml_workloads.sklearn_trial",
        "flashml_workloads.kmeans_shard",
        "flashml_workloads.sgd_trainer",
        "flashml_workloads.fedavg_worker",
    }
)


def test_agent_allowlist_matches_coordinator_allowlist():
    """flashnode must accept every module the coordinator is willing to
    hand out — a mismatch means the coordinator can lease work that every
    real agent refuses client-side, burning all attempts and failing the
    job before training ever runs (see docstring above)."""
    missing_on_agent = COORDINATOR_ALLOWED_TASK_MODULES - DEFAULT_ALLOWED_MODULES
    assert not missing_on_agent, (
        f"coordinator allows {missing_on_agent} but flashnode's "
        "DEFAULT_ALLOWED_MODULES does not — every lease for this module "
        "will be refused client-side on real agents. Add it to "
        "DEFAULT_ALLOWED_MODULES in flashnode/executor/runner.py."
    )


def test_agent_allowlist_does_not_exceed_coordinator_allowlist():
    """The reverse drift: flashnode should not accept modules the
    coordinator itself would never hand out — that would silently widen
    what an agent is willing to execute beyond what the protocol intends."""
    extra_on_agent = DEFAULT_ALLOWED_MODULES - COORDINATOR_ALLOWED_TASK_MODULES
    assert not extra_on_agent, (
        f"flashnode's DEFAULT_ALLOWED_MODULES allows {extra_on_agent}, "
        "which the coordinator's ALLOWED_TASK_MODULES does not list — "
        "narrow flashnode's allowlist or confirm modea.py is genuinely "
        "out of date."
    )
