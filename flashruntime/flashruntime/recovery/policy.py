"""The versioned, deterministic recovery policy.

FailureClass × execution mode → one typed `RecoveryDecision`. The table *is*
the policy: no scoring, no learning, no agent — same failure + same policy
version ⇒ same action, always (ADR-0003 / master report guardrail: recovery
actions are typed, authorized, and logged).

The two execution modes matter because the same failure has different blast
radii: a worker crash in Mode A costs one task retry; in Mode B it stops the
whole group (NCCL state is not repairable in place — whole-group restart
from the latest valid checkpoint is the honest v1 promise).

Encoded rules worth naming:
- Deterministic application errors are never retried — fail fast, tell the
  user. Burning capacity on a bug is the most expensive kind of "recovery".
- Storage outages pause rather than kill: compute waiting on dead storage
  is wasted, but the job's state is intact — protect it and stop the burn.
- Correlated incidents freeze automation entirely. Retry storms during a
  systemic incident are how orchestrators destroy trust; the policy's most
  important action is knowing when to stop acting.
"""

from __future__ import annotations

from flashruntime.protocol.v1alpha1 import (
    FailureClass,
    RecoveryActionType,
    RecoveryDecision,
)

POLICY_VERSION = "0.1.0"

Mode = str  # "independent_tasks" | "coordinated_training"

# (action, scope, cordon_node, needs_checkpoint, reason)
_TABLE: dict[tuple[FailureClass, Mode], tuple[RecoveryActionType, str, bool, bool, str]] = {
    # -- application / data: not infrastructure's fault ---------------------
    (FailureClass.APPLICATION_ERROR, "independent_tasks"): (
        RecoveryActionType.FAIL_JOB, "job", False, False,
        "deterministic application error — retrying burns money on a bug",
    ),
    (FailureClass.APPLICATION_ERROR, "coordinated_training"): (
        RecoveryActionType.FAIL_JOB, "job", False, False,
        "deterministic application error — retrying burns money on a bug",
    ),
    (FailureClass.DATA_ERROR, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "data error on one shard — retry once elsewhere; quarantine on repeat",
    ),
    (FailureClass.DATA_ERROR, "coordinated_training"): (
        RecoveryActionType.FAIL_JOB, "job", False, False,
        "data error in coordinated training — resume would repeat it",
    ),
    # -- process / node -----------------------------------------------------
    (FailureClass.WORKER_CRASH, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "non-deterministic worker crash — lease expires, task requeues",
    ),
    (FailureClass.WORKER_CRASH, "coordinated_training"): (
        RecoveryActionType.RESTART_GROUP, "group", False, True,
        "a lost rank stops the group — restart all workers from latest valid checkpoint",
    ),
    (FailureClass.NODE_LOSS, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", True, False,
        "node gone — cordon it, expire its leases, requeue its tasks",
    ),
    (FailureClass.NODE_LOSS, "coordinated_training"): (
        RecoveryActionType.REPLACE_NODE, "group", True, True,
        "node gone — acquire replacement capacity, then group restart from checkpoint",
    ),
    (FailureClass.ACCELERATOR_FAILURE, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "node", True, False,
        "GPU/driver fault — cordon the accelerator's node, retry elsewhere",
    ),
    (FailureClass.ACCELERATOR_FAILURE, "coordinated_training"): (
        RecoveryActionType.REPLACE_NODE, "group", True, True,
        "GPU/driver fault — replace the node, group restart from checkpoint",
    ),
    # -- communication / network -------------------------------------------
    (FailureClass.COMMUNICATION_ERROR, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "transport error on an independent task — plain retry",
    ),
    (FailureClass.COMMUNICATION_ERROR, "coordinated_training"): (
        RecoveryActionType.RESTART_GROUP, "group", False, True,
        "NCCL-class error — collective state is not repairable in place; whole-group restart",
    ),
    (FailureClass.NETWORK_DEGRADATION, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "degraded link — move the task; the node's reliability score absorbs the signal",
    ),
    (FailureClass.NETWORK_DEGRADATION, "coordinated_training"): (
        RecoveryActionType.RESTART_GROUP, "group", False, True,
        "sustained degradation starves the collective — restart, prefer a better pool",
    ),
    # -- dependencies -------------------------------------------------------
    (FailureClass.STORAGE_TIMEOUT, "independent_tasks"): (
        RecoveryActionType.PAUSE_JOB, "job", False, False,
        "storage outage — pause instead of burning compute against a dead dependency",
    ),
    (FailureClass.STORAGE_TIMEOUT, "coordinated_training"): (
        RecoveryActionType.PAUSE_JOB, "job", False, False,
        "storage outage — pause instead of burning compute against a dead dependency",
    ),
    (FailureClass.ARTIFACT_CORRUPTION, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "output failed validation — reject commit, retry on a different node, score down producer",
    ),
    (FailureClass.ARTIFACT_CORRUPTION, "coordinated_training"): (
        RecoveryActionType.RESTART_GROUP, "group", False, True,
        "corrupt checkpoint/artifact — restart from the previous *valid* manifest",
    ),
    (FailureClass.PREEMPTION, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "priced-in preemption — requeue; no penalty to anyone",
    ),
    (FailureClass.PREEMPTION, "coordinated_training"): (
        RecoveryActionType.REPLACE_NODE, "group", False, True,
        "spot capacity reclaimed — acquire replacement, restart from checkpoint",
    ),
    # -- systemic: stop acting ---------------------------------------------
    (FailureClass.CORRELATED_INCIDENT, "independent_tasks"): (
        RecoveryActionType.FREEZE_AUTOMATION, "pool", False, False,
        "multiple simultaneous failures — no retry storms; preserve state and escalate",
    ),
    (FailureClass.CORRELATED_INCIDENT, "coordinated_training"): (
        RecoveryActionType.FREEZE_AUTOMATION, "pool", False, False,
        "multiple simultaneous failures — no retry storms; preserve state and escalate",
    ),
    (FailureClass.CONTROL_PLANE_FAILURE, "independent_tasks"): (
        RecoveryActionType.PAUSE_JOB, "job", False, False,
        "coordinator unreachable — workers finish current leases; no new grants",
    ),
    (FailureClass.CONTROL_PLANE_FAILURE, "coordinated_training"): (
        RecoveryActionType.PAUSE_JOB, "job", False, False,
        "coordinator unreachable — preserve state; no automated decisions without a ledger",
    ),
    (FailureClass.UNKNOWN, "independent_tasks"): (
        RecoveryActionType.RETRY_TASK, "task", False, False,
        "unclassified failure — one conservative retry; repeats escalate via attempt limits",
    ),
    (FailureClass.UNKNOWN, "coordinated_training"): (
        RecoveryActionType.PAUSE_JOB, "job", False, True,
        "unclassified failure in coordinated training — pause for inspection over blind restart",
    ),
}


def decide(failure: FailureClass, mode: Mode, evidence: dict | None = None) -> RecoveryDecision:
    """Look up the typed action for a classified failure in a given mode.

    Pure table lookup — deliberately no other inputs. Anything that should
    influence the action (repeat counts, correlation) must appear upstream
    as a different FailureClass, keeping every decision explainable by
    (policy_version, failure_class, mode) alone.
    """
    try:
        action, scope, cordon, needs_ckpt, reason = _TABLE[(failure, mode)]
    except KeyError as exc:
        raise ValueError(f"no policy entry for {failure.value} in mode {mode!r}") from exc
    return RecoveryDecision(
        policy_version=POLICY_VERSION,
        failure_class=failure,
        action=action,
        scope=scope,  # type: ignore[arg-type]
        cordon_node=cordon,
        needs_checkpoint=needs_ckpt,
        reason=reason,
        evidence=evidence or {},
    )
