"""Failure taxonomy and the deterministic recovery policy.

Two pure functions and a versioned table:

    from flashruntime.recovery import FailureSignals, classify, decide

    failure = classify(FailureSignals(heartbeat_lost=True))   # NODE_LOSS
    decision = decide(failure, mode="independent_tasks")
    # → RETRY_TASK, cordon_node=True, policy_version=0.1.0, reason=...

`classify` turns raw observed signals (exit codes, heartbeat loss, NCCL
messages, XID events) into one typed FailureClass with precedence-ordered
rules. `decide` maps (FailureClass, execution mode) to one typed, logged
RecoveryDecision. No scoring, no learning, no agent: same failure + same
policy version ⇒ same action.
"""

from flashruntime.recovery.policy import POLICY_VERSION, decide
from flashruntime.recovery.taxonomy import CORRELATED_THRESHOLD, FailureSignals, classify

__all__ = ["FailureSignals", "classify", "decide", "POLICY_VERSION", "CORRELATED_THRESHOLD"]
