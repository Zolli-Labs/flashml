"""Recovery tests: classification precedence and the policy table's shape."""

from __future__ import annotations

import pytest

from flashruntime.protocol.v1alpha1 import FailureClass, RecoveryActionType
from flashruntime.recovery import POLICY_VERSION, FailureSignals, classify, decide


# -- classification ----------------------------------------------------------


def test_classification_precedence_systemic_beats_local():
    """Three concurrent failures are one incident — even when each also has
    node-local evidence."""
    s = FailureSignals(heartbeat_lost=True, exit_code=137, concurrent_failures=3)
    assert classify(s) == FailureClass.CORRELATED_INCIDENT


def test_nccl_message_classifies_as_communication_error():
    s = FailureSignals(exit_code=1, message="NCCL watchdog timeout on rank 3")
    assert classify(s) == FailureClass.COMMUNICATION_ERROR


def test_deterministic_exit_is_application_error_transient_is_crash():
    assert classify(FailureSignals(exit_code=1, exit_deterministic=True)) == FailureClass.APPLICATION_ERROR
    assert classify(FailureSignals(exit_code=137)) == FailureClass.WORKER_CRASH


def test_heartbeat_loss_is_node_loss_xid_is_accelerator():
    assert classify(FailureSignals(heartbeat_lost=True)) == FailureClass.NODE_LOSS
    assert classify(FailureSignals(xid_event=True, heartbeat_lost=True)) == FailureClass.ACCELERATOR_FAILURE


def test_nothing_observed_is_unknown():
    assert classify(FailureSignals()) == FailureClass.UNKNOWN


# -- policy ------------------------------------------------------------------


def test_same_failure_different_blast_radius_by_mode():
    """The reason the two execution modes exist: a crash costs one task in
    Mode A and the whole group in Mode B."""
    a = decide(FailureClass.WORKER_CRASH, "independent_tasks")
    b = decide(FailureClass.WORKER_CRASH, "coordinated_training")
    assert a.action == RecoveryActionType.RETRY_TASK and a.scope == "task"
    assert b.action == RecoveryActionType.RESTART_GROUP and b.scope == "group"
    assert b.needs_checkpoint and not a.needs_checkpoint


def test_application_errors_are_never_retried():
    for mode in ("independent_tasks", "coordinated_training"):
        assert decide(FailureClass.APPLICATION_ERROR, mode).action == RecoveryActionType.FAIL_JOB


def test_node_loss_cordons_the_node():
    assert decide(FailureClass.NODE_LOSS, "independent_tasks").cordon_node


def test_correlated_incident_freezes_automation():
    d = decide(FailureClass.CORRELATED_INCIDENT, "coordinated_training")
    assert d.action == RecoveryActionType.FREEZE_AUTOMATION and d.scope == "pool"


def test_storage_outage_pauses_instead_of_killing():
    assert decide(FailureClass.STORAGE_TIMEOUT, "independent_tasks").action == RecoveryActionType.PAUSE_JOB


def test_every_failure_class_has_a_decision_in_both_modes():
    """The table must be total: an unclassifiable situation is UNKNOWN, but a
    classified failure without a policy entry is a bug."""
    for failure in FailureClass:
        for mode in ("independent_tasks", "coordinated_training"):
            d = decide(failure, mode)
            assert d.policy_version == POLICY_VERSION
            assert d.reason  # every decision explains itself


def test_decisions_are_versioned_and_deterministic():
    d1 = decide(FailureClass.NODE_LOSS, "coordinated_training", evidence={"node": "n1"})
    d2 = decide(FailureClass.NODE_LOSS, "coordinated_training", evidence={"node": "n1"})
    assert d1.action == d2.action == RecoveryActionType.REPLACE_NODE
    assert d1.policy_version == d2.policy_version


def test_unknown_mode_is_refused():
    with pytest.raises(ValueError):
        decide(FailureClass.NODE_LOSS, "definitely_not_a_mode")
