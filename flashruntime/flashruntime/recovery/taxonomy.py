"""Deterministic failure classification from raw signals.

Turns what was *observed* (exit codes, heartbeat loss, NCCL messages, XID
events, storage errors) into one typed `FailureClass`. Rules are
precedence-ordered — the first matching class wins — and the order encodes
root-cause priority: systemic evidence (correlated incidents, control-plane
loss) beats node evidence beats process evidence, because a worker crash
*during* a correlated incident is the incident, not the worker.

No inference beyond the table. Ambiguity resolves to conservative classes
(UNKNOWN classifies, policy decides what UNKNOWN deserves) rather than
optimistic ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from flashruntime.protocol.v1alpha1 import FailureClass

# Substrings that identify communication-layer failures in worker logs.
# [assumption: covers the common NCCL/Gloo surfaces; extend from ledger data]
_COMM_MARKERS = (
    "nccl",
    "rccl",
    "gloo",
    "rendezvous",
    "allreduce",
    "all_reduce",
    "process group",
)
_STORAGE_MARKERS = ("s3", "oss", "minio", "object store", "storage timeout", "slowdown")


@dataclass(frozen=True)
class FailureSignals:
    """The raw evidence available when something broke. All fields optional —
    classification works from whatever was actually observed."""

    exit_code: int | None = None
    exit_deterministic: bool = False  # same error at the same step on retry
    heartbeat_lost: bool = False
    node_unreachable: bool = False
    xid_event: bool = False  # NVIDIA XID / ECC / NVML health failure
    driver_fault: bool = False
    hash_mismatch: bool = False
    validation_failed: bool = False
    preemption_notice: bool = False
    storage_errors: int = 0
    control_plane_unreachable: bool = False
    concurrent_failures: int = 1  # failures in the correlation window, incl. this one
    message: str = ""

    def _msg(self) -> str:
        return self.message.lower()

    def has_comm_marker(self) -> bool:
        return any(m in self._msg() for m in _COMM_MARKERS)

    def has_storage_marker(self) -> bool:
        return any(m in self._msg() for m in _STORAGE_MARKERS)


# How many failures inside the correlation window turn individual failures
# into one systemic incident. [assumption: recalibrate from ledger data]
CORRELATED_THRESHOLD = 3


def classify(s: FailureSignals) -> FailureClass:
    """Precedence-ordered classification. Systemic > node > process > app."""
    if s.concurrent_failures >= CORRELATED_THRESHOLD:
        return FailureClass.CORRELATED_INCIDENT
    if s.control_plane_unreachable:
        return FailureClass.CONTROL_PLANE_FAILURE
    if s.preemption_notice:
        return FailureClass.PREEMPTION
    if s.xid_event or s.driver_fault:
        return FailureClass.ACCELERATOR_FAILURE
    if s.heartbeat_lost or s.node_unreachable:
        return FailureClass.NODE_LOSS
    if s.has_comm_marker():
        return FailureClass.COMMUNICATION_ERROR
    if s.storage_errors > 0 or s.has_storage_marker():
        return FailureClass.STORAGE_TIMEOUT
    if s.hash_mismatch or s.validation_failed:
        return FailureClass.ARTIFACT_CORRUPTION
    if s.exit_code is not None and s.exit_code != 0:
        if s.exit_deterministic:
            return FailureClass.APPLICATION_ERROR
        return FailureClass.WORKER_CRASH
    return FailureClass.UNKNOWN
