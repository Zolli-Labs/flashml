"""Failure taxonomy and the deterministic recovery state machine.

Failure classes (application error, worker crash, node loss, accelerator
fault, network degradation, artifact corruption, preemption, correlated
incident) map to typed, logged recovery actions: retry, reassign, replace
node, restart group, move pool, wait, or stop automation and escalate.
"""
