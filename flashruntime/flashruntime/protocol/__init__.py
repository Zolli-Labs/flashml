"""Versioned public schemas shared by FlashNode and FlashML Cloud.

Current version: v1alpha1 (`flashruntime.protocol.v1alpha1`) — job
specification, job state, event vocabulary, artifact records, and node
registration/heartbeat messages. Every schema carries an explicit version
field; security-relevant fields fail closed on unknown values.

Planning contract: `flashruntime.protocol.plan_v1alpha1` — PlanRequest
(workload + resources + objective), StrategyPlan, PlanReport with candidate
verdicts and explanations.

Future (still scaffolds elsewhere): lease and attempt payloads, checkpoint
manifests.
"""

from flashruntime.protocol import plan_v1alpha1, v1alpha1

__all__ = ["v1alpha1", "plan_v1alpha1"]
