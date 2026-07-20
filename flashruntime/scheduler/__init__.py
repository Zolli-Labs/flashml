"""Placement policy: which task should a claiming node receive?

Today's live behavior is FIFO over PENDING tasks (`LeaseStore.next_pending`
insertion order) — honest and correct while tasks are CPU-sized and
homogeneous. `FifoPlacement` encodes exactly that behavior as the first
concrete policy, so wiring this interface into the claim path changes
nothing until a richer policy is chosen deliberately.

This package becomes load-bearing when payloads carry resource
requirements (vram_gb, min_cpus) and claims must filter by node
capability — research item R9 in `HANDBOOK.md` §7. The master report's §9
discipline applies: capability ("can it run this?"), reliability ("will it
finish?") and trust ("may this data go there?") stay SEPARATE assessments;
this interface handles capability filtering + preference ordering only.
Reliability scoring waits for ledger volume; trust tiers are cloud policy.

Integration point (when the time comes): `LeaseManager.claim` consults the
policy instead of calling `next_pending` directly — the store keeps
returning candidates in insertion order; the policy filters and picks.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from flashruntime.protocol.v1alpha1 import TaskSpec

__all__ = ["PlacementPolicy", "FifoPlacement"]

#: A node as the policy sees it: the registry's view dict
#: ({"node_id", "capabilities": {...}, ...}). Kept as a mapping (not a
#: model) until the capability schema for placement stabilizes with R9.
NodeView = dict[str, Any]


class PlacementPolicy(ABC):
    """Filter + order candidate tasks for one claiming node.

    Contracts:
    - `eligible` is a pure predicate — no I/O, no side effects; called
      once per (task, node) per claim, so it must be cheap.
    - `score` orders eligible tasks (higher = better for THIS node);
      ties break by queue order (fairness/FIFO stays the default
      tiebreak so starvation cannot be introduced accidentally).
    - `choose` is a template method most policies should NOT override —
      override `eligible`/`score` and keep the selection semantics shared.
    - A policy must never *create* work or mutate tasks; it only selects.
    """

    @abstractmethod
    def eligible(self, task: TaskSpec, node: NodeView) -> bool:
        """May this node run this task at all? (capability gate — e.g.
        payload wants vram_gb the node lacks → False). Unknown/absent
        requirements must default to True: fail-open on *placement*,
        because the executor's allowlists still fail-closed on *safety*."""

    def score(self, task: TaskSpec, node: NodeView) -> float:
        """Preference among eligible tasks for this node (higher wins).
        Default 0.0 for all ⇒ pure FIFO. Examples later: prefer tasks
        whose input artifacts this node already cached; prefer short
        tasks for soon-to-drain nodes."""
        return 0.0

    def choose(self, pending: list[TaskSpec], node: NodeView) -> TaskSpec | None:
        """Select the task to lease to `node` from queue-ordered `pending`.

        Semantics: filter by `eligible`, take the max `score`, break ties
        by earliest queue position. Returns None when nothing is eligible
        (the claim endpoint answers 204, exactly like an empty queue).
        """
        best: TaskSpec | None = None
        best_score = float("-inf")
        for task in pending:  # queue order ⇒ first max wins ties
            if not self.eligible(task, node):
                continue
            s = self.score(task, node)
            if s > best_score:
                best, best_score = task, s
        return best


class FifoPlacement(PlacementPolicy):
    """The current system, as a policy: everything eligible, no
    preference, first-come-first-served. Wiring this in is a pure
    refactor — behavior is bit-identical to `next_pending`."""

    def eligible(self, task: TaskSpec, node: NodeView) -> bool:
        return True
