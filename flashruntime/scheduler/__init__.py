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

__all__ = ["PlacementPolicy", "FifoPlacement", "IsolationAwarePlacement"]

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


class IsolationAwarePlacement(PlacementPolicy):
    """FIFO plus the one fail-closed capability gate the isolation contract
    requires: a task whose payload demands sandboxed execution may only go
    to a node advertising `sandbox_capable` — an ABSENT capability counts as
    NOT capable (security-relevant fields fail closed, AGENTS.md rule 3).

    The gate requires *true booleans*; type-confused values fail closed:

    - A node is capable only when `sandbox_capable is True`. A truthy
      stand-in (the JSON string ``"false"``, ``1``, ``"yes"``) does NOT
      count as capable.
    - The task's own `allowFallback` waives the requirement only when it is
      exactly `True`; any other value (including the string ``"false"``)
      does not waive.
    - The isolation payload must be a mapping. If it is present but not a
      dict (e.g. the bare string ``"sandboxed"``), the task is ineligible
      everywhere — fail closed without crashing the predicate.
    - Only tiers `None` / ``""`` / ``"standard"`` run anywhere. ANY other
      tier value (an unknown or mistyped literal like ``"Sandboxed"``) is
      treated like ``"sandboxed"`` and requires capability — no silent
      downgrade to unsandboxed placement.

    A second, independent gate applies to tasks carrying an `argv` payload
    (arbitrary user command lines): the claiming node must advertise
    `argv_capable is True`. This is checked BEFORE the isolation block's
    `allowFallback` waiver below, and the waiver does not apply to it — a
    submitter setting `allowFallback: true` waives the sandbox-tier
    requirement only, never the argv-runner requirement, or arbitrary argv
    could land on a node with no argv runner at all.

    A third gate applies to tasks carrying a `module` payload (the
    "python -m <allowlisted module>" tier): the claiming node must not be
    `module_capable is False`. Its polarity is deliberately the MIRROR of
    the argv gate above, not a copy of it:

    - `argv_capable` is fail-closed (`is True` required) because argv is a
      *safety* boundary — placing argv on a node with no argv runner is a
      security failure (ArgvDockerRunner's own payload check would refuse
      it, but the whole point of placement is to not burn attempts getting
      there).
    - `module_capable` is fail-OPEN (`is False` excludes) because it is an
      *availability* concern only — a module task misplaced on an
      argv-only node wastes retry attempts, it never escapes a sandbox.
      Defaulting to capable means an already-deployed node whose
      registration predates this field (module_capable absent ⇒ None) —
      or any node that simply never opted into an argv-only runner — keeps
      receiving module work. Only a node that explicitly advertises
      `module_capable: false` (an argv-only volunteer) is excluded.

    Do NOT "harmonize" these two gates onto the same polarity — copying
    the argv gate's `is True` pattern onto module_capable would silently
    cut every already-deployed node off from module work the moment this
    field shipped, without anyone having opted out of anything.

    Everything genuinely standard keeps the fail-open placement default."""

    def eligible(self, task: TaskSpec, node: NodeView) -> bool:
        # Checked before the allowFallback waiver below: the waiver relaxes
        # the sandbox-tier requirement, and must never be readable as
        # permission to run argv on a node with no argv runner.
        if "argv" in task.payload and node.get("argv_capable") is not True:
            return False
        # Availability gate, mirrored polarity from the argv gate above —
        # see the class docstring. An argv-only volunteer poisons every
        # module job in the pool otherwise: it claims, ArgvDockerRunner
        # rejects the payload, the attempt fails, and the task requeues
        # into the same node's path until attempts are exhausted.
        if "module" in task.payload and node.get("module_capable") is False:
            return False
        isolation = task.payload.get("isolation")
        if isolation is None:
            return True  # no isolation payload ⇒ standard, runs anywhere
        if not isinstance(isolation, dict):
            return False  # type-confused payload ⇒ fail closed, no crash
        if isolation.get("tier") in (None, "", "standard"):
            return True  # only the known non-isolated tiers run anywhere
        if isolation.get("allowFallback") is True:
            return True  # explicit waiver — genuine boolean only
        return node.get("sandbox_capable") is True  # capable ⇒ genuine boolean only
