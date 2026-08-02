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

    A fourth gate applies to tasks whose payload lists `local_inputs` (data
    the host holds and never uploads): the claiming node must advertise
    EVERY one of those names in its `local_datasets` capability. It shares
    the argv gate's fail-closed polarity, not the module gate's, because
    the host's data is the thing being protected — an absent, `None`, or
    wrongly-typed capability counts as NOT capable:

    - The capability must be a genuine *list* of names. A bare string does
      not count, even when it looks right: ``"patients" in "patients"`` is
      True in Python, so accepting a string would let a node advertise
      every dataset whose name is a substring of anything it said. That is
      the same class of type confusion `sandbox_capable is True` avoids
      for booleans.
    - The requirement itself must be a list. A `local_inputs` payload that
      is present but not a list (e.g. the bare string ``"patients"``) makes
      the task ineligible everywhere — fail closed without crashing the
      predicate, exactly as a non-dict isolation payload does.
    - An empty `local_inputs` requires nothing and so runs anywhere, like
      tier ``"standard"``.

    The `allowFallback` waiver does NOT apply here, for the same reason it
    does not waive the argv gate: it waives the sandbox *tier* requirement
    and nothing else. A waiver is the submitter's statement about their own
    isolation posture; the local-data gate protects the HOST's data, which
    the submitter has no standing to waive. A task placed on a node that
    never advertised the dataset would fail on the agent anyway (the label
    is not in its map) — failing closed here makes it fail before anything
    touches the data, and before an attempt is burned.

    A fifth gate applies to tasks whose payload asks for `gpus: N`: the
    claiming node's `capabilities.gpus` must be a list of at least N entries.
    It takes the argv/local-data polarity — **fail closed** — and deliberately
    NOT the module gate's fail-open one, even though "the node cannot run
    this" sounds like the same availability concern:

    - A misplaced module task wastes retry attempts and nothing else. A CUDA
      job on a CPU-only box does not politely fail and requeue. It either
      crashes on `torch.cuda.is_available()` or, worse, silently falls back
      to the CPU and runs two orders of magnitude slower while reporting
      success. The second outcome is not a failure anything here can detect;
      it is a bill.
    - The capability must be a genuine *list*, because its LENGTH is the
      whole matching rule in v1. Absent, `None`, a bare string, a dict, or a
      bare `int` count as NO GPUs. The bare `int` deserves naming: `1` reads
      like "one GPU" and is exactly what a hand-written node view would put
      there, but accepting it would mean a second, looser matching rule
      beside the one `NodeCapabilities.gpus` actually feeds.
    - `capabilities` itself may be absent or type-confused; that is read as
      no GPUs rather than allowed to raise. `(node.get("capabilities") or
      {}).get(...)` is NOT sufficient for this — a string capabilities value
      has no `.get` and would crash the predicate.
    - The requirement itself must be a non-negative `int`. Anything else
      (`"1"`, `-1`, `1.5`, a list) makes the task ineligible everywhere
      rather than crashing, exactly as a non-list `local_inputs` does.
      **`bool` is a subclass of `int`**, so `True` is excluded explicitly: a
      `gpus: true` typo must not silently mean "one GPU" and place real work.
    - `gpus: 0` requires nothing and runs anywhere, exactly like tier
      `standard` and an empty `local_inputs`.

    The gate is ONE-DIRECTIONAL. A node with GPUs still receives CPU work;
    reserving GPU hosts for GPU jobs is a scheduling optimisation and a
    separate decision, and making it a gate here would idle the scarcest
    hardware on the network. `allowFallback` does not waive this gate either
    — hardware either exists on a host or it does not, and the submitter's
    isolation posture has nothing to say about it.

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
        # Fail-closed like the argv gate, and checked before the allowFallback
        # waiver below for the same reason: the waiver covers the sandbox tier
        # only, and a submitter cannot waive their way onto a host's data.
        local_inputs = task.payload.get("local_inputs")
        if local_inputs is not None:
            if not isinstance(local_inputs, list):
                return False  # type-confused requirement ⇒ fail closed, no crash
            advertised = node.get("local_datasets")
            if local_inputs and not isinstance(advertised, list):
                return False  # absent/None/type-confused capability ⇒ not capable
            if any(name not in advertised for name in local_inputs):
                return False
        # Fail-closed like the argv and local-data gates, and checked before
        # the allowFallback waiver below for the same reason: a submitter
        # cannot waive their way onto hardware a host does not have.
        required_gpus = task.payload.get("gpus")
        if required_gpus is not None:
            # `bool` is a subclass of `int`: without the explicit exclusion,
            # a `gpus: true` typo would read as "1 GPU" and place real work.
            if (
                not isinstance(required_gpus, int)
                or isinstance(required_gpus, bool)
                or required_gpus < 0
            ):
                return False  # type-confused requirement ⇒ fail closed, no crash
            if required_gpus > 0:
                capabilities = node.get("capabilities")
                # isinstance, not `or {}` — a string capabilities value has no
                # `.get` and must fail closed rather than crash the predicate.
                advertised = (
                    capabilities.get("gpus") if isinstance(capabilities, dict) else None
                )
                if not isinstance(advertised, list) or len(advertised) < required_gpus:
                    return False  # absent/short/type-confused ⇒ not capable
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
