"""Workload recipes: everything one workload type needs, behind one interface.

Today each workload is wired by hand in three places: an expansion branch
in `service/modea.py`, a module allowlist entry (both ends), and a task
module in `flashml_workloads/`. A `WorkloadRecipe` bundles those into one
registrable object so adding a workload becomes: write the task module,
write the recipe, register it. The existing hand-rolled expansions
(`hyperparameter_search`, `sharded_kmeans`) are the reference behavior and
should migrate here without any wire-visible change.

A recipe spans three moments in a task's life:
  submit-time  — `validate_params` + `expand` (coordinator)
  run-time     — `task_module` executed by the agent (the §2.2 contract:
                 `python -m <module> --spec spec.json --out OUTDIR`)
  commit-time  — `validate_output` (coordinator; runs BEFORE acceptance,
                 alongside the sha256 check) and, for map/reduce shapes,
                 `reduce` (driver side)

Hard rules inherited from the system: the task module must appear on BOTH
allowlists (coordinator + agent) — a recipe self-declares it, and the
registry is what the allowlists will be generated from once migration
lands. Hugging Face code lives here and only here (four-axes rule): HF is
workload, never a backend.

Status: interface complete (final surface); first concrete recipe is
HF Trainer + PEFT LoRA (SPRINT_PLAN Days 8–10, design in ADR-0004-to-be).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from flashruntime.protocol.v1alpha1 import JobSpec, TaskSpec

__all__ = ["WorkloadRecipe", "register_recipe", "recipe_for"]


class WorkloadRecipe(ABC):
    """One workload type, end to end.

    Implementations must be *pure coordination logic*: no framework
    imports at module import time (import torch/transformers inside the
    task module, which runs on the agent — never in the coordinator's
    process).
    """

    #: The `JobSpec.spec.workload.type` value this recipe owns
    #: (e.g. "hyperparameter_search", "sharded_kmeans", "lora_finetune").
    kind: ClassVar[str]

    #: The task module executed on agents — the value placed in every
    #: payload's "module" field, and the entry both allowlists must carry.
    task_module: ClassVar[str]

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        """Submit-time parameter validation.

        Returns every problem found (empty list = valid); the service
        turns a non-empty list into one 422 carrying all reasons — the
        planner's surface-everything-at-once UX, applied to submission.
        Default: accept anything (expand() remains the backstop).
        """
        return []

    @abstractmethod
    def expand(self, job_id: str, spec: JobSpec) -> list[TaskSpec]:
        """Turn a JobSpec into independent TaskSpecs (Mode A expansion).

        Requirements (mirror the existing expansions exactly):
        - deterministic task_ids (`trial-000`-style) — retries and
          dashboards depend on stable names;
        - every payload carries: module, params, inputs (artifact:// URIs),
          output_prefix, task_id, image, and `checkpoint` when the recipe
          is checkpointable;
        - commit_key = the metrics.json path under output_prefix (the
          idempotency + validation anchor);
        - honor spec.spec.retryPolicy.maxTaskAttempts and the workload's
          lease_seconds parameter.
        Raises ValueError (→ 422) when the spec cannot be expanded.
        """

    @abstractmethod
    def validate_output(self, metrics: dict[str, Any]) -> None:
        """Commit-time semantic validation of a task's metrics.json.

        Runs on the coordinator after the sha256 check passes and before
        the commit is accepted: raise (ValueError with a reason) to fail
        the attempt — the task requeues, exactly like a hash mismatch.
        This is the hook that catches \"the file is intact but the result
        is nonsense\" (NaN loss, missing fields, accuracy > 1.0).
        """

    def reduce(self, outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Combine per-task outputs into a job-level result (map/reduce
        workloads: K-means partials → new centroids; HPO trials → best
        config). Return None for workloads with no reduce step (default).
        Runs driver/coordinator-side on already-validated outputs."""
        return None

    def checkpointable(self) -> bool:
        """Whether tasks of this recipe write resumable checkpoints (turns
        on the agent's checkpoint relay via the payload's `checkpoint`
        key). Default False: stateless tasks retry from scratch, which is
        cheaper than checkpointing for short work."""
        return False

    def plan_hints(self) -> dict[str, Any]:
        """Optional coupling to the planner: static facts the estimators
        may use (e.g. {'mode': 'independent_tasks'}). Keep it data-only —
        the planner never calls recipe *code* (determinism + no-framework
        rules)."""
        return {}


_REGISTRY: dict[str, WorkloadRecipe] = {}


def register_recipe(recipe: WorkloadRecipe) -> None:
    """Register (or replace) the recipe for its `kind`. At service startup
    the registered kinds become the expansion dispatch table, and the set
    of `task_module`s becomes the coordinator-side allowlist — one source
    of truth instead of today's two hand-maintained sets."""
    _REGISTRY[recipe.kind] = recipe


def recipe_for(kind: str) -> WorkloadRecipe:
    """Lookup by workload type; LookupError → the service's 422 listing
    the supported kinds (same UX as the current ExpansionError)."""
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise LookupError(
            f"no recipe for workload type {kind!r} (registered: {sorted(_REGISTRY) or 'none'})"
        )
