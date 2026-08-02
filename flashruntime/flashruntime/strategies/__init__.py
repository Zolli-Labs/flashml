"""Strategy compilers: a frozen `StrategyPlan` → concrete launch configuration.

This is the seam that keeps the planner honest: the planner reasons about
strategies *by name and number* (never importing frameworks), and a
compiler here translates the chosen plan into the real thing — torchrun
argv + env for DDP/FSDP2, a DeepSpeed JSON config file for ZeRO families,
a lease-job JobSpec for `lease_tasks`. One compiler per strategy family;
the registry dispatches on `supports()`.

Design rules (ADR-0003, HANDBOOK §5):
- Compilers may import framework *constants* knowledge but must not
  require the framework installed to *compile* (emitting an argv string
  needs no torch). Anything that must run framework code belongs in the
  launcher or the task, not here.
- Compilation is pure and deterministic: same plan ⇒ same LaunchSpec.
  No environment inspection, no network. Preflight checks (is torchrun on
  PATH? is the image pullable?) belong to `Launcher.healthy()`.
- A compiler must refuse loudly (`CompileError` listing every problem)
  rather than emit a config it cannot stand behind — the same fail-closed
  stance as the executor allowlists.

Status: interface complete (final surface, by design review); concrete
compilers land per SPRINT_PLAN (torchrun/DDP first, with the LoRA recipe).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from pydantic import BaseModel, Field

from flashruntime.protocol.plan_v1alpha1 import StrategyPlan

__all__ = [
    "LaunchSpec",
    "StrategyCompiler",
    "CompileError",
    "register_compiler",
    "compiler_for",
]


class CompileError(Exception):
    """The plan cannot be compiled by this family — carries *all* reasons
    at once (like planner rejections: half the value is the explanation)."""


class LaunchSpec(BaseModel):
    """Everything a `Launcher` needs to start one worker group / job.

    Backend-neutral on purpose: the same model describes a torchrun
    process group, a DeepSpeed launch, or a Mode A lease job. Fields a
    given launcher does not need are simply ignored by it.
    """

    argv: list[str] = Field(description="Command to execute per worker (rank 0's view)")
    env: dict[str, str] = Field(
        default_factory=dict,
        description="Environment to merge over the launcher's base env "
        "(e.g. WORLD_SIZE, precision flags). Never secrets — credentials "
        "are the launcher's concern.",
    )
    files: dict[str, str] = Field(
        default_factory=dict,
        description="Config files to materialize into the attempt OUTPUT dir "
        "(the one exported as FLASHML_OUTPUT_DIR) before launch, name → "
        "content (e.g. 'ds_config.json' for DeepSpeed). Keeps compilation "
        "pure while supporting file-driven frameworks. Note: files land in "
        "the output dir, NOT the command's cwd — wiring cwd-reading "
        "frameworks to find them is a flagged follow-up to settle before any "
        "files-emitting compiler ships.",
    )
    world_size: int = Field(default=1, ge=1, description="Total processes across all nodes")
    gpus_per_process: int = Field(default=0, ge=0)
    rendezvous: dict[str, str] = Field(
        default_factory=dict,
        description="Rendezvous hints the launcher resolves at start time "
        "(e.g. {'backend': 'c10d'}). Addresses/ports are ALWAYS resolved "
        "by the launcher — a compiled plan must stay host-agnostic.",
    )
    workdir_hint: str = Field(
        default="", description="Relative workdir layout the command expects, if any"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Human-readable compilation notes for the ledger "
        "(mirrors the planner's explanation habit).",
    )


class StrategyCompiler(ABC):
    """One strategy family's translator.

    Lifecycle: `supports(plan)` → `validate(plan)` → `compile(plan)`.
    The registry calls `supports` to route; callers may run `validate`
    separately for dry-run UX (surface every problem before spending
    anything); `compile` implies validation and raises `CompileError` on
    any defect.
    """

    #: Family name this compiler owns — must match
    #: `StrategyPlan.strategy_family` values (e.g. "ddp", "fsdp2",
    #: "zero3_cpu_offload", "lease_tasks").
    family: ClassVar[str]

    @abstractmethod
    def supports(self, plan: StrategyPlan) -> bool:
        """Cheap routing predicate. Must not raise; unknown plans → False."""

    def validate(self, plan: StrategyPlan) -> list[str]:
        """Return every reason this plan cannot compile (empty = fine).

        Default: no extra checks beyond `supports`. Override to verify
        family-specific invariants (e.g. FSDP2 requires workers > 1;
        DeepSpeed offload requires `plan.offload != 'none'`). Never check
        the *environment* here — that is `Launcher.healthy()`.
        """
        return [] if self.supports(plan) else [f"{self.family}: unsupported plan"]

    @abstractmethod
    def compile(self, plan: StrategyPlan) -> LaunchSpec:
        """Translate the plan into a LaunchSpec.

        Input: a frozen, planner-emitted StrategyPlan (treat as immutable).
        Output: a complete LaunchSpec — argv, env, config files.
        Raises: CompileError with *all* problems when validation fails.
        Determinism contract: identical plan ⇒ identical LaunchSpec
        (the ledger records the spec hash next to the plan_id).
        """


_REGISTRY: list[StrategyCompiler] = []


def register_compiler(compiler: StrategyCompiler) -> None:
    """Register a compiler instance (idempotent per family: a re-register
    of the same family replaces the previous one — supports test setups
    and plugin reloads)."""
    _REGISTRY[:] = [c for c in _REGISTRY if c.family != compiler.family]
    _REGISTRY.append(compiler)


def compiler_for(plan: StrategyPlan) -> StrategyCompiler:
    """Dispatch: first registered compiler whose `supports(plan)` is True.

    Raises LookupError naming the family — the caller surfaces this as a
    422 (\"plan compiled by no installed strategy family\"), never a 500.
    """
    for compiler in _REGISTRY:
        if compiler.supports(plan):
            return compiler
    raise LookupError(
        f"no strategy compiler for family {plan.strategy_family!r} "
        f"(registered: {[c.family for c in _REGISTRY] or 'none'})"
    )
