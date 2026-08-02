"""User-facing description of a "bring your own code" workload.

A CommandWorkload names WHAT to run (a command in a source directory,
optionally in a pinned image) and what FlashRuntime should do around it
(inputs, outputs, isolation, Mode A fan-out). It never describes HOW
distributed math happens — that belongs to the user's code and its
framework (ADR-0003: FlashRuntime operates jobs, it does not train).

Pydantic-only: importing this module must never require torch, sklearn,
kubernetes, or fastapi — it is part of the clean core.
"""

from __future__ import annotations

import shlex
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from flashruntime.protocol.plan_v1alpha1 import CheckpointPolicy
from flashruntime.protocol.v1alpha1 import (
    ExecutionSpec,
    ImageSpec,
    IsolationSpec,
    JobMetadata,
    JobSpec,
    JobSpecInner,
    WorkloadSpec,
)
from flashruntime.providers import Requirements


class Source(BaseModel):
    """Where the user's code lives. v1 executes from a local directory;
    `git_revision` is reserved for remote packaging (spec §10 follow-up)."""

    path: str = "."
    git_revision: str | None = None


class OutputSpec(BaseModel):
    """What to keep after a run. `collect` globs are resolved against the
    script's working directory; `primary_metric` names the metrics.json key
    Run.best_trial() ranks by."""

    prefix: str = "artifact://jobs/{job_id}/"
    collect: list[str] = Field(default_factory=lambda: ["metrics.json"])
    primary_metric: str | None = None
    maximize: bool = True


class CommandWorkload(BaseModel):
    """One command, operated by FlashRuntime.

    `command` may be a shell-style string (shlex-split, never shell=True —
    pipes need an explicit `bash -c "..."`) or an argv list. `{name}`
    placeholders are filled per `task_params` entry for Mode A fan-out.
    """

    command: str | list[str]
    source: Source = Field(default_factory=Source)
    image: ImageSpec | None = None
    env: dict[str, str] = Field(default_factory=dict)
    inputs: dict[str, str] = Field(default_factory=dict)
    outputs: OutputSpec = Field(default_factory=OutputSpec)
    resources: Requirements = Field(default_factory=Requirements)
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    mode: Literal["auto", "local", "independent_tasks", "coordinated"] = "auto"
    checkpoint: CheckpointPolicy | None = None
    task_params: list[dict] | None = None

    @field_validator("inputs")
    @classmethod
    def _artifact_scheme(cls, v: dict[str, str]) -> dict[str, str]:
        for name, uri in v.items():
            if not str(uri).startswith("artifact://"):
                raise ValueError(f"input '{name}' must be an artifact:// URI, got {uri!r}")
        return v

    def argv(self, params: dict | None = None) -> list[str]:
        """Exec-ready argv. `params` fills `{name}` placeholders; a
        placeholder with no matching param raises KeyError (a silent empty
        substitution would corrupt the command)."""
        tokens = shlex.split(self.command) if isinstance(self.command, str) else list(self.command)
        if params:
            tokens = [t.format(**params) for t in tokens]
        return tokens

    def resolved_mode(self) -> str:
        """Deterministic `auto` resolution (spec §4.1): fan-out params ⇒
        independent_tasks; a multi-process launcher command ⇒ coordinated;
        else local. An explicit `mode` always wins."""
        if self.mode != "auto":
            return self.mode
        if self.task_params:
            return "independent_tasks"
        tokens = self.argv()
        if tokens and tokens[0] in ("torchrun", "accelerate"):
            return "coordinated"
        return "local"


def to_jobspec(workload: CommandWorkload, name: str, image: ImageSpec | None = None) -> JobSpec:
    """Wire form for the coordinator: JobSpec{execution.backend: leases,
    workload.type: "command"}. A pinned image is required — remote runs
    must be reproducible (the schema already rejects 'latest')."""
    img = image or workload.image
    if img is None:
        raise ValueError("a pinned image is required to submit a command workload to the service")
    parameters: dict = {
        "command": workload.argv(),  # normalized argv, placeholders intact
        "env": dict(workload.env),
        "inputs": dict(workload.inputs),
    }
    if workload.task_params is not None:
        parameters["task_params"] = workload.task_params
    if workload.checkpoint is not None:
        parameters["checkpoint"] = workload.checkpoint.model_dump()
    return JobSpec(
        metadata=JobMetadata(name=name),
        spec=JobSpecInner(
            execution=ExecutionSpec(backend="leases"),
            image=img,
            workload=WorkloadSpec(type="command", parameters=parameters),
            isolation=workload.isolation,
        ),
    )
