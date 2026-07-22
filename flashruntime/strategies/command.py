"""Compile a CommandWorkload into the backend-neutral LaunchSpec.

Pure and deterministic (same rules as StrategyCompiler): no environment
inspection, no filesystem access — resolving the workdir and creating
output directories is the launcher's job.

Note: this is a module function, not a StrategyCompiler subclass — a
StrategyPlan carries no argv, so a plan-driven compiler for commands is
meaningless until flash.run() wiring lands (spec §10 follow-up).
"""

from __future__ import annotations

from flashruntime.strategies import LaunchSpec
from flashruntime.workloads.command import CommandWorkload


def compile_workload(workload: CommandWorkload, params: dict | None = None) -> LaunchSpec:
    argv = workload.argv(params)
    env = {k: (v.format(**params) if params else v) for k, v in workload.env.items()}
    world_size = 1
    notes = [f"mode={workload.resolved_mode()}"]
    if argv and argv[0] == "torchrun":
        for token in argv:
            if token.startswith("--nproc-per-node="):
                world_size = int(token.split("=", 1)[1])
                notes.append(f"world_size from torchrun: {world_size}")
    return LaunchSpec(
        argv=argv,
        env=env,
        world_size=world_size,
        workdir_hint=workload.source.path,
        notes=notes,
    )
