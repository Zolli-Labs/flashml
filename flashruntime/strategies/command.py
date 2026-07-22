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
        # torchrun's worker count flag has two spellings (hyphen/underscore)
        # and two forms (--flag=N or --flag N). Values may also be
        # symbolic (auto/gpu/cpu), resolved by torchrun at launch — those
        # must not crash compilation: leave world_size=1 and note it.
        nproc_flags = ("--nproc-per-node", "--nproc_per_node")
        i = 1
        while i < len(argv):
            token = argv[i]
            flag = value = None
            if "=" in token:
                head, _, tail = token.partition("=")
                if head in nproc_flags:
                    flag, value = head, tail
            elif token in nproc_flags:
                flag = token
                value = argv[i + 1] if i + 1 < len(argv) else None
                i += 1  # consume the value token
            if flag is not None and value is not None:
                try:
                    world_size = int(value)
                    notes.append(f"world_size from torchrun: {world_size}")
                except ValueError:
                    notes.append(
                        f"world_size unresolved: {flag}={value} (resolved at launch time)"
                    )
            i += 1
    return LaunchSpec(
        argv=argv,
        env=env,
        world_size=world_size,
        workdir_hint=workload.source.path,
        notes=notes,
    )
