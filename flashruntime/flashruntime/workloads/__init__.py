"""Workload descriptions: WHAT the user wants run (four-axes rule: this is
the axis-zero input the recipes/strategies/launchers axes consume)."""

from flashruntime.workloads.command import CommandWorkload, OutputSpec, Source, to_jobspec

__all__ = ["CommandWorkload", "OutputSpec", "Source", "to_jobspec"]
