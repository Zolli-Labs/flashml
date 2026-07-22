"""The generic command recipe: JobSpec{workload.type: "command"} → lease tasks.

The first concrete WorkloadRecipe. Payloads carry `argv` — the §2.2
executor contract generalized from `module` — plus the isolation
requirement the placement gate enforces fail-closed. Executing argv
payloads is flashnode's runner tier (cross-repo, versioned change); this
recipe defines the coordinator half of that contract. Until flashnode
ships it, command jobs expand and lease correctly but only argv-aware
executors can run them.
"""

from __future__ import annotations

from typing import Any, ClassVar

from flashruntime.protocol.v1alpha1 import JobSpec, TaskSpec
from flashruntime.recipes import WorkloadRecipe, register_recipe


class CommandRecipe(WorkloadRecipe):
    kind: ClassVar[str] = "command"
    #: argv payloads name no task module — the isolation tier, not a module
    #: allowlist, is the security control for this workload type.
    task_module: ClassVar[str] = ""

    def validate_params(self, params: dict[str, Any]) -> list[str]:
        problems: list[str] = []
        command = params.get("command")
        if (
            not command
            or not isinstance(command, list)
            or not all(isinstance(t, str) for t in command)
        ):
            problems.append("'command' must be a non-empty argv list of strings")
        for name, uri in (params.get("inputs") or {}).items():
            if not str(uri).startswith("artifact://"):
                problems.append(f"input '{name}' must be an artifact:// URI")
        task_params = params.get("task_params")
        if task_params is not None and (
            not isinstance(task_params, list)
            or not all(isinstance(p, dict) for p in task_params)
        ):
            problems.append("'task_params' must be a list of objects")
        return problems

    def expand(self, job_id: str, spec: JobSpec) -> list[TaskSpec]:
        p = spec.spec.workload.parameters
        problems = self.validate_params(p)
        if problems:
            raise ValueError("; ".join(problems))

        param_sets: list[dict | None] = p.get("task_params") or [None]
        env: dict[str, str] = dict(p.get("env") or {})
        inputs = dict(p.get("inputs") or {})
        isolation = {
            "tier": spec.spec.isolation.tier,
            "allowFallback": spec.spec.isolation.allowFallback,
        }

        tasks: list[TaskSpec] = []
        for i, params in enumerate(param_sets):
            task_id = f"task-{i:03d}"
            try:
                argv = [t.format(**params) for t in p["command"]] if params else list(p["command"])
                task_env = {
                    k: (v.format(**params) if params else v) for k, v in env.items()
                }
            except KeyError as exc:
                raise ValueError(
                    f"task {i}: placeholder {exc} has no value in task_params[{i}]"
                ) from None
            payload: dict[str, Any] = {
                "argv": argv,
                "env": task_env,
                "inputs": inputs,
                "output_prefix": f"jobs/{job_id}/{task_id}/",
                "task_id": task_id,
                "image": spec.spec.image.reference,
                "isolation": isolation,
            }
            if p.get("checkpoint") is not None:
                payload["checkpoint"] = p["checkpoint"]
            tasks.append(
                TaskSpec(
                    task_id=task_id,
                    job_id=job_id,
                    commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                    max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                    lease_seconds=float(p.get("lease_seconds", 60.0)),
                    payload=payload,
                )
            )
        return tasks

    def validate_output(self, metrics: dict[str, Any]) -> None:
        if not isinstance(metrics, dict):
            raise ValueError("metrics.json must contain a JSON object")


register_recipe(CommandRecipe())
