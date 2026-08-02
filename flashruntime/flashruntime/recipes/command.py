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

import os
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
        problems.extend(self._unpack_problems(params))
        return problems

    @staticmethod
    def _unpack_problems(params: dict[str, Any]) -> list[str]:
        """Validate `unpack_inputs` — the names flashnode may run an archive
        extractor over.

        This value decides which downloaded bytes get unpacked into a
        directory tree on a *volunteer's* machine, so it is validated at
        expansion time rather than forwarded as given. Naming an input that
        was never declared is the interesting case: flashnode refuses it
        too, but only after the task has been leased, claimed and half-run,
        which surfaces to the submitter as a mysterious node-side failure
        instead of the spec error it is. Catching it here fails on the
        submitter's side, before anything is placed.
        """
        unpack = params.get("unpack_inputs")
        if unpack is None:
            return []
        if not isinstance(unpack, list) or not all(isinstance(n, str) for n in unpack):
            return ["'unpack_inputs' must be a list of input names (strings)"]
        problems: list[str] = []
        duplicates = sorted({n for n in unpack if unpack.count(n) > 1})
        if duplicates:
            problems.append(
                f"'unpack_inputs' names {duplicates} more than once — an input "
                f"is unpacked at most once"
            )
        declared = set((params.get("inputs") or {}).keys())
        unknown = sorted(set(unpack) - declared)
        if unknown:
            problems.append(
                f"'unpack_inputs' names inputs that are not declared in "
                f"'inputs': {unknown}"
            )
        return problems

    def expand(self, job_id: str, spec: JobSpec) -> list[TaskSpec]:
        isolation_spec = spec.spec.isolation
        if isolation_spec.allowFallback:
            # allowFallback waives the sandbox capability requirement at
            # placement time. Honouring it for argv would let a submitter
            # place arbitrary code on an unsandboxed node.
            raise ValueError(
                "command jobs may not set isolation.allowFallback — "
                "argv execution is container-only"
            )
        if isolation_spec.tier != "sandboxed":
            # Coordinator-side opt-in only: the operator running the pool
            # decides, never the submitter.
            if os.environ.get("FLASHML_ALLOW_UNSANDBOXED_ARGV") != "1":
                raise ValueError(
                    f"command jobs require isolation.tier 'sandboxed', got "
                    f"{isolation_spec.tier!r} (set FLASHML_ALLOW_UNSANDBOXED_ARGV=1 "
                    f"on the coordinator to allow a trusted fleet)"
                )
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
            except (KeyError, IndexError, ValueError) as exc:
                # KeyError: a named {placeholder} with no matching param.
                # IndexError/ValueError: an auto/positional field ({}, {0}) or
                # a malformed brace str.format cannot fill from a params dict.
                # All are user-input errors → ValueError (→422), never a 500.
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
            if p.get("unpack_inputs") is not None:
                # Absent stays absent, never an empty list: flashnode reads
                # `payload.get("unpack_inputs")` and an omitted key is the
                # path where every input keeps its plain-file behaviour byte
                # for byte. Emitting `[]` would mean the same thing today but
                # would stop exercising that path.
                payload["unpack_inputs"] = list(p["unpack_inputs"])
            if p.get("local_inputs") is not None:
                # Labels for host-supplied directories. This forward is what
                # makes the local-data feature real: `IsolationAwarePlacement`
                # reads `task.payload["local_inputs"]` to decide eligibility,
                # and flashnode reads it again to know what to mount.
                #
                # Dropping it does NOT fail closed. The gate sees a task
                # requiring nothing, places it on any node, and flashnode
                # mounts nothing — so the task runs without the data it asked
                # for. Both ends of this hop have tests that pass while it is
                # broken, because each constructs the payload directly.
                #
                # `list(...)` so a payload never aliases the caller's spec.
                payload["local_inputs"] = list(p["local_inputs"])
            gpus = spec.spec.resources.gpuPerTask
            if gpus:
                # The GPU requirement is the ONLY payload key sourced from
                # `spec.spec.resources` rather than `workload.parameters` —
                # a branch of the spec nothing else in this method reads.
                # That makes it the easiest forward in the file to leave out
                # and the hardest to notice missing: there is no unrecognised
                # parameter sitting in `p` to go looking for.
                #
                # Dropping it does NOT fail closed. `IsolationAwarePlacement`
                # reads `task.payload["gpus"]`; with the key absent it sees a
                # task requiring nothing, places it on any node, and a CUDA
                # job lands on a CPU-only volunteer. Both ends of this hop
                # have tests that pass while it is broken, because each
                # constructs the payload directly.
                #
                # Absent stays absent, never 0 — the no-GPU path must keep
                # exercising the key-missing branch, as unpack_inputs and
                # local_inputs do.
                payload["gpus"] = int(gpus)
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
