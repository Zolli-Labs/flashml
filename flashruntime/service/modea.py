"""Mode A over HTTP: the lease coordinator, node registry, and local
artifact hosting for the self-hosted profile.

This module makes the pure-library pieces reachable by remote workers:

- **Leases** — HTTP verbs over the `LeaseManager` (claim / heartbeat /
  complete / fail) plus a background sweep. FlashNode's executor is the
  intended client, but anything that speaks the protocol can pull work.
- **Node registry** — minimal register/heartbeat/list so a self-hosted
  coordinator knows its workers (FlashML Cloud fronts this with join codes
  and trust tiers in the managed product; the wire models are the same).
- **Local artifacts** — PUT/GET raw bytes under a local directory, so
  shared data (datasets in, trial outputs back) needs no cloud and no
  MinIO: the coordinator *is* the artifact host for the local loop. Keys
  are sha256-verified on upload; the same `artifact://` URIs used by the
  cloud stores apply, keeping job specs portable.

Job → task expansion lives here too: a JobSpec with
`execution.backend: leases` and `workload.type: hyperparameter_search`
becomes N `TaskSpec`s whose payloads carry the executor contract
(module, params, input artifact keys, output prefix).
"""

from __future__ import annotations

import hashlib
import itertools
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

import flashruntime.recipes.command  # noqa: F401 — registers the "command" recipe
from flashruntime.leases import LeaseManager
from flashruntime.protocol.v1alpha1 import (
    ArtifactRecord,
    JobSpec,
    NodeHeartbeat,
    NodeRegistration,
    TaskSpec,
    TaskState,
)
from flashruntime.recipes import recipe_for
from flashruntime.scheduler import IsolationAwarePlacement

NODE_OFFLINE_AFTER_S = 15.0

# Task modules the *coordinator* will hand out. The executor enforces its own
# allowlist too — both ends fail closed.
ALLOWED_TASK_MODULES = {
    "flashml_workloads.sklearn_trial",
    "flashml_workloads.kmeans_shard",
    "flashml_workloads.sgd_trainer",
}


class ExpansionError(ValueError):
    """The JobSpec cannot be expanded into tasks (bad workload/parameters)."""


def expand_tasks(job_id: str, spec: JobSpec) -> list[TaskSpec]:
    """Turn a lease-mode JobSpec into independent TaskSpecs.

    hyperparameter_search parameters:
      trials:      explicit list of param dicts, or
      grid:        {param: [values, ...]} — cartesian product
      module:      task module (allowlisted; default sklearn_trial)
      inputs:      {name: "artifact://key"} shared data, downloaded per task
      lease_seconds: heartbeat window per attempt (default 60)
    """
    workload = spec.spec.workload
    try:
        recipe = recipe_for(workload.type)
    except LookupError:
        recipe = None
    if recipe is not None:
        try:
            return recipe.expand(job_id, spec)
        except ValueError as exc:
            raise ExpansionError(str(exc)) from None
    if workload.type == "sharded_kmeans":
        return _expand_kmeans(job_id, spec)
    if workload.type != "hyperparameter_search":
        raise ExpansionError(
            f"lease backend supports workload types 'hyperparameter_search' and "
            f"'sharded_kmeans', got '{workload.type}'"
        )
    p = workload.parameters
    trials: list[dict] = list(p.get("trials") or [])
    if not trials and p.get("grid"):
        grid: dict[str, list] = p["grid"]
        keys = sorted(grid)
        trials = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    if not trials:
        raise ExpansionError("hyperparameter_search needs 'trials' (list) or 'grid' (dict of lists)")

    module = p.get("module", "flashml_workloads.sklearn_trial")
    if module not in ALLOWED_TASK_MODULES:
        raise ExpansionError(f"task module '{module}' is not allowlisted")
    inputs = dict(p.get("inputs") or {})
    for name, uri in inputs.items():
        if not str(uri).startswith("artifact://"):
            raise ExpansionError(f"input '{name}' must be an artifact:// URI, got {uri!r}")

    checkpoint = p.get("checkpoint")  # non-None turns the executor's relay on
    # Stamp the isolation requirement so the placement gate can fail closed —
    # a sandboxed job must never lease to a non-sandbox node (mirrors
    # recipes/command.py; the legacy expansions were dropping this).
    isolation = {
        "tier": spec.spec.isolation.tier,
        "allowFallback": spec.spec.isolation.allowFallback,
    }
    tasks = []
    for i, params in enumerate(trials):
        task_id = f"trial-{i:03d}"
        payload = {
            "module": module,
            "params": params,
            "inputs": inputs,
            "output_prefix": f"jobs/{job_id}/{task_id}/",
            "task_id": task_id,
            # the docker-runner tier resolves and allowlists this
            "image": spec.spec.image.reference,
            "isolation": isolation,
        }
        if checkpoint is not None:
            payload["checkpoint"] = checkpoint
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


def _expand_kmeans(job_id: str, spec: JobSpec) -> list[TaskSpec]:
    """One K-means *iteration*: one task per data shard, each computing
    partial sums against the broadcast centroids. The driver
    (`flashml_workloads.kmeans_driver`) reduces and submits the next
    iteration as a new job — stage composition, not a new execution mode."""
    p = spec.spec.workload.parameters
    shards: list[str] = list(p.get("shards") or [])
    centroids = p.get("centroids")
    if not shards or not centroids:
        raise ExpansionError("sharded_kmeans needs 'shards' (artifact:// list) and 'centroids'")
    for uri in shards:
        if not str(uri).startswith("artifact://"):
            raise ExpansionError(f"shard must be an artifact:// URI, got {uri!r}")
    iteration = int(p.get("iteration", 0))
    # Same fail-closed stamp as the hyperparameter_search path (mirrors
    # recipes/command.py) — without it a sandboxed job leases anywhere.
    isolation = {
        "tier": spec.spec.isolation.tier,
        "allowFallback": spec.spec.isolation.allowFallback,
    }

    tasks = []
    for i, shard_uri in enumerate(shards):
        task_id = f"it{iteration:02d}-shard-{i:03d}"
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_id=job_id,
                commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                lease_seconds=float(p.get("lease_seconds", 60.0)),
                payload={
                    "module": "flashml_workloads.kmeans_shard",
                    "params": {"centroids": centroids},
                    "inputs": {"shard": shard_uri},
                    "output_prefix": f"jobs/{job_id}/{task_id}/",
                    "task_id": task_id,
                    "image": spec.spec.image.reference,
                    "isolation": isolation,
                },
            )
        )
    return tasks


class _NodeEntry(BaseModel):
    registration: NodeRegistration
    last_heartbeat: datetime
    accepted_tasks: int = 0


class ModeAState:
    """Shared state behind the router. In-memory by design for the local
    loop (the ledger keeps the durable event history); the Stage-6 upgrade
    swaps the store for Postgres without touching the endpoints."""

    def __init__(
        self,
        manager: LeaseManager,
        artifacts_dir: Path,
        join_code: str | None = None,
        max_artifact_bytes: int = 256 * 1024 * 1024,
    ):
        self.manager = manager
        self.artifacts_dir = artifacts_dir
        self.join_code = join_code  # None = open registration (self-hosted default)
        self.max_artifact_bytes = max_artifact_bytes
        self.nodes: dict[str, _NodeEntry] = {}
        self.lease_jobs: set[str] = set()  # job_ids running on the lease path

    def node_view(self) -> list[dict]:
        now = datetime.now(timezone.utc)
        out = []
        for entry in self.nodes.values():
            age = (now - entry.last_heartbeat).total_seconds()
            out.append(
                {
                    "node_id": entry.registration.node_id,
                    "hostname": entry.registration.hostname,
                    "environment": entry.registration.environment,
                    "argv_capable": entry.registration.argv_capable,
                    "capabilities": entry.registration.capabilities.model_dump(),
                    "online": age < NODE_OFFLINE_AFTER_S,
                    "last_heartbeat_age_s": round(age, 1),
                    "accepted_tasks": entry.accepted_tasks,
                }
            )
        return sorted(out, key=lambda n: n["node_id"])


class ClaimRequest(BaseModel):
    node_id: str
    job_id: str | None = None


class CompleteRequest(BaseModel):
    output_sha256: str


class FailRequest(BaseModel):
    reason: str


def _output_valid(artifacts_dir: Path, commit_key: str, claimed_sha256: str) -> bool:
    path = artifacts_dir / commit_key
    if not path.is_file():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == claimed_sha256


def _safe_key(key: str) -> str:
    """Artifact keys are relative paths under the artifacts dir — refuse
    anything that could escape it."""
    if key.startswith("/") or ".." in key.split("/"):
        raise HTTPException(status_code=400, detail=f"invalid artifact key: {key}")
    return key


def build_router(state: ModeAState) -> APIRouter:
    router = APIRouter(prefix="/v1alpha1")
    manager = state.manager

    # -- node registry ------------------------------------------------------

    @router.post("/nodes/register")
    async def register_node(reg: NodeRegistration, request: Request):
        if state.join_code is not None:
            supplied = request.headers.get("X-FlashML-Join-Code")
            if supplied != state.join_code:
                raise HTTPException(status_code=403, detail="invalid or missing join code")
        state.nodes[reg.node_id] = _NodeEntry(
            registration=reg, last_heartbeat=datetime.now(timezone.utc)
        )
        return {"node_id": reg.node_id, "status": "registered"}

    @router.post("/nodes/{node_id}/heartbeat")
    async def node_heartbeat(node_id: str, hb: NodeHeartbeat):
        entry = state.nodes.get(node_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id} — register first")
        entry.last_heartbeat = hb.timestamp
        return {"status": "ok"}

    @router.get("/nodes")
    async def list_nodes():
        return state.node_view()

    # -- leases -------------------------------------------------------------

    @router.post("/leases/claim")
    async def claim(req: ClaimRequest):
        entry = state.nodes.get(req.node_id)
        if entry is None:
            raise HTTPException(status_code=403, detail="unregistered node — register first")
        node_view = {
            "node_id": req.node_id,
            "sandbox_capable": entry.registration.sandbox_capable,
            "argv_capable": entry.registration.argv_capable,
            "capabilities": entry.registration.capabilities.model_dump(),
        }
        lease = manager.claim(
            req.node_id,
            job_id=req.job_id,
            policy=IsolationAwarePlacement(),
            node=node_view,
        )
        if lease is None:
            return Response(status_code=204)  # nothing claimable right now
        return lease

    @router.post("/attempts/{lease_id}/heartbeat")
    async def attempt_heartbeat(lease_id: str):
        from flashruntime.leases import LeaseError

        try:
            return manager.heartbeat(lease_id)
        except LeaseError as exc:
            # 410 Gone: the worker must stop — its lease is dead.
            raise HTTPException(status_code=410, detail=str(exc))

    @router.post("/attempts/{lease_id}/complete")
    async def attempt_complete(lease_id: str, req: CompleteRequest):
        from flashruntime.leases import LeaseError

        lease = manager.lease_info(lease_id)
        if lease is None:
            raise HTTPException(status_code=404, detail=f"unknown lease {lease_id}")

        # Accepted work = validated output: the artifact at the task's
        # commit_key must exist and hash to what the worker claims. A bad
        # upload fails the attempt (task requeues elsewhere); it never
        # commits. Fault tolerance that accepts wrong results is worse than
        # failure.
        record = next(
            (r for r in manager.records(lease.job_id) if r.spec.task_id == lease.task_id), None
        )
        if record is not None and not _output_valid(
            state.artifacts_dir, record.spec.commit_key, req.output_sha256
        ):
            try:
                manager.fail(
                    lease_id, f"output validation failed for {record.spec.commit_key}"
                )
                return {"accepted": False, "detail": "output validation failed; attempt requeued"}
            except LeaseError:
                pass  # lease already dead → fall through to the late-commit rejection

        try:
            accepted = manager.complete(lease_id, output_sha256=req.output_sha256)
        except LeaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if accepted and lease.node_id in state.nodes:
            # credit accepted work only — the contribution-accounting rule
            state.nodes[lease.node_id].accepted_tasks += 1
        return {"accepted": accepted}

    @router.post("/attempts/{lease_id}/fail")
    async def attempt_fail(lease_id: str, req: FailRequest):
        from flashruntime.leases import LeaseError

        try:
            manager.fail(lease_id, req.reason)
        except LeaseError as exc:
            raise HTTPException(status_code=410, detail=str(exc))
        return {"status": "requeued-or-exhausted"}

    # -- tasks view ---------------------------------------------------------

    @router.get("/jobs/{job_id}/tasks")
    async def job_tasks(job_id: str):
        out = []
        for r in manager.records(job_id):
            lease = r.active_lease
            last = None
            if r.lease_history:
                last = list(r.lease_history.values())[-1]
            out.append(
                {
                    "task_id": r.spec.task_id,
                    "state": r.state.value,
                    "attempts": r.attempts_used,
                    "max_attempts": r.spec.max_attempts,
                    "node_id": (lease or last).node_id if (lease or last) else None,
                    "deadline": lease.deadline.isoformat() if lease else None,
                }
            )
        return sorted(out, key=lambda t: t["task_id"])

    # -- local artifact hosting --------------------------------------------

    @router.put("/artifacts/{key:path}")
    async def put_artifact(key: str, request: Request):
        key = _safe_key(key)
        data = await request.body()
        if len(data) > state.max_artifact_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"artifact exceeds {state.max_artifact_bytes} bytes",
            )
        path = state.artifacts_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ArtifactRecord(
            uri=f"artifact://{key}",
            backend="local",
            bucket=str(state.artifacts_dir),
            object_key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    @router.get("/artifacts/{key:path}")
    async def get_artifact(key: str):
        key = _safe_key(key)
        path = state.artifacts_dir / key
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no artifact at {key}")
        return Response(content=path.read_bytes(), media_type="application/octet-stream")

    @router.get("/jobs/{job_id}/artifacts")
    async def job_artifacts(job_id: str):
        base = state.artifacts_dir / "jobs" / job_id
        if not base.is_dir():
            return []
        out = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                key = str(path.relative_to(state.artifacts_dir))
                out.append(
                    {
                        "uri": f"artifact://{key}",
                        "key": key,
                        "size_bytes": path.stat().st_size,
                    }
                )
        return out

    return router


def lease_job_state(manager: LeaseManager, job_id: str) -> tuple[str, dict[str, int]]:
    """Derive a JobState name from the task counts (status is never a
    hand-mutated field — it falls out of the lease table)."""
    counts = manager.job_state(job_id)
    total = sum(counts.values())
    done = counts.get(TaskState.COMPLETED.value, 0)
    failed = counts.get(TaskState.FAILED.value, 0)
    active = counts.get(TaskState.PENDING.value, 0) + counts.get(TaskState.LEASED.value, 0)
    if total == 0:
        return "PENDING", counts
    if active > 0:
        return "RUNNING", counts
    if failed > 0:
        return "FAILED", counts
    if done == total:
        return "SUCCEEDED", counts
    return "CANCELLED", counts
