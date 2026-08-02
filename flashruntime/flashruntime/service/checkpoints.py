"""Checkpoint HTTP surface: the CheckpointCatalog over the wire.

Scoped per (job, task): each Mode A task owns its checkpoint lineage, so a
retried attempt — possibly on a different machine — resumes from *its
task's* latest valid manifest. Internally the scope is a composite catalog
key; the catalog itself (parts-first / manifest-last, validation ladder,
selection) is untouched and separately tested.

Known limitation, on purpose: the catalog is in-memory, so manifests do not
survive a coordinator restart (the checkpoint *files* in the artifact store
do). Manifest persistence joins the Postgres stage; the ledger already
records every commit event.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from flashruntime.checkpoint import CheckpointCatalog, CheckpointError
from flashruntime.protocol.v1alpha1 import CheckpointPart
from flashruntime.service.modea import authorize_task_write


def _scope(job_id: str, task_id: str) -> str:
    return f"{job_id}::{task_id}"


class RegisterPartRequest(BaseModel):
    attempt_id: str
    step: int = Field(ge=0)
    part: CheckpointPart


class CommitRequest(BaseModel):
    attempt_id: str
    step: int = Field(ge=0)
    expected_parts: list[CheckpointPart]
    storage_prefix: str
    world_size: int = 1
    framework: str = ""
    strategy_family: str = ""


def build_router(catalog: CheckpointCatalog, state, manager) -> APIRouter:
    router = APIRouter(prefix="/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints")

    @router.post("/parts")
    async def register_part(job_id: str, task_id: str, req: RegisterPartRequest, request: Request):
        authorize_task_write(state, manager, request, job_id, task_id)
        catalog.register_part(_scope(job_id, task_id), req.attempt_id, req.step, req.part)
        return {"status": "registered", "key": req.part.key}

    @router.post("/commit")
    async def commit(job_id: str, task_id: str, req: CommitRequest, request: Request):
        authorize_task_write(state, manager, request, job_id, task_id)
        try:
            manifest = catalog.commit(
                job_id=_scope(job_id, task_id),
                attempt_id=req.attempt_id,
                step=req.step,
                expected_parts=req.expected_parts,
                storage_prefix=req.storage_prefix,
                world_size=req.world_size,
                framework=req.framework,
                strategy_family=req.strategy_family,
            )
        except CheckpointError as exc:
            # 409: the manifest was refused — parts missing or hashes wrong.
            raise HTTPException(status_code=409, detail=str(exc))
        return manifest

    @router.get("/latest")
    async def latest(job_id: str, task_id: str, world_size: int | None = None):
        manifest = catalog.latest_valid(_scope(job_id, task_id), world_size=world_size)
        if manifest is None:
            raise HTTPException(status_code=404, detail="no valid checkpoint")
        return manifest

    @router.get("/lost-work")
    async def lost_work(job_id: str, task_id: str, failed_at_step: int):
        manifest = catalog.latest_valid(_scope(job_id, task_id))
        if manifest is None:
            return {"lost_steps": None, "latest_step": None}
        return {
            "lost_steps": max(0, failed_at_step - manifest.step),
            "latest_step": manifest.step,
        }

    return router
