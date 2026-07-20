"""FlashRuntime API.

Endpoints (all under /v1alpha1):
  POST /jobs               submit a JobSpec
  GET  /jobs               list job records
  GET  /jobs/{id}          one job record
  GET  /jobs/{id}/events   ordered event ledger
  GET  /jobs/{id}/logs     driver logs from the backend
  POST /jobs/{id}/cancel   cancel
  GET  /healthz            liveness

Configuration is environment-driven (see `RuntimeSettings`); the public
JobSpec never carries deployment details.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from fastapi import FastAPI, HTTPException

from flashruntime.artifacts import artifact_uri_to_key, store_from_env
from flashruntime.backends.base import SpecValidationError
from flashruntime.backends.kuberay import KubeRayBackendConfig, KubeRayExecutionBackend
from flashruntime.checkpoint import CheckpointCatalog
from flashruntime.leases import LeaseManager
from flashruntime.leases.sqlite_store import SqliteLeaseStore
from flashruntime.protocol.v1alpha1 import (
    Event,
    EventType,
    JobRecord,
    JobSpec,
    JobState,
    utcnow,
)
from flashruntime.service import checkpoints, dashboard, modea
from flashruntime.service.ledger import Ledger
from flashruntime.service.modea import ModeAState

log = logging.getLogger("flashruntime.service")
logging.basicConfig(
    level=logging.INFO,
    format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)s}',
)


def _jlog(msg: str, **kv) -> str:
    return json.dumps({"text": msg, **kv})


@dataclass
class RuntimeSettings:
    profile: str = field(default_factory=lambda: os.environ.get("FLASHML_PROFILE", "local"))
    namespace: str = field(default_factory=lambda: os.environ.get("FLASHML_NAMESPACE", "flashml"))
    ledger_path: str = field(
        default_factory=lambda: os.environ.get("FLASHML_LEDGER_PATH", "/data/flashruntime.db")
    )
    # Mode B (KubeRay) is optional: the self-hosted local loop runs leases +
    # local artifacts with no Kubernetes anywhere near it.
    enable_kuberay: bool = field(
        default_factory=lambda: os.environ.get("FLASHML_ENABLE_KUBERAY", "1") == "1"
    )
    artifacts_dir: str = field(
        default_factory=lambda: os.environ.get("FLASHML_LOCAL_ARTIFACTS_DIR", "/data/artifacts")
    )
    # Optional first auth primitive: when set, node registration requires the
    # matching X-FlashML-Join-Code header. Unset = open (self-hosted default).
    join_code: str | None = field(
        default_factory=lambda: os.environ.get("FLASHML_JOIN_CODE") or None
    )
    max_artifact_mb: int = field(
        default_factory=lambda: int(os.environ.get("FLASHML_MAX_ARTIFACT_MB", "256"))
    )
    standard_node_selector: dict[str, str] | None = None
    sandbox_node_selector: dict[str, str] = field(default_factory=dict)
    sandbox_runtime_class: str | None = None

    @staticmethod
    def _parse_selector(value: str) -> dict[str, str]:
        return dict(pair.split("=", 1) for pair in value.split(",") if "=" in pair)

    @classmethod
    def from_env(cls) -> "RuntimeSettings":
        settings = cls()
        standard = os.environ.get("FLASHML_STANDARD_NODE_SELECTOR", "")
        if standard:
            settings.standard_node_selector = cls._parse_selector(standard)
        sandbox = os.environ.get("FLASHML_SANDBOX_NODE_SELECTOR", "")
        if sandbox:
            settings.sandbox_node_selector = cls._parse_selector(sandbox)
        settings.sandbox_runtime_class = os.environ.get("FLASHML_RUNTIME_CLASS") or None
        return settings


def create_app(settings: RuntimeSettings | None = None) -> FastAPI:
    settings = settings or RuntimeSettings.from_env()
    Path(settings.ledger_path).parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(settings.ledger_path)

    backend = None
    if settings.enable_kuberay:
        backend_cfg = KubeRayBackendConfig(
            namespace=settings.namespace,
            deployment_profile=settings.profile,
            image_registry_prefix=os.environ.get("FLASHML_IMAGE_REGISTRY_PREFIX", ""),
            sandbox_node_selector=settings.sandbox_node_selector,
            sandbox_runtime_class=settings.sandbox_runtime_class,
            **(
                {"standard_node_selector": settings.standard_node_selector}
                if settings.standard_node_selector is not None
                else {}
            ),
            # Workload pods read the artifact store from the same env contract the
            # service itself uses; pass it through.
            extra_env={
                k: v
                for k, v in os.environ.items()
                if k.startswith("FLASHML_ARTIFACT_") and "SECRET" not in k and "ACCESS" not in k
            },
            env_secret=os.environ.get("FLASHML_ARTIFACT_CREDENTIALS_SECRET") or None,
        )
        backend = KubeRayExecutionBackend(backend_cfg)

    app = FastAPI(title="FlashRuntime", version="0.1.0")
    app.state.watchers = {}

    # -- Mode A: lease coordinator + node registry + local artifacts --------
    artifacts_dir = Path(settings.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    # Durable lease table beside the ledger: tasks, leases, and attempt
    # history survive coordinator restarts (nodes re-register on their own).
    lease_store = SqliteLeaseStore(Path(settings.ledger_path).with_name("leases.db"))
    lease_manager = LeaseManager(store=lease_store, on_event=lambda e: record_event(e))
    modea_state = ModeAState(
        lease_manager,
        artifacts_dir,
        join_code=settings.join_code,
        max_artifact_bytes=settings.max_artifact_mb * 1024 * 1024,
    )
    app.state.modea = modea_state
    app.include_router(modea.build_router(modea_state))
    checkpoint_catalog = CheckpointCatalog(on_event=lambda e: record_event(e))
    app.include_router(checkpoints.build_router(checkpoint_catalog))
    app.include_router(dashboard.build_router())

    def refresh_lease_job(job: JobRecord) -> JobRecord:
        """Lease-mode job status is *derived* from the task table on read."""
        if job.backend != "leases" or job.state.terminal:
            return job
        name, counts = modea.lease_job_state(lease_manager, job.job_id)
        new_state = JobState(name)
        if new_state != job.state:
            job.state = new_state
            if new_state.terminal:
                job.finished_at = utcnow()
                record_event(Event(
                    job_id=job.job_id,
                    type=EventType.JOB_SUCCEEDED if new_state == JobState.SUCCEEDED else EventType.JOB_FAILED,
                    source="flashruntime.leases",
                    message=f"lease job finished: {counts}",
                ))
            ledger.upsert_job(job)
        return job

    async def lease_sweeper() -> None:
        """Expire dead leases and refresh derived job states every 2 s, so
        recovery happens even when no worker is currently claiming."""
        while True:
            try:
                lease_manager.sweep()
                for job in ledger.list_jobs():
                    refresh_lease_job(job)
            except Exception:
                log.exception("lease sweeper iteration failed")
            await asyncio.sleep(2)

    @app.on_event("startup")
    async def _start_sweeper():
        app.state.sweeper = asyncio.create_task(lease_sweeper())

    @app.on_event("shutdown")
    async def _stop_sweeper():
        app.state.sweeper.cancel()

    def record_event(event: Event) -> None:
        ledger.append_event(event)
        log.info(_jlog(event.message or event.type.value, job_id=event.job_id,
                       event=event.type.value, source=event.source))

    async def ingest_workload_artifacts(job: JobRecord) -> None:
        """After success: collect artifact records from the store and fold the
        workload's own attempt/recovery evidence into the ledger."""
        try:
            store = store_from_env()
        except Exception as exc:
            log.warning(_jlog("artifact store unavailable", error=str(exc)))
            return
        prefix = artifact_uri_to_key(
            job.spec.spec.artifacts.outputPrefix.replace("{job_id}", job.job_id)
        )
        record_event(Event(job_id=job.job_id, type=EventType.ARTIFACT_UPLOAD_STARTED,
                           source="flashruntime.service",
                           message=f"collecting artifacts under {prefix}"))
        records = await store.list_prefix(prefix)
        job.artifacts = records
        for record in records:
            record_event(Event(
                job_id=job.job_id, type=EventType.ARTIFACT_COMMITTED,
                source=store.backend, message=f"artifact committed: {record.uri}",
                data={"object_key": record.object_key, "size_bytes": record.size_bytes,
                      "etag": record.etag},
            ))
        # The workload writes execution-summary.json + recovery-events.json
        # with real per-task attempt metadata; fold them into the ledger.
        for name in ("recovery-events.json",):
            key = f"{prefix}{name}"
            if await store.exists(key):
                with tempfile.TemporaryDirectory() as tmp:
                    dest = Path(tmp) / name
                    await store.get_file(key, dest)
                    for raw in json.loads(dest.read_text()):
                        try:
                            record_event(Event.model_validate(raw))
                        except Exception as exc:
                            log.warning(_jlog("bad workload event", error=str(exc)))

    async def watch_job(job: JobRecord) -> None:
        try:
            async for event in backend.stream_events(job.runtime_execution_id):
                record_event(event)
                if event.type == EventType.RAY_WORKER_LOST:
                    job.state = JobState.RECOVERING
                    ledger.upsert_job(job)
                elif event.type == EventType.RAY_WORKER_REPLACED and job.state == JobState.RECOVERING:
                    job.state = JobState.RUNNING
                    ledger.upsert_job(job)
                elif event.type == EventType.JOB_SUCCEEDED:
                    job.state = JobState.SUCCEEDED
                elif event.type == EventType.JOB_FAILED:
                    job.state = JobState.FAILED
                    job.error = event.message
            status = await backend.get_status(job.runtime_execution_id)
            if status.state.terminal:
                job.state = status.state
            job.finished_at = utcnow()
            if job.state == JobState.SUCCEEDED:
                await ingest_workload_artifacts(job)
            ledger.upsert_job(job)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            log.exception("watcher failed")
            job.state = JobState.FAILED
            job.error = f"watcher error: {exc}"
            job.finished_at = utcnow()
            ledger.upsert_job(job)

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok", "profile": settings.profile}

    @app.post("/v1alpha1/jobs", status_code=201)
    async def submit_job(spec: JobSpec):
        # Mode A: expand into leased tasks; no cluster backend involved.
        if spec.spec.execution.backend == "leases":
            job = JobRecord(
                job_id=uuid.uuid4().hex[:12],
                spec=spec,
                backend="leases",
                deployment_profile=settings.profile,
                state=JobState.RUNNING,
            )
            try:
                tasks = modea.expand_tasks(job.job_id, spec)
            except modea.ExpansionError as exc:
                raise HTTPException(status_code=422, detail=str(exc))
            record_event(Event(job_id=job.job_id, type=EventType.JOB_ACCEPTED,
                               source="flashruntime.service",
                               message=f"job '{spec.metadata.name}' accepted"))
            record_event(Event(job_id=job.job_id, type=EventType.BACKEND_SELECTED,
                               source="flashruntime.service",
                               message=f"backend: leases ({len(tasks)} independent tasks)"))
            for task in tasks:
                lease_manager.add_task(task)
            modea_state.lease_jobs.add(job.job_id)
            ledger.upsert_job(job)
            return job

        if backend is None:
            raise HTTPException(
                status_code=503,
                detail="ray backend disabled on this coordinator (FLASHML_ENABLE_KUBERAY=0); "
                       "use execution.backend: leases",
            )
        try:
            await backend.validate(spec)
        except SpecValidationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        job = JobRecord(
            job_id=uuid.uuid4().hex[:12],
            spec=spec,
            backend=backend.name,
            deployment_profile=settings.profile,
        )
        record_event(Event(job_id=job.job_id, type=EventType.JOB_ACCEPTED,
                           source="flashruntime.service",
                           message=f"job '{spec.metadata.name}' accepted"))
        record_event(Event(job_id=job.job_id, type=EventType.BACKEND_SELECTED,
                           source="flashruntime.service",
                           message=f"backend: {backend.name} (profile: {settings.profile})"))
        try:
            execution = await backend.submit(job)
        except Exception as exc:
            job.state = JobState.FAILED
            job.error = str(exc)
            ledger.upsert_job(job)
            raise HTTPException(status_code=502, detail=f"backend submit failed: {exc}")

        job.runtime_execution_id = execution.execution_id
        job.state = JobState.SUBMITTED
        ledger.upsert_job(job)
        record_event(Event(job_id=job.job_id, type=EventType.RAYJOB_CREATED,
                           source="kuberay.rayjob",
                           message=f"RayJob {execution.execution_id} created",
                           data={"execution_id": execution.execution_id,
                                 "namespace": settings.namespace}))
        app.state.watchers[job.job_id] = asyncio.create_task(watch_job(job))
        return job

    @app.get("/v1alpha1/jobs")
    async def list_jobs():
        return [refresh_lease_job(j) for j in ledger.list_jobs()]

    def _job_or_404(job_id: str) -> JobRecord:
        job = ledger.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"no such job: {job_id}")
        return refresh_lease_job(job)

    @app.get("/v1alpha1/jobs/{job_id}")
    async def get_job(job_id: str):
        return _job_or_404(job_id)

    @app.get("/v1alpha1/jobs/{job_id}/events")
    async def get_events(job_id: str):
        _job_or_404(job_id)
        return ledger.events_for(job_id)

    @app.get("/v1alpha1/jobs/{job_id}/logs")
    async def get_logs(job_id: str):
        job = _job_or_404(job_id)
        if not job.runtime_execution_id:
            return {"logs": ""}
        return {"logs": await backend.get_logs(job.runtime_execution_id)}

    @app.post("/v1alpha1/jobs/{job_id}/cancel")
    async def cancel_job(job_id: str):
        job = _job_or_404(job_id)
        if job.state.terminal:
            return job
        if job.backend == "leases":
            for record in lease_manager.records(job_id):
                lease_manager.cancel_task(record.spec.task_id)
        if job.runtime_execution_id:
            await backend.cancel(job.runtime_execution_id)
        watcher = app.state.watchers.pop(job_id, None)
        if watcher:
            watcher.cancel()
        job.state = JobState.CANCELLED
        job.finished_at = utcnow()
        ledger.upsert_job(job)
        return job

    return app


app = create_app() if os.environ.get("FLASHML_SERVICE_AUTOINIT", "1") == "1" else None
