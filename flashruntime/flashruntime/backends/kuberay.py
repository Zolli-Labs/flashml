"""KubeRay execution backend: FlashRuntime JobSpec -> RayJob custom resource.

Responsibilities (and nothing more):
- validate a JobSpec against this deployment profile;
- render the RayJob manifest (`build_rayjob_manifest` is pure and unit-tested
  without a cluster);
- create/read/delete the RayJob via the Kubernetes API;
- map RayJob + pod signals onto FlashRuntime `JobState` and `Event`s.

Ray task scheduling, pod lifecycle, and image building belong to Ray,
KubeRay, and the build pipeline respectively — never to this module.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, AsyncIterator

from flashruntime.backends.base import (
    BackendExecution,
    BackendStatus,
    BackendUnavailableError,
    SpecValidationError,
)
from flashruntime.protocol.v1alpha1 import (
    LABEL_BACKEND,
    LABEL_JOB_ID,
    LABEL_PROFILE,
    LABEL_RUNTIME_ID,
    ArtifactRecord,
    Event,
    EventType,
    JobRecord,
    JobSpec,
    JobState,
)

RAYJOB_GROUP = "ray.io"
RAYJOB_VERSION = "v1"
RAYJOB_PLURAL = "rayjobs"

SANDBOX_UNAVAILABLE_MSG = (
    "Sandboxed execution is unavailable in the local profile. "
    "Use a compatible ACK secure node pool."
)


@dataclass
class KubeRayBackendConfig:
    """Deployment-side configuration. Everything cloud- or cluster-specific
    (namespaces, node selectors, RuntimeClass names, secret names) lives here,
    never in the public JobSpec."""

    namespace: str = "flashml"
    deployment_profile: str = "local"  # "local" | "alibaba-ack"
    ray_version: str = "2.46.0"
    # Prepended to the JobSpec's repository:tag — "localhost:5001/" for the
    # local Kind registry, the ACR endpoint + namespace on Alibaba. Keeps
    # registry endpoints out of the public JobSpec.
    image_registry_prefix: str = ""
    # Node selection for standard workloads (the simulated compute workers
    # locally; the standard node pool on ACK).
    standard_node_selector: dict[str, str] = field(
        default_factory=lambda: {"flashml.dev/compute": "true"}
    )
    # Secure pool wiring; empty means the profile has no sandbox tier.
    sandbox_node_selector: dict[str, str] = field(default_factory=dict)
    sandbox_runtime_class: str | None = None
    sandbox_tolerations: list[dict[str, Any]] = field(default_factory=list)
    # Plain env vars injected into head/worker containers (artifact endpoint,
    # bucket, deployment profile, ...). Secret-backed vars go via env_secret.
    extra_env: dict[str, str] = field(default_factory=dict)
    env_secret: str | None = None  # name of a Secret mounted via envFrom
    ttl_seconds_after_finished: int = 600
    active_deadline_seconds: int = 1800
    image_pull_policy: str = "IfNotPresent"
    head_memory: str = "1536Mi"
    worker_memory: str = "1536Mi"
    worker_cpu: str = "1"


def _workload_entrypoint(job: JobRecord) -> str:
    """The container image owns the workload code; the entrypoint convention
    is `python -m flashml_workloads.<type>`. Parameters travel as one JSON
    env var so the entrypoint stays stable across workloads."""
    workload = job.spec.spec.workload
    return f"python -m flashml_workloads.{workload.type}"


def build_rayjob_manifest(job: JobRecord, cfg: KubeRayBackendConfig) -> dict[str, Any]:
    """Pure translation of a FlashRuntime job to a RayJob manifest."""
    spec = job.spec.spec
    labels = {
        LABEL_JOB_ID: job.job_id,
        LABEL_RUNTIME_ID: job.runtime_execution_id or rayjob_name(job.job_id),
        LABEL_BACKEND: "kuberay",
        LABEL_PROFILE: cfg.deployment_profile,
    }

    env = [
        {"name": "FLASHML_JOB_ID", "value": job.job_id},
        {"name": "FLASHML_DEPLOYMENT_PROFILE", "value": cfg.deployment_profile},
        {"name": "FLASHML_WORKLOAD_PARAMS", "value": json.dumps(spec.workload.parameters)},
        {"name": "FLASHML_MAX_TASK_ATTEMPTS", "value": str(spec.retryPolicy.maxTaskAttempts)},
        {"name": "FLASHML_ARTIFACT_PREFIX", "value": spec.artifacts.outputPrefix.replace(
            "{job_id}", job.job_id
        )},
    ] + [{"name": k, "value": v} for k, v in sorted(cfg.extra_env.items())]

    env_from = (
        [{"secretRef": {"name": cfg.env_secret}}] if cfg.env_secret else []
    )

    if spec.isolation.tier == "sandboxed":
        node_selector = dict(cfg.sandbox_node_selector)
        runtime_class = cfg.sandbox_runtime_class
        tolerations = list(cfg.sandbox_tolerations)
    else:
        node_selector = dict(cfg.standard_node_selector)
        runtime_class = None
        tolerations = []

    # Downward-API identity so Ray tasks can report which Kubernetes node
    # and pod they actually ran on (recovery evidence, node contributions).
    downward_env = [
        {"name": "K8S_NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
        {"name": "K8S_POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
    ]

    def pod_template(cpu: str, memory: str, is_head: bool) -> dict[str, Any]:
        pod_spec: dict[str, Any] = {
            "containers": [
                {
                    "name": "ray-head" if is_head else "ray-worker",
                    "image": f"{cfg.image_registry_prefix}{spec.image.reference}",
                    "imagePullPolicy": cfg.image_pull_policy,
                    "env": env + downward_env,
                    **({"envFrom": env_from} if env_from else {}),
                    "resources": {
                        "requests": {"cpu": cpu, "memory": memory},
                        "limits": {"cpu": cpu, "memory": memory},
                    },
                }
            ],
        }
        if not is_head:
            # Spread workers across nodes (preferred, not required) so the
            # demo genuinely exercises multiple machines.
            pod_spec["affinity"] = {
                "podAntiAffinity": {
                    "preferredDuringSchedulingIgnoredDuringExecution": [
                        {
                            "weight": 100,
                            "podAffinityTerm": {
                                "topologyKey": "kubernetes.io/hostname",
                                "labelSelector": {
                                    "matchLabels": {LABEL_JOB_ID: job.job_id}
                                },
                            },
                        }
                    ]
                }
            }
            if node_selector:
                pod_spec["nodeSelector"] = node_selector
            if runtime_class:
                pod_spec["runtimeClassName"] = runtime_class
            if tolerations:
                pod_spec["tolerations"] = tolerations
        return {"metadata": {"labels": labels}, "spec": pod_spec}

    return {
        "apiVersion": f"{RAYJOB_GROUP}/{RAYJOB_VERSION}",
        "kind": "RayJob",
        "metadata": {
            "name": job.runtime_execution_id or rayjob_name(job.job_id),
            "namespace": cfg.namespace,
            "labels": labels,
        },
        "spec": {
            "entrypoint": _workload_entrypoint(job),
            "shutdownAfterJobFinishes": True,
            "ttlSecondsAfterFinished": cfg.ttl_seconds_after_finished,
            "activeDeadlineSeconds": cfg.active_deadline_seconds,
            "rayClusterSpec": {
                "rayVersion": cfg.ray_version,
                "headGroupSpec": {
                    # Head schedules no tasks (num-cpus: 0): compute happens
                    # only on the worker group, i.e. the simulated devices.
                    "rayStartParams": {"num-cpus": "0"},
                    "template": pod_template("1", cfg.head_memory, is_head=True),
                },
                "workerGroupSpecs": [
                    {
                        "groupName": "workers",
                        "replicas": spec.resources.maximumWorkers,
                        "minReplicas": spec.resources.minimumWorkers,
                        "maxReplicas": spec.resources.maximumWorkers,
                        "rayStartParams": {},
                        "template": pod_template(
                            cfg.worker_cpu, cfg.worker_memory, is_head=False
                        ),
                    }
                ],
            },
        },
    }


def rayjob_name(job_id: str) -> str:
    return f"flashml-{job_id}"


def map_rayjob_status(status: dict[str, Any]) -> BackendStatus:
    """Map RayJob .status onto FlashRuntime JobState.

    KubeRay reports two axes: jobDeploymentStatus (cluster/provisioning) and
    jobStatus (the Ray job itself). jobStatus wins once present.
    """
    deployment = status.get("jobDeploymentStatus", "")
    ray_status = status.get("jobStatus", "")

    if ray_status == "SUCCEEDED":
        state = JobState.SUCCEEDED
    elif ray_status == "FAILED" or deployment == "Failed":
        state = JobState.FAILED
    elif ray_status == "STOPPED":
        state = JobState.CANCELLED
    elif ray_status == "RUNNING":
        state = JobState.RUNNING
    elif ray_status == "PENDING":
        state = JobState.SUBMITTED
    elif deployment in ("Initializing", "Waiting", ""):
        state = JobState.SUBMITTED
    elif deployment in ("Running", "Complete"):
        # Deployment running but Ray job not started/reported yet.
        state = JobState.RUNNING if deployment == "Running" else JobState.SUCCEEDED
    else:
        state = JobState.SUBMITTED

    reason = status.get("message", "") or f"deployment={deployment} job={ray_status}"
    return BackendStatus(state=state, reason=reason, raw=dict(status))


class KubeRayExecutionBackend:
    """ExecutionBackend implementation backed by the KubeRay operator.

    Uses the official kubernetes client (sync) behind asyncio.to_thread —
    call volume is tiny (submit/poll/logs), so thread offloading beats an
    extra async client dependency.
    """

    name = "kuberay"

    def __init__(self, cfg: KubeRayBackendConfig | None = None):
        self.cfg = cfg or KubeRayBackendConfig()
        self._api = None  # lazily constructed CustomObjectsApi
        self._core = None  # lazily constructed CoreV1Api

    # -- kubernetes client plumbing ----------------------------------------

    def _ensure_clients(self):
        if self._api is None:
            from kubernetes import client, config as kube_config

            try:
                kube_config.load_incluster_config()
            except Exception:
                try:
                    kube_config.load_kube_config()
                except Exception as exc:  # pragma: no cover
                    raise BackendUnavailableError(
                        f"no Kubernetes configuration available: {exc}"
                    ) from exc
            self._api = client.CustomObjectsApi()
            self._core = client.CoreV1Api()
        return self._api, self._core

    # -- ExecutionBackend --------------------------------------------------

    async def validate(self, spec: JobSpec) -> None:
        if spec.spec.execution.backend != "ray":
            raise SpecValidationError(
                f"backend '{spec.spec.execution.backend}' is not supported; "
                "supported backends: ray"
            )
        if spec.spec.isolation.tier == "sandboxed" and not self.cfg.sandbox_node_selector:
            if spec.spec.isolation.allowFallback:
                return  # falls back to standard tier at submit time
            raise SpecValidationError(SANDBOX_UNAVAILABLE_MSG)

    async def submit(self, job: JobRecord) -> BackendExecution:
        api, _ = self._ensure_clients()
        name = rayjob_name(job.job_id)
        job.runtime_execution_id = name
        manifest = build_rayjob_manifest(job, self.cfg)

        def _create():
            return api.create_namespaced_custom_object(
                RAYJOB_GROUP, RAYJOB_VERSION, self.cfg.namespace, RAYJOB_PLURAL, manifest
            )

        await asyncio.to_thread(_create)
        return BackendExecution(
            execution_id=name,
            backend=self.name,
            submitted_at=datetime.now(timezone.utc),
            details={"namespace": self.cfg.namespace, "manifest": manifest},
        )

    async def get_status(self, execution_id: str) -> BackendStatus:
        api, _ = self._ensure_clients()

        def _get():
            return api.get_namespaced_custom_object(
                RAYJOB_GROUP, RAYJOB_VERSION, self.cfg.namespace, RAYJOB_PLURAL, execution_id
            )

        obj = await asyncio.to_thread(_get)
        return map_rayjob_status(obj.get("status", {}) or {})

    async def stream_events(self, execution_id: str) -> AsyncIterator[Event]:
        """Poll RayJob status + pod events and yield normalized FlashRuntime
        events until the job reaches a terminal state. Every event carries its
        raw source so recovery evidence stays honest."""
        job_id = execution_id.removeprefix("flashml-")
        seen_pods: dict[str, str] = {}  # pod name -> phase
        seen_event_uids: set[str] = set()
        emitted_cluster_starting = False
        last_state: JobState | None = None

        while True:
            status = await self.get_status(execution_id)
            if not emitted_cluster_starting:
                emitted_cluster_starting = True
                yield Event(
                    job_id=job_id,
                    type=EventType.RAY_CLUSTER_STARTING,
                    source="kuberay.rayjob",
                    message="RayJob accepted by KubeRay; ephemeral Ray cluster starting",
                    data={"execution_id": execution_id},
                )

            # Pod lifecycle -> worker lost/replaced events.
            async for ev in self._pod_events(execution_id, job_id, seen_pods, seen_event_uids):
                yield ev

            if status.state != last_state:
                last_state = status.state
                if status.state == JobState.SUCCEEDED:
                    yield Event(
                        job_id=job_id,
                        type=EventType.JOB_SUCCEEDED,
                        source="kuberay.rayjob",
                        message=status.reason,
                        data=status.raw,
                    )
                    return
                if status.state == JobState.FAILED:
                    yield Event(
                        job_id=job_id,
                        type=EventType.JOB_FAILED,
                        source="kuberay.rayjob",
                        message=status.reason,
                        data=status.raw,
                    )
                    return
            elif status.state.terminal:
                return
            await asyncio.sleep(2)

    async def _pod_events(
        self,
        execution_id: str,
        job_id: str,
        seen_pods: dict[str, str],
        seen_event_uids: set[str],
    ) -> AsyncIterator[Event]:
        _, core = self._ensure_clients()
        selector = f"{LABEL_JOB_ID}={job_id}"

        def _list():
            return core.list_namespaced_pod(self.cfg.namespace, label_selector=selector)

        pods = await asyncio.to_thread(_list)
        current = {}
        for pod in pods.items:
            name = pod.metadata.name
            phase = pod.status.phase or "Unknown"
            deleting = pod.metadata.deletion_timestamp is not None
            current[name] = phase
            is_worker = "workers" in name
            if name not in seen_pods:
                if is_worker and seen_pods:
                    # A worker appearing after startup means KubeRay replaced
                    # a lost one (group replicas are fixed in the POC).
                    prior_workers = [p for p in seen_pods if "workers" in p]
                    if any(p not in current for p in prior_workers):
                        yield Event(
                            job_id=job_id,
                            type=EventType.RAY_WORKER_REPLACED,
                            source="kubernetes.pod",
                            message=f"replacement Ray worker pod {name} created",
                            data={"pod": name, "phase": phase},
                        )
            elif deleting and seen_pods.get(name) != "Deleting" and is_worker:
                current[name] = "Deleting"
                yield Event(
                    job_id=job_id,
                    type=EventType.RAY_WORKER_LOST,
                    source="kubernetes.pod",
                    message=f"Ray worker pod {name} is terminating",
                    data={"pod": name},
                )
        for name in list(seen_pods):
            if name not in current and "workers" in name:
                if seen_pods[name] != "Deleting":
                    yield Event(
                        job_id=job_id,
                        type=EventType.RAY_WORKER_LOST,
                        source="kubernetes.pod",
                        message=f"Ray worker pod {name} disappeared",
                        data={"pod": name},
                    )
        seen_pods.clear()
        seen_pods.update(current)

    async def get_logs(self, execution_id: str) -> str:
        """Logs of the RayJob submitter pod (driver output)."""
        _, core = self._ensure_clients()

        def _logs() -> str:
            pods = core.list_namespaced_pod(
                self.cfg.namespace, label_selector=f"job-name={execution_id}"
            )
            chunks = []
            for pod in pods.items:
                try:
                    chunks.append(
                        core.read_namespaced_pod_log(pod.metadata.name, self.cfg.namespace)
                    )
                except Exception as exc:  # pod may be pending
                    chunks.append(f"[no logs from {pod.metadata.name}: {exc}]")
            return "\n".join(chunks)

        return await asyncio.to_thread(_logs)

    async def cancel(self, execution_id: str) -> None:
        api, _ = self._ensure_clients()

        def _delete():
            api.delete_namespaced_custom_object(
                RAYJOB_GROUP, RAYJOB_VERSION, self.cfg.namespace, RAYJOB_PLURAL, execution_id
            )

        await asyncio.to_thread(_delete)

    async def collect_artifacts(self, execution_id: str) -> list[ArtifactRecord]:
        """Artifact records are written by the workload itself through the
        ArtifactStore; the runtime service collects them from the store by
        prefix (see flashruntime.artifacts). The backend has nothing to add."""
        return []
