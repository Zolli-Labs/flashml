"""FlashML public protocol, version v1alpha1.

Every wire-visible model in the FlashML system lives here: the JobSpec users
submit, the job/event/artifact records FlashML Cloud stores and displays, and
the node registration/heartbeat messages FlashNode sends. flashnode and
flashml-cloud import these — they never define their own copies.

Versioning: `apiVersion` on the JobSpec and `schema_version` on wire messages
identify this revision. Additive changes are allowed within v1alpha1;
breaking changes require a new module (v1alpha2, ...).

Deployment-private details (ACK cluster IDs, ACR credentials, OSS keys, SLS
projects, Kubernetes namespaces, RuntimeClass names) are deliberately absent:
they belong to backend/deployment configuration, never the public spec.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

API_VERSION = "flashml.dev/v1alpha1"
SCHEMA_VERSION = "v1alpha1"

# Label keys stamped on every backend resource created for a job.
LABEL_JOB_ID = "flashml.dev/job-id"
LABEL_RUNTIME_ID = "flashml.dev/runtime-id"
LABEL_BACKEND = "flashml.dev/backend"
LABEL_PROFILE = "flashml.dev/deployment-profile"

_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# JobSpec
# ---------------------------------------------------------------------------


class ExecutionSpec(BaseModel):
    # "ray": coordinated execution on a managed pool (KubeRay backend).
    # "leases": Mode A — the job expands into independent tasks that
    #           FlashNode workers pull, heartbeat, and commit (additive, July 2026).
    backend: Literal["ray", "leases"] = "ray"
    environment: Literal["auto", "local", "alibaba-ack"] = "auto"


class ImageSpec(BaseModel):
    repository: str
    tag: str

    @field_validator("tag")
    @classmethod
    def _no_latest(cls, v: str) -> str:
        if v == "latest" or not v:
            raise ValueError("image tag must be pinned; 'latest' is not allowed")
        return v

    @property
    def reference(self) -> str:
        return f"{self.repository}:{self.tag}"


class WorkloadSpec(BaseModel):
    type: str
    parameters: dict[str, Any] = Field(default_factory=dict)


class ResourcesSpec(BaseModel):
    minimumWorkers: int = Field(ge=1, default=2)
    maximumWorkers: int = Field(ge=1, default=3)
    cpuPerTask: float = Field(gt=0, default=1)
    memoryPerTask: str = "512Mi"

    @field_validator("maximumWorkers")
    @classmethod
    def _max_ge_min(cls, v: int, info: Any) -> int:
        minimum = info.data.get("minimumWorkers")
        if minimum is not None and v < minimum:
            raise ValueError("maximumWorkers must be >= minimumWorkers")
        return v


class PlacementSpec(BaseModel):
    pool: Literal["any", "local", "edge", "standard-cloud", "secure-cloud"] = "any"
    architectures: list[Literal["amd64", "arm64"]] = Field(default_factory=lambda: ["amd64"])


class IsolationSpec(BaseModel):
    tier: Literal["standard", "sandboxed"] = "standard"
    allowFallback: bool = False


class RetryPolicySpec(BaseModel):
    maxTaskAttempts: int = Field(ge=1, default=4)
    retryWorkerLoss: bool = True


class ArtifactsSpec(BaseModel):
    # Backend-neutral prefix; "{job_id}" is substituted at submission time.
    outputPrefix: str = "artifact://jobs/{job_id}/"

    @field_validator("outputPrefix")
    @classmethod
    def _scheme(cls, v: str) -> str:
        if not v.startswith("artifact://"):
            raise ValueError("outputPrefix must use the artifact:// scheme")
        return v


class JobMetadata(BaseModel):
    name: str
    labels: dict[str, str] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def _dns_name(cls, v: str) -> str:
        if not _NAME_RE.match(v):
            raise ValueError(
                "job name must be a lowercase DNS-1123 label (a-z, 0-9, '-', max 63 chars)"
            )
        return v


class JobSpecInner(BaseModel):
    execution: ExecutionSpec = Field(default_factory=ExecutionSpec)
    image: ImageSpec
    workload: WorkloadSpec
    resources: ResourcesSpec = Field(default_factory=ResourcesSpec)
    placement: PlacementSpec = Field(default_factory=PlacementSpec)
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    retryPolicy: RetryPolicySpec = Field(default_factory=RetryPolicySpec)
    artifacts: ArtifactsSpec = Field(default_factory=ArtifactsSpec)


class JobSpec(BaseModel):
    apiVersion: Literal["flashml.dev/v1alpha1"] = API_VERSION
    kind: Literal["Job"] = "Job"
    metadata: JobMetadata
    spec: JobSpecInner


# ---------------------------------------------------------------------------
# Job state and events
# ---------------------------------------------------------------------------


class JobState(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    RUNNING = "RUNNING"
    RECOVERING = "RECOVERING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"

    @property
    def terminal(self) -> bool:
        return self in (JobState.SUCCEEDED, JobState.FAILED, JobState.CANCELLED)


class EventType(str, Enum):
    JOB_ACCEPTED = "JOB_ACCEPTED"
    BACKEND_SELECTED = "BACKEND_SELECTED"
    RAYJOB_CREATED = "RAYJOB_CREATED"
    RAY_CLUSTER_STARTING = "RAY_CLUSTER_STARTING"
    NODE_REGISTERED = "NODE_REGISTERED"
    NODE_READY = "NODE_READY"
    TASK_ATTEMPT_STARTED = "TASK_ATTEMPT_STARTED"
    TASK_ATTEMPT_COMPLETED = "TASK_ATTEMPT_COMPLETED"
    NODE_HEARTBEAT_LOST = "NODE_HEARTBEAT_LOST"
    RAY_WORKER_LOST = "RAY_WORKER_LOST"
    TASK_ATTEMPT_RETRIED = "TASK_ATTEMPT_RETRIED"
    RAY_WORKER_REPLACED = "RAY_WORKER_REPLACED"
    ITERATION_COMPLETED = "ITERATION_COMPLETED"
    ARTIFACT_UPLOAD_STARTED = "ARTIFACT_UPLOAD_STARTED"
    ARTIFACT_COMMITTED = "ARTIFACT_COMMITTED"
    JOB_SUCCEEDED = "JOB_SUCCEEDED"
    JOB_FAILED = "JOB_FAILED"
    # Mode A lease lifecycle (additive, July 2026)
    TASK_CREATED = "TASK_CREATED"
    LEASE_CLAIMED = "LEASE_CLAIMED"
    LEASE_RENEWED = "LEASE_RENEWED"
    LEASE_EXPIRED = "LEASE_EXPIRED"
    TASK_REQUEUED = "TASK_REQUEUED"
    TASK_COMMIT_ACCEPTED = "TASK_COMMIT_ACCEPTED"
    TASK_COMMIT_REJECTED = "TASK_COMMIT_REJECTED"
    TASK_ATTEMPT_FAILED = "TASK_ATTEMPT_FAILED"
    TASK_EXHAUSTED = "TASK_EXHAUSTED"
    # Checkpoint catalog lifecycle (additive, July 2026)
    CHECKPOINT_PARTS_REGISTERED = "CHECKPOINT_PARTS_REGISTERED"
    CHECKPOINT_MANIFEST_COMMITTED = "CHECKPOINT_MANIFEST_COMMITTED"
    CHECKPOINT_RESTORE_VERIFIED = "CHECKPOINT_RESTORE_VERIFIED"
    CHECKPOINT_REJECTED = "CHECKPOINT_REJECTED"
    # Recovery policy (additive, July 2026)
    FAILURE_CLASSIFIED = "FAILURE_CLASSIFIED"
    RECOVERY_ACTION_SELECTED = "RECOVERY_ACTION_SELECTED"
    RECOVERY_FROZEN = "RECOVERY_FROZEN"


class Event(BaseModel):
    """One entry in the FlashRuntime event ledger.

    `source` names the raw signal the event was derived from (e.g.
    "kubernetes.pod", "ray.job-log", "workload.summary", "flashnode") so
    recovery evidence is traceable, never fabricated.
    """

    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    job_id: str
    type: EventType
    timestamp: datetime = Field(default_factory=utcnow)
    source: str
    message: str = ""
    data: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Artifacts
# ---------------------------------------------------------------------------


class ArtifactRecord(BaseModel):
    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    uri: str  # backend-neutral, e.g. artifact://jobs/<job-id>/centroids.json
    backend: Literal["minio", "oss", "local"]
    bucket: str
    object_key: str
    etag: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    created_at: datetime = Field(default_factory=utcnow)


# ---------------------------------------------------------------------------
# Job record (runtime-owned view served to FlashML Cloud)
# ---------------------------------------------------------------------------


class JobRecord(BaseModel):
    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    job_id: str
    spec: JobSpec
    state: JobState = JobState.PENDING
    backend: str = "kuberay"
    deployment_profile: str = "local"
    runtime_execution_id: str | None = None  # e.g. the RayJob name
    created_at: datetime = Field(default_factory=utcnow)
    finished_at: datetime | None = None
    error: str | None = None
    artifacts: list[ArtifactRecord] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# FlashNode <-> FlashML Cloud
# ---------------------------------------------------------------------------


class NodeEnvironment(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    EDGE = "edge"


class NodeCapabilities(BaseModel):
    cpu_cores: float | None = None
    memory_bytes: int | None = None
    gpus: list[dict[str, Any]] = Field(default_factory=list)
    os: str = ""
    architecture: str = ""


class NodeRegistration(BaseModel):
    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    node_id: str
    kubernetes_node: str
    hostname: str
    capabilities: NodeCapabilities
    environment: NodeEnvironment = NodeEnvironment.LOCAL
    sandbox_capable: bool = False
    #: This node runs an argv-capable sandboxed runner. Defaults False so
    #: every already-deployed agent is excluded from argv work until it is
    #: upgraded and explicitly opted in (security fields fail closed).
    argv_capable: bool = False
    pool: str = "local"
    runtime_profile: str = "kubernetes"
    labels: dict[str, str] = Field(default_factory=dict)
    agent_version: str = ""


class NodeHeartbeat(BaseModel):
    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    node_id: str
    timestamp: datetime = Field(default_factory=utcnow)
    status: Literal["online", "draining", "terminating"] = "online"


class NodeStatusView(BaseModel):
    """Cloud-side view of a node, derived from registration + heartbeats."""

    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    registration: NodeRegistration
    online: bool
    last_heartbeat: datetime | None = None
    accepted_task_count: int = 0


# ---------------------------------------------------------------------------
# Mode A: tasks, leases, attempts (additive, July 2026)
#
# The lease pattern: work is never pushed. A worker *claims* a time-bounded
# lease on a task, renews it with heartbeats, and only the first attempt to
# commit a valid result wins — late duplicates are rejected. A dead worker
# needs no handling: its lease expires and the task requeues.
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    PENDING = "PENDING"  # claimable
    LEASED = "LEASED"  # one live lease holds it
    COMPLETED = "COMPLETED"  # a commit was accepted (terminal)
    FAILED = "FAILED"  # attempts exhausted (terminal)
    CANCELLED = "CANCELLED"  # withdrawn by the job (terminal)


class TaskSpec(BaseModel):
    """One unit of independent work. `payload` is workload-defined (e.g. a
    hyperparameter trial config); `commit_key` is the idempotency anchor —
    exactly one accepted output may ever exist under it."""

    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    task_id: str
    job_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
    commit_key: str
    max_attempts: int = Field(default=3, ge=1)
    lease_seconds: float = Field(default=60.0, gt=0)


class Lease(BaseModel):
    """A time-bounded right to execute one attempt of one task."""

    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    lease_id: str
    task_id: str
    job_id: str
    node_id: str
    attempt_number: int
    deadline: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class TaskAttempt(BaseModel):
    """One execution try. `accepted` flips true only on the winning commit."""

    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    attempt_id: str
    task_id: str
    job_id: str
    node_id: str
    attempt_number: int
    started: datetime
    finished: datetime | None = None
    outcome: Literal["running", "committed", "rejected", "failed", "expired"] = "running"
    output_sha256: str | None = None
    accepted: bool = False


# ---------------------------------------------------------------------------
# Checkpoint manifests (additive, July 2026)
#
# A checkpoint is not a path — it is a *manifest* proving completeness and
# compatibility. Parts upload first; the manifest is written last, only
# after every expected part's hash verifies. No manifest ⇒ no checkpoint.
# ---------------------------------------------------------------------------


class CheckpointPart(BaseModel):
    key: str  # storage key relative to the manifest's prefix
    sha256: str
    size_bytes: int = Field(ge=0)


class CheckpointValidation(str, Enum):
    HASH_VERIFIED = "hash_verified"  # all parts present, hashes match
    RESTORE_VERIFIED = "restore_verified"  # a real load succeeded from this manifest
    INVALID = "invalid"  # quarantined; recovery must never select it


class CheckpointManifest(BaseModel):
    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    manifest_id: str
    job_id: str
    attempt_id: str
    step: int = Field(ge=0)
    framework: str = ""  # e.g. "pytorch-2.9"
    strategy_family: str = ""  # e.g. "ddp", "fsdp2"
    world_size: int = Field(default=1, ge=1)
    compatible_world_sizes: list[int] = Field(default_factory=list)  # via resharding
    storage_prefix: str  # e.g. "artifact://checkpoints/<job>/<attempt>/<step>/"
    parts: list[CheckpointPart]
    validation: CheckpointValidation
    created: datetime = Field(default_factory=utcnow)
    checkpoint_duration_s: float | None = None


# ---------------------------------------------------------------------------
# Failure taxonomy and recovery decisions (additive, July 2026)
# ---------------------------------------------------------------------------


class FailureClass(str, Enum):
    APPLICATION_ERROR = "application_error"
    DATA_ERROR = "data_error"
    WORKER_CRASH = "worker_crash"
    NODE_LOSS = "node_loss"
    ACCELERATOR_FAILURE = "accelerator_failure"
    COMMUNICATION_ERROR = "communication_error"  # NCCL/RCCL/rendezvous
    NETWORK_DEGRADATION = "network_degradation"
    STORAGE_TIMEOUT = "storage_timeout"
    ARTIFACT_CORRUPTION = "artifact_corruption"
    PREEMPTION = "preemption"
    CORRELATED_INCIDENT = "correlated_incident"
    CONTROL_PLANE_FAILURE = "control_plane_failure"
    UNKNOWN = "unknown"


class RecoveryActionType(str, Enum):
    FAIL_JOB = "fail_job"  # deterministic app error: fail fast, tell the user
    RETRY_TASK = "retry_task"  # Mode A: requeue the attempt elsewhere
    RESTART_GROUP = "restart_group"  # Mode B: whole-group restart from checkpoint
    REPLACE_NODE = "replace_node"  # cordon + acquire replacement, then restart/requeue
    PAUSE_JOB = "pause_job"  # e.g. storage outage: protect state, stop burning compute
    FREEZE_AUTOMATION = "freeze_automation"  # correlated incident: no retry storms; escalate


class RecoveryDecision(BaseModel):
    """One typed, logged recovery decision. `policy_version` + inputs make it
    reproducible; there is no discretionary path."""

    schema_version: Literal["v1alpha1"] = SCHEMA_VERSION
    policy_version: str
    failure_class: FailureClass
    action: RecoveryActionType
    scope: Literal["task", "node", "group", "job", "pool"]
    cordon_node: bool = False
    needs_checkpoint: bool = False
    reason: str
    evidence: dict[str, Any] = Field(default_factory=dict)
    decided_at: datetime = Field(default_factory=utcnow)
