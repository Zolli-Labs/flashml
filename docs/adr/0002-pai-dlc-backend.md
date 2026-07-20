# ADR 0002: PAI-DLC as a second ExecutionBackend (boundary definition)

Status: accepted (interface boundary only — deliberately not implemented)
Date: 2026-07-17

## Context

The POC's first backend is KubeRay (`KubeRayExecutionBackend`), running Ray
on Kubernetes that we operate (Kind locally, ACK in the cloud). Alibaba PAI
also offers DLC (Deep Learning Containers): a managed job service that runs
Ray/PyTorch/MPI/custom jobs on PAI-managed resources, with ACR custom
images, OSS I/O, preemptible capacity, and AIMaster-based fault tolerance
for supported job types.

## Decision

Define — but do not implement — `PAIDLCExecutionBackend` conforming to
`flashruntime.backends.base.ExecutionBackend`:

- `validate`: check the JobSpec against DLC-supported job shapes; reject
  `isolation.tier: sandboxed` (DLC has its own isolation model — we do not
  translate the sandbox tier to it).
- `submit`: create a DLC job via the Alibaba SDK; image = ACR reference
  derived from the JobSpec + deployment registry prefix; data I/O bound to
  OSS through deployment configuration; record the returned DLC job ID as
  `runtime_execution_id`.
- `get_status`: map DLC states (Creating/Queuing/Running/Succeeded/Failed/
  Stopped) onto FlashRuntime `JobState`.
- `stream_events` / `get_logs`: poll DLC events and log APIs; normalize into
  the same `Event` vocabulary (worker loss and retries only where DLC
  actually reports them — no fabricated recovery evidence).
- `collect_artifacts`: unchanged — the workload writes to the ArtifactStore
  (OSS) itself.

Honesty constraints:

- **AIMaster fault-tolerance is documented for specific job types; we do not
  claim it supports Ray jobs.** A future PyTorch workload is the natural
  fit for AIMaster monitoring, computing-power health checks, preemptible
  resources, and OSS checkpoints.
- KubeRay remains the reference backend; PAI-DLC must never become a
  prerequisite for the open-source runtime.

## Implementation gate

Implement only after the KubeRay POC passes end-to-end (it has), credentials
are available, and then only as one narrow experiment: submit one simple DLC
job, record the job ID, poll status, retrieve OSS output. Everything beyond
that is out of scope until the protocol (leases/checkpoints) lands.
