import asyncio

import pytest

from flashruntime.backends.kuberay import (
    SANDBOX_UNAVAILABLE_MSG,
    KubeRayBackendConfig,
    KubeRayExecutionBackend,
    build_rayjob_manifest,
    map_rayjob_status,
    rayjob_name,
)
from flashruntime.backends.base import SpecValidationError
from flashruntime.protocol.v1alpha1 import JobRecord, JobSpec, JobState


def make_job(isolation=None, resources=None) -> JobRecord:
    spec = {
        "apiVersion": "flashml.dev/v1alpha1",
        "kind": "Job",
        "metadata": {"name": "kmeans-demo"},
        "spec": {
            "image": {"repository": "flashml/kmeans", "tag": "poc-v1"},
            "workload": {"type": "sharded_kmeans", "parameters": {"shards": 36}},
            **({"isolation": isolation} if isolation else {}),
            **({"resources": resources} if resources else {}),
        },
    }
    return JobRecord(job_id="abc123", spec=JobSpec.model_validate(spec))


def test_manifest_basics():
    job = make_job()
    m = build_rayjob_manifest(job, KubeRayBackendConfig())
    assert m["kind"] == "RayJob"
    assert m["metadata"]["name"] == rayjob_name("abc123") == "flashml-abc123"
    assert m["metadata"]["labels"]["flashml.dev/job-id"] == "abc123"
    assert m["metadata"]["labels"]["flashml.dev/backend"] == "kuberay"
    assert m["spec"]["entrypoint"] == "python -m flashml_workloads.sharded_kmeans"
    assert m["spec"]["shutdownAfterJobFinishes"] is True


def test_manifest_worker_group_reflects_resources():
    job = make_job(resources={"minimumWorkers": 2, "maximumWorkers": 3})
    m = build_rayjob_manifest(job, KubeRayBackendConfig())
    group = m["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]
    assert group["replicas"] == 3
    assert group["minReplicas"] == 2
    worker_pod = group["template"]["spec"]
    assert worker_pod["nodeSelector"] == {"flashml.dev/compute": "true"}
    assert worker_pod["containers"][0]["image"] == "flashml/kmeans:poc-v1"


def test_manifest_head_schedules_no_tasks():
    m = build_rayjob_manifest(make_job(), KubeRayBackendConfig())
    head = m["spec"]["rayClusterSpec"]["headGroupSpec"]
    assert head["rayStartParams"]["num-cpus"] == "0"


def test_manifest_env_carries_job_context():
    m = build_rayjob_manifest(make_job(), KubeRayBackendConfig(deployment_profile="local"))
    all_env = m["spec"]["rayClusterSpec"]["headGroupSpec"]["template"]["spec"][
        "containers"
    ][0]["env"]
    env = {e["name"]: e["value"] for e in all_env if "value" in e}
    assert env["FLASHML_JOB_ID"] == "abc123"
    assert env["FLASHML_ARTIFACT_PREFIX"] == "artifact://jobs/abc123/"
    assert env["FLASHML_DEPLOYMENT_PROFILE"] == "local"
    # Downward-API identity for task-level node attribution.
    downward = {e["name"] for e in all_env if "valueFrom" in e}
    assert {"K8S_NODE_NAME", "K8S_POD_NAME"} <= downward


def test_sandboxed_manifest_uses_secure_pool_and_runtime_class():
    cfg = KubeRayBackendConfig(
        deployment_profile="alibaba-ack",
        sandbox_node_selector={"flashml.dev/pool": "secure-cloud"},
        sandbox_runtime_class="runv",
    )
    job = make_job(isolation={"tier": "sandboxed", "allowFallback": False})
    m = build_rayjob_manifest(job, cfg)
    worker_pod = m["spec"]["rayClusterSpec"]["workerGroupSpecs"][0]["template"]["spec"]
    assert worker_pod["nodeSelector"] == {"flashml.dev/pool": "secure-cloud"}
    assert worker_pod["runtimeClassName"] == "runv"


def test_local_profile_rejects_sandboxed():
    backend = KubeRayExecutionBackend(KubeRayBackendConfig())  # no sandbox pool
    job = make_job(isolation={"tier": "sandboxed", "allowFallback": False})
    with pytest.raises(SpecValidationError) as excinfo:
        asyncio.run(backend.validate(job.spec))
    assert str(excinfo.value) == SANDBOX_UNAVAILABLE_MSG


def test_sandboxed_with_fallback_allowed_locally():
    backend = KubeRayExecutionBackend(KubeRayBackendConfig())
    job = make_job(isolation={"tier": "sandboxed", "allowFallback": True})
    asyncio.run(backend.validate(job.spec))  # must not raise


@pytest.mark.parametrize(
    "status,expected",
    [
        ({}, JobState.SUBMITTED),
        ({"jobDeploymentStatus": "Initializing"}, JobState.SUBMITTED),
        ({"jobDeploymentStatus": "Running", "jobStatus": "PENDING"}, JobState.SUBMITTED),
        ({"jobDeploymentStatus": "Running", "jobStatus": "RUNNING"}, JobState.RUNNING),
        ({"jobDeploymentStatus": "Complete", "jobStatus": "SUCCEEDED"}, JobState.SUCCEEDED),
        ({"jobDeploymentStatus": "Failed", "jobStatus": "FAILED"}, JobState.FAILED),
        ({"jobStatus": "STOPPED"}, JobState.CANCELLED),
    ],
)
def test_rayjob_status_mapping(status, expected):
    assert map_rayjob_status(status).state is expected
