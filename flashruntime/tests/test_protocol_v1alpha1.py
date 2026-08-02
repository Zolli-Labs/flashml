import pytest
from pydantic import ValidationError

from flashruntime.protocol.v1alpha1 import (
    ArtifactRecord,
    Event,
    EventType,
    JobSpec,
    JobState,
)


def make_spec(**overrides):
    base = {
        "apiVersion": "flashml.dev/v1alpha1",
        "kind": "Job",
        "metadata": {"name": "distributed-kmeans-demo"},
        "spec": {
            "image": {"repository": "flashml/kmeans", "tag": "poc-v1"},
            "workload": {
                "type": "sharded_kmeans",
                "parameters": {"samples": 150000, "clusters": 6, "shards": 36},
            },
        },
    }
    base["spec"].update(overrides)
    return base


def test_valid_spec_parses_with_defaults():
    spec = JobSpec.model_validate(make_spec())
    assert spec.spec.execution.backend == "ray"
    assert spec.spec.resources.minimumWorkers == 2
    assert spec.spec.isolation.tier == "standard"
    assert spec.spec.retryPolicy.maxTaskAttempts == 4
    assert spec.spec.artifacts.outputPrefix.startswith("artifact://")


def test_unsupported_backend_rejected():
    with pytest.raises(ValidationError, match="backend"):
        JobSpec.model_validate(make_spec(execution={"backend": "dask"}))


def test_latest_tag_rejected():
    with pytest.raises(ValidationError, match="latest"):
        JobSpec.model_validate(
            make_spec(image={"repository": "flashml/kmeans", "tag": "latest"})
        )


def test_bad_job_name_rejected():
    bad = make_spec()
    bad["metadata"]["name"] = "Bad_Name!"
    with pytest.raises(ValidationError, match="DNS-1123"):
        JobSpec.model_validate(bad)


def test_max_workers_must_cover_min():
    with pytest.raises(ValidationError, match="maximumWorkers"):
        JobSpec.model_validate(
            make_spec(resources={"minimumWorkers": 3, "maximumWorkers": 2})
        )


def test_artifact_prefix_scheme_enforced():
    with pytest.raises(ValidationError, match="artifact://"):
        JobSpec.model_validate(make_spec(artifacts={"outputPrefix": "s3://bucket/x"}))


def test_event_serialization_round_trip():
    ev = Event(
        job_id="abc123",
        type=EventType.RAY_WORKER_LOST,
        source="kubernetes.pod",
        message="worker pod deleted",
        data={"pod": "w-1"},
    )
    restored = Event.model_validate_json(ev.model_dump_json())
    assert restored.type is EventType.RAY_WORKER_LOST
    assert restored.schema_version == "v1alpha1"
    assert restored.data["pod"] == "w-1"


def test_artifact_record_uri_and_metadata():
    record = ArtifactRecord(
        uri="artifact://jobs/abc123/centroids.json",
        backend="minio",
        bucket="flashml-artifacts",
        object_key="jobs/abc123/centroids.json",
        size_bytes=1024,
    )
    assert record.schema_version == "v1alpha1"
    payload = record.model_dump()
    assert payload["backend"] == "minio"


def test_terminal_states():
    assert JobState.SUCCEEDED.terminal
    assert JobState.FAILED.terminal
    assert JobState.CANCELLED.terminal
    assert not JobState.RUNNING.terminal
    assert not JobState.RECOVERING.terminal
