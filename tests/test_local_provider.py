"""Lightweight version of the conformance checks docs/PROVIDERS.md promises
for every connector, run here against the Local provider."""

import numpy as np
import pytest

import flashruntime
from flashruntime.adapters.base import (
    Offer,
    Provider,
    ProvisionError,
    ResourceSpec,
    Storage,
    Task,
    WorkerPool,
    WorkerSpec,
)
from flashruntime.adapters.registry import get_provider, register


def test_registered_providers_includes_local():
    assert "local" in flashruntime.registered_providers()


def test_lifecycle_provision_and_teardown():
    provider = get_provider("local")
    pool = provider.provision(WorkerSpec(accelerator="cpu-small", entrypoint="tests.test_local_provider:_echo"), count=2)
    assert pool.size == 2
    provider.teardown(pool)
    provider.teardown(pool)  # idempotent, must not raise


def test_provision_rejects_zero_workers():
    provider = get_provider("local")
    with pytest.raises(ProvisionError):
        provider.provision(WorkerSpec(accelerator="cpu-small", entrypoint="tests.test_local_provider:_echo"), count=0)


def _echo(payload, storage):
    return {"echo": payload}


def test_round_trip_echo_task():
    provider = get_provider("local")
    pool = provider.provision(WorkerSpec(accelerator="cpu-small", entrypoint="tests.test_local_provider:_echo"), count=2)
    task = Task(task_id="t1", job_id="j1", payload={"x": 42})
    handle = pool.submit(task)
    [result] = pool.gather([handle], timeout_s=5.0)
    assert result.ok
    assert result.payload == {"echo": {"x": 42}}
    provider.teardown(pool)


def test_storage_round_trip():
    provider = get_provider("local")
    storage = provider.storage()
    storage.put("job1/blob", b"hello")
    assert storage.exists("job1/blob")
    assert storage.get("job1/blob") == b"hello"
    storage.delete_prefix("job1")
    assert not storage.exists("job1/blob")


def test_kmeans_end_to_end_on_local_cluster():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 10]])
    X = np.vstack([rng.normal(loc=c, scale=0.5, size=(50, 2)) for c in centers])

    with flashruntime.Cluster(provider="local", workers=2) as cluster:
        job = cluster.train(
            algorithm=flashruntime.algorithms.KMeans(k=2, n_shards=2, random_state=0),
            dataset=X,
            max_iterations=10,
            convergence_threshold=1e-3,
        )
        result = job.result()

    found = sorted(result["centroids"], key=sum)
    expected = sorted(centers.tolist(), key=sum)
    for f, e in zip(found, expected):
        assert np.linalg.norm(np.array(f) - np.array(e)) < 1.0


def test_job_result_after_stream_does_not_hang():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 2))
    with flashruntime.Cluster(provider="local", workers=2) as cluster:
        job = cluster.train(
            algorithm=flashruntime.algorithms.KMeans(k=2, n_shards=2, random_state=0),
            dataset=X,
            max_iterations=3,
        )
        for _ in job.stream():
            pass
        # Regression test: result() must not re-drain an already-consumed
        # event queue (that previously deadlocked -- see Job.result()).
        job.result()


@register("storage-failure-test")
class _StorageFailureProvider(Provider):
    """Test connector proving provisioned resources are always released."""

    name = "storage-failure-test"
    last_instance = None

    def __init__(self, **_kwargs):
        self.pool = None
        self.torn_down = False
        type(self).last_instance = self

    def offers(self, spec: ResourceSpec) -> list[Offer]:
        return []

    def provision(self, spec: WorkerSpec, count: int) -> WorkerPool:
        provider = get_provider("local")
        self.pool = provider.provision(spec, count)
        return self.pool

    def storage(self) -> Storage:
        raise RuntimeError("volume attachment failed")

    def teardown(self, pool: WorkerPool) -> None:
        get_provider("local").teardown(pool)
        self.torn_down = True


def test_cluster_tears_down_pool_when_storage_setup_fails():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20, 2))

    with pytest.raises(RuntimeError, match="volume attachment failed"):
        with flashruntime.Cluster(provider="storage-failure-test", workers=2) as cluster:
            cluster.train(
                algorithm=flashruntime.algorithms.KMeans(k=2, n_shards=2),
                dataset=X,
            )

    assert _StorageFailureProvider.last_instance.torn_down
