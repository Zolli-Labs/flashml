import pytest

from flashruntime.protocol.v1alpha1 import JobSpec
from flashruntime.service.modea import ExpansionError, expand_tasks


def _spec(**params):
    base = {"round": 0, "num_shards": 3, "local_steps": 5, "lr": 0.05,
            "batch_size": 8, "seed": 0, "in_dim": 4, "hidden": 8,
            "out_dim": 2, "dataset_size": 64}
    base.update(params)
    return JobSpec.model_validate({
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": "fedavg"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": "local/tier1", "tag": "dev"},
            "workload": {"type": "federated_averaging", "parameters": base},
        },
    })


def test_expands_one_task_per_shard():
    tasks = expand_tasks("job-1", _spec())
    assert [t.task_id for t in tasks] == ["shard-000", "shard-001", "shard-002"]


def test_each_task_carries_its_shard_index_and_the_worker_module():
    tasks = expand_tasks("job-1", _spec())
    for i, t in enumerate(tasks):
        assert t.payload["module"] == "flashml_workloads.fedavg_worker"
        assert t.payload["params"]["shard"] == i
        assert t.payload["params"]["num_shards"] == 3


def test_commit_key_is_root_metrics_json():
    tasks = expand_tasks("job-1", _spec())
    assert tasks[0].commit_key == "jobs/job-1/shard-000/metrics.json"


def test_round_zero_declares_no_weights_input():
    tasks = expand_tasks("job-1", _spec(round=0))
    assert "weights" not in tasks[0].payload["inputs"]


def test_later_round_declares_the_weights_artifact():
    tasks = expand_tasks("job-1", _spec(round=2, weights="artifact://jobs/j/r1/weights.json"))
    assert tasks[0].payload["inputs"]["weights"] == "artifact://jobs/j/r1/weights.json"
    assert tasks[0].payload["params"]["round"] == 2


def test_weights_must_be_an_artifact_uri():
    with pytest.raises(ExpansionError, match="artifact://"):
        expand_tasks("job-1", _spec(round=1, weights="/etc/passwd"))


def test_isolation_is_stamped_so_placement_can_fail_closed():
    tasks = expand_tasks("job-1", _spec())
    assert "tier" in tasks[0].payload["isolation"]


def test_rejects_zero_shards():
    with pytest.raises(ExpansionError, match="num_shards"):
        expand_tasks("job-1", _spec(num_shards=0))


def test_rejects_more_than_999_shards():
    """Task ids are zero-padded to 3 digits (shard-000..shard-999) and the
    driver sorts them as strings when collecting a round's results. Above
    999 shards, "shard-1000" < "shard-999" lexically, scrambling the
    participant order and (since float summation isn't associative) making
    the aggregate non-reproducible run to run. Fail closed instead of
    silently widening the padding, which just moves the cliff."""
    with pytest.raises(ExpansionError, match="num_shards"):
        expand_tasks("job-1", _spec(num_shards=1000))


def test_accepts_999_shards_the_upper_boundary():
    tasks = expand_tasks("job-1", _spec(num_shards=999))
    assert len(tasks) == 999
    assert tasks[-1].task_id == "shard-998"


def test_rejects_a_spec_missing_worker_parameters():
    """fedavg_worker reads every one of these unconditionally. Omitting the
    check would defer the failure to a KeyError inside a container on a
    volunteer's machine, burning an attempt and looking like a node fault."""
    spec = _spec()
    del spec.spec.workload.parameters["lr"]
    with pytest.raises(ExpansionError, match="lr"):
        expand_tasks("job-1", spec)
