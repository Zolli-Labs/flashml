import ast
import inspect

import pytest

from flashruntime.protocol.v1alpha1 import JobSpec
from flashruntime.service.modea import (
    FEDAVG_DRIVER_SUPPLIED_KEYS,
    FEDAVG_WORKER_PARAM_KEYS,
    MAX_LEASE_SECONDS,
    ExpansionError,
    expand_tasks,
)


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


# -- I8: the expansion's key list and the worker's reads must not drift -----


def _worker_param_reads() -> set[str]:
    """Every constant-string key `fedavg_worker` reads out of its params.

    Parsed from the worker's own SOURCE rather than asserted from memory:
    the whole point of this test is that no human has to remember to update
    two lists. The params dict is bound to `p` in `run_worker` and passed on
    as `params` to `_make_shard`, so both names count.
    """
    import flashml_workloads.fedavg_worker as worker

    tree = ast.parse(inspect.getsource(worker))
    found = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Name)
                and node.value.id in {"p", "params"}
                and isinstance(node.slice, ast.Constant)
                and isinstance(node.slice.value, str)):
            found.add(node.slice.value)
    return found


def test_worker_param_reads_are_bound_to_the_expansions_key_list():
    """`_expand_fedavg`'s key tuple and `fedavg_worker`'s `p[...]` reads are
    two places that must agree, with nothing in the code binding them. This
    is the same defect shape as the coordinator/agent task-module allowlist
    drift that already caused an outage here: each side is perfectly correct
    in isolation, and the failure only appears at run time on somebody
    else's machine. A test is the binding.
    """
    reads = _worker_param_reads()
    assert reads, "AST scan found no params reads — the scan itself is broken"

    submitter_supplied = reads - set(FEDAVG_DRIVER_SUPPLIED_KEYS)
    assert submitter_supplied == set(FEDAVG_WORKER_PARAM_KEYS), (
        "fedavg_worker's parameter reads and modea's FEDAVG_WORKER_PARAM_KEYS "
        f"have drifted.\n  worker reads, not validated at expansion: "
        f"{sorted(submitter_supplied - set(FEDAVG_WORKER_PARAM_KEYS))}\n"
        f"  validated at expansion, never read by the worker: "
        f"{sorted(set(FEDAVG_WORKER_PARAM_KEYS) - submitter_supplied)}"
    )


def test_every_forwarded_key_actually_reaches_the_task_payload():
    tasks = expand_tasks("job-1", _spec())
    for key in FEDAVG_WORKER_PARAM_KEYS:
        assert key in tasks[0].payload["params"], key
    for key in FEDAVG_DRIVER_SUPPLIED_KEYS:
        assert key in tasks[0].payload["params"], key


# -- lease_seconds is submitter-controlled and must be bounded --------------


def test_lease_seconds_is_clamped_to_a_sane_maximum():
    """A lease deadline is the ONLY thing that returns an abandoned task to
    the queue. `1e9` seconds pins a task to a machine that closed its laptop
    for ~31 years — every other volunteer sees the shard as permanently
    taken."""
    tasks = expand_tasks("job-1", _spec(lease_seconds=1e9))
    assert tasks[0].lease_seconds == MAX_LEASE_SECONDS


def test_lease_seconds_rejects_non_finite():
    """`float("inf")` is worse than merely long: `timedelta(seconds=inf)`
    raises OverflowError inside the coordinator's claim path, so the damage
    lands on a node trying to pick up work, not on the submitter."""
    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ExpansionError, match="finite"):
            expand_tasks("job-1", _spec(lease_seconds=bad))


def test_lease_seconds_rejects_non_positive_and_non_numeric():
    for bad in (0, -1):
        with pytest.raises(ExpansionError, match="> 0"):
            expand_tasks("job-1", _spec(lease_seconds=bad))
    with pytest.raises(ExpansionError, match="must be a number"):
        expand_tasks("job-1", _spec(lease_seconds="soon"))


def test_a_reasonable_lease_seconds_passes_through_untouched():
    tasks = expand_tasks("job-1", _spec(lease_seconds=90.0))
    assert tasks[0].lease_seconds == 90.0


def test_the_clamp_covers_the_other_lease_expansions_too():
    """Same submitter-controlled field, same defect, three expansions — the
    'two places, each fine alone' shape again. Fixing only the one the
    review named would leave the identical hole in hyperparameter_search."""
    spec = JobSpec.model_validate({
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": "hps"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": "local/tier1", "tag": "dev"},
            "workload": {"type": "hyperparameter_search",
                         "parameters": {"trials": [{"a": 1}],
                                        "lease_seconds": 1e9}},
        },
    })
    assert expand_tasks("job-2", spec)[0].lease_seconds == MAX_LEASE_SECONDS
