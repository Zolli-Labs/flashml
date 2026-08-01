import json

import pytest

from flashml_workloads.fedavg_driver import ArtifactNotFound, QuorumNotMet, run_fedavg


class FakeCoordinator:
    """Stands in for the HTTP coordinator.

    `commits[r]` lists (shard, delta_value, samples) that appear for round r.
    A shard absent from a round never commits — that is how a straggler or a
    closed laptop is simulated.
    """

    def __init__(self, commits, param_name="w"):
        self.commits = commits
        self.param_name = param_name
        self.submitted = []
        self.uploaded = {}
        self._round = -1

    def submit(self, body):
        self._round = body["spec"]["workload"]["parameters"]["round"]
        job_id = f"job-r{self._round}"
        self.submitted.append((job_id, body))
        return {"job_id": job_id}

    def job_state(self, job_id):
        return "RUNNING"

    def artifacts(self, job_id):
        r = int(job_id.split("r")[1])
        out = []
        for shard, _, _ in self.commits.get(r, []):
            out.append({"key": f"jobs/{job_id}/shard-{shard:03d}/metrics.json"})
            out.append({"key": f"jobs/{job_id}/shard-{shard:03d}/delta.json"})
        return out

    def get_artifact(self, key):
        # Anything previously PUT wins. Without this the fake would
        # re-derive a *delta* for a weights key and a resume test could pass
        # against entirely the wrong data path.
        if key in self.uploaded:
            return self.uploaded[key]
        parts = key.split("/")
        if len(parts) < 3 or not parts[2].startswith("shard-"):
            raise ArtifactNotFound(key)
        r = int(parts[1].split("r")[1])
        shard = int(parts[2].split("-")[1])
        match = [(v, n) for s, v, n in self.commits.get(r, []) if s == shard]
        if not match:
            raise ArtifactNotFound(key)
        value, samples = match[0]
        if key.endswith("metrics.json"):
            return {"round": r, "shard": shard, "samples": samples,
                    "loss": 1.0 / (r + 1), "local_steps": 5,
                    "delta_file": "delta.json"}
        return {self.param_name: {"shape": [1], "data": [value]}}

    def put_artifact(self, key, body):
        self.uploaded[key] = body


def _params():
    return {"local_steps": 5, "lr": 0.05, "batch_size": 8, "seed": 0,
            "in_dim": 4, "hidden": 8, "out_dim": 2, "dataset_size": 64}


def test_runs_requested_number_of_rounds():
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(3)})
    result = run_fedavg(fake, rounds=3, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert len(result["history"]) == 3
    assert [h["round"] for h in result["history"]] == [0, 1, 2]


def test_weights_accumulate_reduced_deltas():
    # Every round both shards report +1.0 -> weights walk 0 -> 1 -> 2 -> 3.
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(3)})
    result = run_fedavg(fake, rounds=3, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["weights"]["w"]["data"] == [pytest.approx(3.0)]


def test_aggregates_on_quorum_without_waiting_for_stragglers():
    # 3 shards, only 2 ever commit; quorum of 2 must still complete the round.
    fake = FakeCoordinator({0: [(0, 2.0, 10), (2, 4.0, 10)]})
    result = run_fedavg(fake, rounds=1, num_shards=3, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["history"][0]["participants"] == 2
    assert result["weights"]["w"]["data"] == [pytest.approx(3.0)]


def test_quorum_uses_sample_weighting():
    fake = FakeCoordinator({0: [(0, 1.0, 100), (1, 5.0, 300)]})
    result = run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["weights"]["w"]["data"] == [pytest.approx(4.0)]


def test_raises_when_quorum_never_met():
    fake = FakeCoordinator({0: [(0, 1.0, 10)]})
    with pytest.raises(QuorumNotMet, match="1 of 2"):
        run_fedavg(fake, rounds=1, num_shards=3, min_participants=2,
                   worker_params=_params(), round_timeout_s=0.2, poll_seconds=0.01,
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_each_round_declares_previous_round_weights_as_input():
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(2)})
    run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
               worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    first = fake.submitted[0][1]["spec"]["workload"]["parameters"]
    second = fake.submitted[1][1]["spec"]["workload"]["parameters"]
    assert "weights" not in first
    assert second["weights"].startswith("artifact://")


def test_rejects_a_task_declaring_a_traversing_delta_file():
    """metrics.json is written by an untrusted volunteer node. A delta_file
    naming another job's artifact must be refused, not averaged in."""
    from flashml_workloads.fedavg_driver import _safe_delta_key

    for evil in ("../../other-job/weights.json", "a/b.json", "..", ""):
        with pytest.raises(ValueError, match="unsafe delta_file"):
            _safe_delta_key("jobs/job-r0/shard-000/metrics.json", evil)

    assert _safe_delta_key("jobs/job-r0/shard-000/metrics.json", "delta.json") == \
        "jobs/job-r0/shard-000/delta.json"


def test_deltas_are_not_downloaded_until_quorum_is_reached():
    """Deltas are megabytes; re-fetching them on every poll tick would
    dominate the round's transfer cost."""
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})
    fetched = []
    real_get = fake.get_artifact
    fake.get_artifact = lambda k: (fetched.append(k), real_get(k))[1]

    run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
               worker_params=_params(),
               initial_weights={"w": {"shape": [1], "data": [0.0]}})

    # Exactly one metrics + one delta read per participant, no repeats.
    assert sorted(fetched) == sorted([
        "jobs/job-r0/shard-000/metrics.json", "jobs/job-r0/shard-000/delta.json",
        "jobs/job-r0/shard-001/metrics.json", "jobs/job-r0/shard-001/delta.json",
    ])


def test_on_round_callback_receives_progress():
    seen = []
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(2)})
    run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
               worker_params=_params(), on_round=seen.append,
               initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert [s["round"] for s in seen] == [0, 1]
    assert all("mean_loss" in s for s in seen)
