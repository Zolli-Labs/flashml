import json
import time

import pytest

from flashml_workloads.fedavg_driver import ArtifactNotFound, QuorumNotMet, run_fedavg


class FakeCoordinator:
    """Stands in for the HTTP coordinator.

    `commits[r]` lists entries for round r: either `(shard, delta_value,
    samples)`, or `(shard, delta_value, samples, loss)` when a test needs to
    pin a specific per-shard loss. A shard absent from a round never
    commits — that is how a straggler or a closed laptop is simulated.

    `reveal_at` optionally staggers when a shard's commit becomes visible:
    `{shard: n}` means that shard's artifacts are absent from `artifacts()`
    until strictly more than `n` calls have been made, simulating a commit
    that lands mid-poll rather than being present from the first tick. A
    shard with no entry (the default) is visible immediately, matching the
    original behavior. This lets a test construct a genuine pre-quorum tick
    with a partial, sub-quorum commit set — something an all-or-nothing
    fake can never produce, and therefore can never use to catch an
    implementation that fetches deltas from inside the poll loop instead of
    once after quorum.
    """

    def __init__(self, commits, param_name="w", reveal_at=None):
        self.commits = commits
        self.param_name = param_name
        self.reveal_at = reveal_at or {}
        self.submitted = []
        self.uploaded = {}
        self.artifacts_calls = 0
        self._round = -1

    def submit(self, body):
        self._round = body["spec"]["workload"]["parameters"]["round"]
        job_id = f"job-r{self._round}"
        self.submitted.append((job_id, body))
        return {"job_id": job_id}

    def job_state(self, job_id):
        return "RUNNING"

    def artifacts(self, job_id):
        self.artifacts_calls += 1
        r = int(job_id.split("r")[1])
        out = []
        for entry in self.commits.get(r, []):
            shard = entry[0]
            if self.artifacts_calls <= self.reveal_at.get(shard, 0):
                continue
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
        match = [e for e in self.commits.get(r, []) if e[0] == shard]
        if not match:
            raise ArtifactNotFound(key)
        entry = match[0]
        value, samples = entry[1], entry[2]
        loss = entry[3] if len(entry) > 3 else 1.0 / (r + 1)
        if key.endswith("metrics.json"):
            return {"round": r, "shard": shard, "samples": samples,
                    "loss": loss, "local_steps": 5,
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


def test_round_deadline_is_not_overrun_by_a_full_poll_interval():
    """round_timeout_s=0.05 with poll_seconds=5.0: an un-clamped sleep
    checks the deadline only once every 5 seconds, so a straggler round
    would overrun its budget by up to a full poll tick. The deadline must
    be hit promptly regardless of how coarse the poll interval is."""
    fake = FakeCoordinator({0: []})
    started = time.monotonic()
    with pytest.raises(QuorumNotMet):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                   worker_params=_params(), round_timeout_s=0.05, poll_seconds=5.0,
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})
    elapsed = time.monotonic() - started
    assert elapsed < 1.0


def test_mean_loss_is_sample_weighted():
    # (100 * 1.0 + 300 * 5.0) / 400 = 4.0, not the unweighted (1.0 + 5.0) / 2 = 3.0.
    fake = FakeCoordinator({0: [(0, 1.0, 100, 1.0), (1, 1.0, 300, 5.0)]})
    result = run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                        worker_params=_params(), initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["history"][0]["mean_loss"] == pytest.approx(4.0)


def test_job_failure_before_quorum_raises_quorum_not_met():
    fake = FakeCoordinator({0: [(0, 1.0, 10)]})
    fake.job_state = lambda job_id: "FAILED"
    with pytest.raises(QuorumNotMet, match="FAILED"):
        run_fedavg(fake, rounds=1, num_shards=3, min_participants=2,
                   worker_params=_params(),
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_rejects_min_participants_greater_than_num_shards():
    fake = FakeCoordinator({0: []})
    with pytest.raises(ValueError, match="exceeds num_shards"):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=3,
                   worker_params=_params(),
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
    naming another job's artifact must be refused, not averaged in. This is
    an allowlist, not a denylist of specific bad substrings: %2F, a leading
    ~, and an embedded NUL byte must all be refused too, not just literal
    slashes and dot-dot."""
    from flashml_workloads.fedavg_driver import _safe_delta_key

    for evil in ("../../other-job/weights.json", "a/b.json", "..", "",
                 "%2F..", "~/x.json", "a\x00b.json", ".hidden"):
        with pytest.raises(ValueError, match="unsafe delta_file"):
            _safe_delta_key("jobs/job-r0/shard-000/metrics.json", evil)

    for safe in ("delta.json", "delta-1.json", "delta_1.json"):
        assert _safe_delta_key("jobs/job-r0/shard-000/metrics.json", safe) == \
            f"jobs/job-r0/shard-000/{safe}"


def test_deltas_are_not_downloaded_until_quorum_is_reached():
    """Deltas are megabytes; fetching them on every poll tick would
    dominate the round's transfer cost.

    `reveal_at={0: 2, 1: 4}` staggers the two shards' commits: shard 0
    becomes visible on tick 3, shard 1 not until tick 5. That produces two
    genuine pre-quorum ticks (3 and 4) where the commit set is non-empty
    but still short of the `min_participants=2` quorum — a fake that
    reveals everything in one all-or-nothing jump can never produce that
    state, and without it an implementation that fetches deltas from
    inside the poll loop (instead of once, after quorum) would pass this
    test undetected: it would just re-fetch on an all-empty tick, which
    touches nothing.
    """
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]},
                           reveal_at={0: 2, 1: 4})
    fetched = []
    real_get = fake.get_artifact
    fake.get_artifact = lambda k: (fetched.append(k), real_get(k))[1]

    real_artifacts = fake.artifacts

    def spy_artifacts(job_id):
        listing = real_artifacts(job_id)
        metrics_seen = [e for e in listing if e["key"].endswith("metrics.json")]
        if len(metrics_seen) < 2:  # below min_participants: still pre-quorum
            assert fetched == [], (
                "an artifact was fetched while the poll was still "
                "pre-quorum (fewer than min_participants committed)"
            )
        return listing

    fake.artifacts = spy_artifacts

    run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
               worker_params=_params(), poll_seconds=0.01,
               initial_weights={"w": {"shape": [1], "data": [0.0]}})

    assert fake.artifacts_calls >= 5
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
