import json
import time
import urllib.error
import urllib.request

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

    `extra_keys` ({round: [key, ...]}) injects artifact keys that no task in
    the round legitimately owns — the recursive upload of a nested output
    tree, or a key naming a shard index beyond `num_shards`.

    `completed` ({round: [shard, ...]}) decouples "uploaded an artifact"
    from "the coordinator ACCEPTED the commit". Uploads happen before the
    commit is offered, so an attempt the coordinator rejected still leaves
    its metrics.json behind; by default every committed shard is reported
    COMPLETED, which is the honest case.
    """

    def __init__(self, commits, param_name="w", reveal_at=None,
                 extra_keys=None, completed=None):
        self.commits = commits
        self.param_name = param_name
        self.reveal_at = reveal_at or {}
        self.extra_keys = extra_keys or {}
        self.completed = completed
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
        for key in self.extra_keys.get(r, []):
            out.append({"key": key})
        return out

    def tasks(self, job_id):
        r = int(job_id.split("r")[1])
        if self.completed is None:
            shards = [e[0] for e in self.commits.get(r, [])]
        else:
            shards = self.completed.get(r, [])
        return [{"task_id": f"shard-{s:03d}", "state": "COMPLETED"} for s in shards]

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


from flashml_workloads.fedavg_driver import HttpCoordinator, resume_state


def test_http_coordinator_sends_auth_headers(monkeypatch):
    captured = {}

    def fake_request(method, url, data=None, headers=None, timeout=None):
        captured.update({"method": method, "url": url, "headers": headers or {}})
        return {"job_id": "job-r0"}

    coord = HttpCoordinator("http://c:8100", headers={"Authorization": "Bearer t"})
    monkeypatch.setattr(coord, "_request", fake_request)
    coord.submit({"spec": {}})
    assert captured["headers"]["Authorization"] == "Bearer t"
    assert captured["url"] == "http://c:8100/v1alpha1/jobs"


def test_http_coordinator_sends_auth_headers_on_every_method(monkeypatch):
    """submit is not the only call the cloud API's credential travels
    through — job_state/artifacts/get_artifact/put_artifact all pass
    `headers=self.headers` too. A regression dropping the header from any
    one of them would ship silently unauthenticated in production, so pin
    all five rather than only the one the earlier test happened to drive.
    """
    calls = []

    def fake_request(method, url, data=None, headers=None, timeout=None):
        calls.append({"method": method, "url": url, "headers": headers or {}})
        if method == "GET" and url.endswith("/v1alpha1/jobs/job-r0"):
            return {"state": "RUNNING"}
        if method == "GET" and url.endswith("/artifacts"):
            return []
        if method == "GET" and "/v1alpha1/artifacts/" in url:
            return {"w": {"shape": [1], "data": [1.0]}}
        return {"job_id": "job-r0"}

    coord = HttpCoordinator("http://c:8100", headers={"Authorization": "Bearer t"})
    monkeypatch.setattr(coord, "_request", fake_request)

    coord.submit({"spec": {}})
    coord.job_state("job-r0")
    coord.artifacts("job-r0")
    coord.get_artifact("jobs/job-r0/round-000/weights.json")
    coord.put_artifact("jobs/job-r0/round-000/weights.json", {"w": {}})

    assert len(calls) == 5
    for call in calls:
        assert call["headers"]["Authorization"] == "Bearer t", call


def test_resume_state_finds_last_completed_round():
    # 42.0 is deliberately unlike any delta value in `commits` — if the fake
    # ever re-derives a delta for this key instead of returning what was PUT,
    # this assertion must fail rather than coincide.
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {
        "w": {"shape": [1], "data": [42.0]}
    }
    next_round, weights, uri = resume_state(fake, [(0, "job-r0")])
    assert next_round == 1
    assert weights["w"]["data"] == [42.0]
    assert uri == "artifact://jobs/job-r0/round-000/weights.json"


def test_resume_state_picks_the_newest_round_not_the_first():
    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {"w": {"shape": [1], "data": [1.0]}}
    fake.uploaded["jobs/job-r1/round-001/weights.json"] = {"w": {"shape": [1], "data": [2.0]}}
    next_round, weights, _ = resume_state(fake, [(0, "job-r0"), (1, "job-r1")])
    assert next_round == 2
    assert weights["w"]["data"] == [2.0]


def test_resume_state_skips_a_round_that_never_aggregated():
    # Round 1 was submitted but crashed before writing weights: resume at 1.
    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {"w": {"shape": [1], "data": [7.0]}}
    next_round, weights, _ = resume_state(fake, [(0, "job-r0"), (1, "job-r1")])
    assert next_round == 1
    assert weights["w"]["data"] == [7.0]


def test_resume_state_propagates_transport_errors():
    """An unreachable coordinator must NOT look like 'no rounds completed' —
    that would silently restart a finished run from scratch."""
    class Unreachable(FakeCoordinator):
        def get_artifact(self, key):
            raise ConnectionError("coordinator unreachable")

    with pytest.raises(ConnectionError):
        resume_state(Unreachable({}), [(0, "job-r0")])


def test_resume_state_on_empty_history_starts_at_zero():
    fake = FakeCoordinator({})
    assert resume_state(fake, []) == (0, {}, None)


def test_resume_state_raises_on_present_but_empty_weights_artifact():
    """A key that exists (no ArtifactNotFound) but resolves to `{}` — or
    `None`, matching `HttpCoordinator._request`'s empty-body-200 case — is
    not "this round never completed"; it is a corrupt or truncated commit.
    Silently treating it as absent and walking further back would either
    redo completed work or resume from stale weights from an earlier
    round. It must be surfaced, not swallowed.
    """
    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {}
    with pytest.raises(RuntimeError):
        resume_state(fake, [(0, "job-r0")])

    fake2 = FakeCoordinator({})
    fake2.uploaded["jobs/job-r0/round-000/weights.json"] = None
    with pytest.raises(RuntimeError):
        resume_state(fake2, [(0, "job-r0")])


def test_http_coordinator_get_artifact_maps_404_and_propagates_503(monkeypatch):
    """Verified only by reading the source until now — drive the real
    `_request` -> `urllib.request.urlopen` path (stubbing at the urlopen
    level, not `_request` itself) so the 404 -> ArtifactNotFound mapping
    and 5xx passthrough are exercised, not just eyeballed.
    """
    coord = HttpCoordinator("http://c:8100")

    def raise_404(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, "Not Found", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_404)
    with pytest.raises(ArtifactNotFound):
        coord.get_artifact("jobs/job-r0/round-000/weights.json")

    def raise_503(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 503, "Service Unavailable", None, None)

    monkeypatch.setattr(urllib.request, "urlopen", raise_503)
    with pytest.raises(urllib.error.HTTPError):
        coord.get_artifact("jobs/job-r0/round-000/weights.json")


# -- C1: quorum counts uploaded artifacts, not accepted commits -------------


def test_quorum_ignores_metrics_files_outside_the_expected_task_set():
    """A single lease must never mint more than one participant.

    flashnode uploads a task's whole output tree RECURSIVELY (loop.py's
    `rglob`), so a worker that writes `out/a/metrics.json` and
    `out/b/metrics.json` gets both keys into the job's artifact listing. A
    suffix test (`key.endswith("metrics.json")`) counts them as two more
    participants bought with one lease — which both dilutes every honest
    shard's weight and multiplies the attacker's. A key naming a shard index
    beyond `num_shards` is the same trick with no lease at all.

    Only shard 0 legitimately committed, so quorum of 2 must NOT be met.
    """
    fake = FakeCoordinator(
        {0: [(0, 1.0, 10)]},
        extra_keys={0: ["jobs/job-r0/shard-000/a/metrics.json",
                        "jobs/job-r0/shard-000/b/metrics.json",
                        "jobs/job-r0/shard-009/metrics.json",
                        "jobs/job-r0/metrics.json"]},
    )
    with pytest.raises(QuorumNotMet, match="1 of 2"):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                   worker_params=_params(), round_timeout_s=0.2, poll_seconds=0.01,
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_quorum_counts_only_tasks_the_coordinator_accepted():
    """Artifacts are PUT before the commit is offered, so an attempt the
    coordinator REJECTED (lost lease, sha256 mismatch, attempts exhausted)
    still leaves its metrics.json in the bucket. Both shards uploaded here;
    only shard 0's commit was accepted, so there is one participant, not two.
    """
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 99.0, 10)]},
                           completed={0: [0]})
    with pytest.raises(QuorumNotMet, match="1 of 2"):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                   worker_params=_params(), round_timeout_s=0.2, poll_seconds=0.01,
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_quorum_still_works_against_a_coordinator_without_a_tasks_view():
    """The task cross-check is defence in depth, not a hard dependency: a
    Coordinator implementation that predates `tasks()` must still run, with
    the expected-key filter alone bounding the count at num_shards."""
    class LegacyCoordinator:
        """Exactly the five original Coordinator methods — no `tasks`."""

        def __init__(self, inner):
            self._inner = inner

        def submit(self, body):
            return self._inner.submit(body)

        def job_state(self, job_id):
            return self._inner.job_state(job_id)

        def artifacts(self, job_id):
            return self._inner.artifacts(job_id)

        def get_artifact(self, key):
            return self._inner.get_artifact(key)

        def put_artifact(self, key, body):
            return self._inner.put_artifact(key, body)

    fake = LegacyCoordinator(FakeCoordinator({0: [(0, 1.0, 10), (1, 3.0, 10)]}))
    assert not hasattr(fake, "tasks")
    result = run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                        worker_params=_params(),
                        initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["history"][0]["participants"] == 2


# -- C2/C3: poisoned contributions must not reach the model -----------------


def test_a_negative_sample_count_cannot_amplify_one_shards_delta():
    """(delta=-999, n=-999) with (delta=1.0, n=1000) totals a healthy 1
    sample but produces a weight of 999001.0 where the honest step is 1.0.
    The round must fail, not aggregate."""
    fake = FakeCoordinator({0: [(0, -999.0, -999), (1, 1.0, 1000)]})
    with pytest.raises(ValueError, match="non-positive sample count"):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                   worker_params=_params(),
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_a_nan_delta_raises_instead_of_poisoning_every_later_round():
    """Without the guard: every weight becomes NaN, rounds 1..N train from
    NaN weights, and run_fedavg still returns success."""
    from flashml_workloads.fedavg_weights import NonFiniteWeights

    fake = FakeCoordinator({r: [(0, float("nan"), 10), (1, 1.0, 10)]
                            for r in range(3)})
    with pytest.raises(NonFiniteWeights):
        run_fedavg(fake, rounds=3, num_shards=2, min_participants=2,
                   worker_params=_params(),
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


def test_an_inf_delta_raises_too():
    from flashml_workloads.fedavg_weights import NonFiniteWeights

    fake = FakeCoordinator({0: [(0, float("inf"), 10), (1, 1.0, 10)]})
    with pytest.raises(NonFiniteWeights):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                   worker_params=_params(),
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


# -- I7: the round's image and isolation tier are caller-settable -----------


def test_round_body_defaults_to_the_dev_image_and_standard_tier():
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})
    run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
               worker_params=_params(),
               initial_weights={"w": {"shape": [1], "data": [0.0]}})
    spec = fake.submitted[0][1]["spec"]
    assert spec["image"] == {"repository": "local/tier1", "tag": "dev"}
    assert spec["isolation"] == {"tier": "standard", "allowFallback": False}


def test_image_and_isolation_tier_are_settable_by_the_caller():
    """`local/tier1:dev` exists only in this repo's e2e fixtures. It is inert
    today solely because SubprocessRunner ignores `image` — a docker-tier
    volunteer would fail to pull it. The caller must be able to say what to
    run, so the value is not pinned in two places that only agree by luck."""
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})
    run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
               worker_params=_params(), image="ghcr.io/zolli/fedavg:1.2.3",
               isolation_tier="sandboxed", allow_fallback=False,
               initial_weights={"w": {"shape": [1], "data": [0.0]}})
    spec = fake.submitted[0][1]["spec"]
    assert spec["image"] == {"repository": "ghcr.io/zolli/fedavg", "tag": "1.2.3"}
    assert spec["isolation"]["tier"] == "sandboxed"


def test_an_image_without_a_tag_is_rejected():
    fake = FakeCoordinator({0: [(0, 1.0, 10)]})
    with pytest.raises(ValueError, match="repository:tag"):
        run_fedavg(fake, rounds=1, num_shards=1, min_participants=1,
                   worker_params=_params(), image="ghcr.io/zolli/fedavg",
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


# -- I9: a transient coordinator error must not end a multi-round run -------


def test_poll_retries_a_transient_coordinator_error():
    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})
    real_artifacts, real_state = fake.artifacts, fake.job_state
    calls = {"artifacts": 0, "state": 0}

    def flaky_artifacts(job_id):
        calls["artifacts"] += 1
        if calls["artifacts"] <= 2:
            raise ConnectionError("connection reset by peer")
        return real_artifacts(job_id)

    def flaky_state(job_id):
        calls["state"] += 1
        raise urllib.error.HTTPError("u", 502, "Bad Gateway", None, None)

    fake.artifacts = flaky_artifacts
    fake.job_state = flaky_state

    result = run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                        worker_params=_params(), poll_backoff_s=0.001,
                        initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert result["history"][0]["participants"] == 2
    assert calls["artifacts"] == 3  # two failures, then the real listing


def test_poll_retries_are_bounded_and_fail_with_context():
    """Retry, but never swallow indefinitely: a coordinator that is simply
    gone must end the run with an explanation, not hang forever."""
    from flashml_workloads.fedavg_driver import CoordinatorUnavailable

    fake = FakeCoordinator({0: [(0, 1.0, 10), (1, 1.0, 10)]})

    def dead(job_id):
        raise ConnectionError("coordinator unreachable")

    fake.artifacts = dead
    with pytest.raises(CoordinatorUnavailable,
                       match="round 0: listing committed shards"):
        run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                   worker_params=_params(), poll_backoff_s=0.001,
                   round_timeout_s=5.0,
                   initial_weights={"w": {"shape": [1], "data": [0.0]}})


# -- I6: the round <-> list-index identity breaks on the SECOND resume ------


def test_resume_state_rejects_a_bare_job_id_list():
    fake = FakeCoordinator({})
    with pytest.raises(TypeError, match="round, job_id"):
        resume_state(fake, ["job-r0"])


def test_resume_state_uses_the_carried_round_not_the_list_position():
    """The second-resume bug, minimally: a run resumed at round 5 has round 5
    at index 0. Position-as-round probes `round-000` under the round-5 job,
    gets ArtifactNotFound, and reports "restart from scratch"."""
    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r5/round-005/weights.json"] = {
        "w": {"shape": [1], "data": [5.0]}
    }
    next_round, weights, uri = resume_state(fake, [(5, "job-r5")])
    assert next_round == 6
    assert weights["w"]["data"] == [5.0]
    assert uri == "artifact://jobs/job-r5/round-005/weights.json"


def test_a_second_resume_does_not_silently_restart_from_scratch():
    """Full two-crash sequence against the driver, not just resume_state.

    Crash 1 after round 0 -> resume at round 1. Crash 2 after round 1 ->
    the carried history is [(1, 'job-r1')], whose only entry is at index 0.
    Position-as-round probes `jobs/job-r1/round-000/weights.json`, misses,
    and returns (0, {}, None): training silently starts over, discarding
    two completed rounds.
    """
    fake = FakeCoordinator({r: [(0, 1.0, 10), (1, 1.0, 10)] for r in range(4)})

    first = run_fedavg(fake, rounds=1, num_shards=2, min_participants=2,
                       worker_params=_params(),
                       initial_weights={"w": {"shape": [1], "data": [0.0]}})
    assert first["job_ids"] == [(0, "job-r0")]

    start, weights, uri = resume_state(fake, first["job_ids"])
    assert start == 1

    second = run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
                        worker_params=_params(), start_round=start,
                        weights_uri=uri, initial_weights=weights)
    assert second["job_ids"] == [(1, "job-r1")]

    start2, weights2, uri2 = resume_state(fake, second["job_ids"])
    assert start2 == 2, "a second resume must not restart from round 0"
    assert weights2["w"]["data"] == [pytest.approx(2.0)]
    assert uri2 == "artifact://jobs/job-r1/round-001/weights.json"


def test_prior_job_ids_are_carried_into_the_returned_history():
    fake = FakeCoordinator({1: [(0, 1.0, 10), (1, 1.0, 10)]})
    result = run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
                        worker_params=_params(), start_round=1,
                        prior_job_ids=[(0, "job-r0")],
                        weights_uri="artifact://jobs/job-r0/round-000/weights.json",
                        initial_weights={"w": {"shape": [1], "data": [5.0]}})
    assert result["job_ids"] == [(0, "job-r0"), (1, "job-r1")]


def test_resume_state_refuses_to_guess_when_the_history_is_incomplete():
    """A resumed run whose rounds all failed carries only rounds >= 1. There
    is no evidence about rounds 0..N-1 in that list, so "start from scratch"
    is a guess that throws away completed work. Say so instead."""
    fake = FakeCoordinator({})
    with pytest.raises(RuntimeError, match="carried history starts at round 5"):
        resume_state(fake, [(5, "job-r5"), (6, "job-r6")])


# -- C4 (read path): a resumed weights artifact is not gated on finiteness --


def test_resume_state_rejects_a_non_finite_weights_artifact():
    """resume_state is the READ path: a weights.json written by an
    unauthenticated artifact PUT (or corrupted some other way) must not be
    handed back to the caller with a NaN silently inside it. Without a gate
    here, `run_fedavg(..., weights_uri=uri, initial_weights=weights)` would
    resume training from NaN and still report success."""
    from flashml_workloads.fedavg_weights import NonFiniteWeights

    fake = FakeCoordinator({})
    fake.uploaded["jobs/job-r0/round-000/weights.json"] = {
        "w": {"shape": [1], "data": [float("nan")]}
    }
    with pytest.raises(NonFiniteWeights):
        resume_state(fake, [(0, "job-r0")])


def test_run_fedavg_resumes_from_start_round():
    fake = FakeCoordinator({1: [(0, 1.0, 10), (1, 1.0, 10)]})
    result = run_fedavg(fake, rounds=2, num_shards=2, min_participants=2,
                        worker_params=_params(), start_round=1,
                        initial_weights={"w": {"shape": [1], "data": [5.0]}},
                        weights_uri="artifact://jobs/job-r0/round-000/weights.json")
    # Only round 1 runs; weights walk 5.0 -> 6.0
    assert [h["round"] for h in result["history"]] == [1]
    assert result["weights"]["w"]["data"] == [pytest.approx(6.0)]
    # The resumed round's job must declare the passed-in weights_uri as its
    # input. Asserting only on history/weights (as above) can't catch a
    # regression that reintroduces a local `weights_uri = None` alongside
    # the parameter — every resumed run would silently restart from the
    # initial model while these two assertions kept passing.
    params = fake.submitted[0][1]["spec"]["workload"]["parameters"]
    assert params["weights"] == "artifact://jobs/job-r0/round-000/weights.json"
