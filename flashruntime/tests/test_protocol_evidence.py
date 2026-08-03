"""`ExecutionEvidence` — what the agent says about the run it just finished.

Slice 2 of the verification design
(`2026-08-03-result-verification-design.md` §3). These are protocol tests,
not verifier tests: they pin the *wire* behaviour the cloud's evidence slice
is later allowed to trust, exactly as `test_protocol_gpu.py` does for
`GpuInfo`.

Two properties carry the whole slice, and both are tested here rather than
assumed:

1. **`evidence` is OPTIONAL and must stay optional.** Every agent deployed
   today predates the field. A required field would 422 every completion in
   the fleet the moment this coordinator shipped — the same fail-safe
   reasoning as `module_capable`'s fail-open polarity, and for the same
   reason: availability, not security.
2. **Unknown is not zero.** `None` means "not measured"; `0.0` means
   "measured, and it was zero". A `0.0` standing in for absence is exactly
   the value that later reads as "this node did nothing" and gets an honest
   volunteer flagged (§5: absence of evidence is not evidence of anything).
"""

from __future__ import annotations

import pathlib

import pytest

from flashruntime.protocol.v1alpha1 import ExecutionEvidence


# ---------------------------------------------------------------------------
# ExecutionEvidence: the unknown sentinels
# ---------------------------------------------------------------------------


def test_empty_evidence_is_all_unknown_and_survives_json():
    """An agent that could measure nothing still sends a well-formed block.
    Every numeric field is None — never 0.0 — and the omission survives the
    round trip rather than being defaulted into a number on the way back."""
    restored = ExecutionEvidence.model_validate_json(ExecutionEvidence().model_dump_json())

    assert restored.wall_seconds is None
    assert restored.cpu_percent_mean is None
    assert restored.gpu_util_percent_mean is None
    assert restored.image_digest == ""
    assert restored.exit_code is None


def test_a_fully_populated_reading_round_trips():
    evidence = ExecutionEvidence(
        wall_seconds=9.13,
        cpu_percent_mean=87.5,
        gpu_util_percent_mean=94.0,
        image_digest="sha256:" + "ab" * 32,
        exit_code=0,
    )
    restored = ExecutionEvidence.model_validate_json(evidence.model_dump_json())

    assert restored.wall_seconds == 9.13
    assert restored.cpu_percent_mean == 87.5
    assert restored.gpu_util_percent_mean == 94.0
    assert restored.image_digest == "sha256:" + "ab" * 32
    assert restored.exit_code == 0


def test_measured_zero_is_not_the_same_value_as_not_measured():
    """THE field-level invariant of this slice. An idle GPU on a task that
    asked for one is the single strongest signal slice 2 carries (§3.2); a
    node with no sampler at all is no signal whatsoever. Collapsing them onto
    one value would turn "we could not look" into "it did nothing"."""
    measured = ExecutionEvidence(cpu_percent_mean=0.0, gpu_util_percent_mean=0.0)
    unmeasured = ExecutionEvidence()

    assert measured.cpu_percent_mean == 0.0
    assert measured.gpu_util_percent_mean == 0.0
    assert unmeasured.cpu_percent_mean is None
    assert unmeasured.gpu_util_percent_mean is None
    assert measured != unmeasured

    # And the distinction has to survive the wire, not just the constructor.
    assert ExecutionEvidence.model_validate_json(measured.model_dump_json()).cpu_percent_mean == 0.0
    assert (
        ExecutionEvidence.model_validate_json(unmeasured.model_dump_json()).cpu_percent_mean is None
    )


def test_a_partial_reading_is_valid():
    """The normal case on a CPU-only volunteer: wall clock and exit code are
    always knowable, GPU utilisation never is. A model that demanded all or
    nothing would push agents towards sending a fabricated 0.0."""
    evidence = ExecutionEvidence.model_validate(
        {"wall_seconds": 4.0, "exit_code": 0, "gpu_util_percent_mean": None}
    )

    assert evidence.wall_seconds == 4.0
    assert evidence.gpu_util_percent_mean is None


def test_fields_a_newer_agent_invents_are_ignored_not_rejected():
    """Forward compatibility in the other direction: an upgraded agent
    reporting a field this coordinator has never heard of must still be able
    to commit its work."""
    evidence = ExecutionEvidence.model_validate(
        {"wall_seconds": 4.0, "rss_bytes_peak": 123, "tpm_quote": "..."}
    )

    assert evidence.wall_seconds == 4.0
    assert not hasattr(evidence, "rss_bytes_peak")


@pytest.mark.parametrize(
    "field, value",
    [
        pytest.param("wall_seconds", -1.0, id="negative-wall-clock"),
        pytest.param("cpu_percent_mean", 3200.0, id="cpu-above-100"),
        pytest.param("gpu_util_percent_mean", -5.0, id="negative-utilisation"),
        pytest.param("exit_code", -9, id="killed-by-signal"),
    ],
)
def test_implausible_readings_are_accepted_by_the_wire_and_left_to_the_verifier(field, value):
    """Deliberately UNCONSTRAINED, and this is the test that says so.

    A `ge=0` on these fields would make an implausible reading a 422 — i.e.
    the agent's self-report could refuse the agent's own work. That inverts
    the design: evidence is advisory, nothing is ever enforced (§5), and a
    verdict of `flag` on a nonsense value is worth strictly more than a
    rejected commit. It would also teach a liar exactly which values pass,
    which is the one thing validation here would reliably achieve.

    `exit_code=-9` is not even implausible: it is what a SIGKILL looks like.
    """
    assert getattr(ExecutionEvidence.model_validate({field: value}), field) == value


# ---------------------------------------------------------------------------
# CompleteRequest: optional, and it must stay optional
# ---------------------------------------------------------------------------


def test_complete_request_without_evidence_is_valid():
    """The whole deployed fleet. Not one agent alive today sends this block,
    and every one of them must keep committing work."""
    from flashruntime.service.modea import CompleteRequest

    req = CompleteRequest.model_validate({"output_sha256": "a" * 64})

    assert req.output_sha256 == "a" * 64
    assert req.evidence is None


def test_complete_request_with_explicit_null_evidence_is_valid():
    """An agent that measured nothing may say so explicitly. `null` and
    absent mean the same thing and neither is an error."""
    from flashruntime.service.modea import CompleteRequest

    assert CompleteRequest.model_validate(
        {"output_sha256": "a" * 64, "evidence": None}
    ).evidence is None


def test_complete_request_parses_evidence_into_the_typed_model():
    from flashruntime.service.modea import CompleteRequest

    req = CompleteRequest.model_validate(
        {
            "output_sha256": "a" * 64,
            "evidence": {"wall_seconds": 9.1, "cpu_percent_mean": 0.0, "exit_code": 0},
        }
    )

    assert isinstance(req.evidence, ExecutionEvidence)
    assert req.evidence.wall_seconds == 9.1
    assert req.evidence.cpu_percent_mean == 0.0  # measured zero, not absence
    assert req.evidence.gpu_util_percent_mean is None  # absence, not zero


# ---------------------------------------------------------------------------
# The endpoint: accept it, and store nothing
# ---------------------------------------------------------------------------


DIGEST = "sha256:" + "cafe" * 16


@pytest.fixture()
def coordinator(tmp_path):
    """A bare Mode A coordinator with one claimable task."""
    import fastapi
    from fastapi.testclient import TestClient

    from flashruntime.leases import LeaseManager
    from flashruntime.protocol.v1alpha1 import TaskSpec
    from flashruntime.service.modea import ModeAState, build_router

    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    state = ModeAState(LeaseManager(), artifacts_dir=pathlib.Path(artifacts))
    app = fastapi.FastAPI()
    app.include_router(build_router(state))
    state.manager.add_task(
        TaskSpec(task_id="task-000", job_id="job-a", commit_key="jobs/job-a/task-000/metrics.json")
    )
    client = TestClient(app)
    r = client.post(
        "/v1alpha1/nodes/register",
        json={
            "node_id": "node-a", "kubernetes_node": "", "hostname": "node-a",
            "capabilities": {"cpu_cores": 8},
        },
    )
    assert r.status_code == 200
    return client


def _claim_and_upload(client):
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    record = client.put(
        "/v1alpha1/artifacts/jobs/job-a/task-000/metrics.json", content=b'{"loss": 0.1}'
    ).json()
    return lease["lease_id"], record["sha256"]


def test_a_completion_carrying_no_evidence_still_succeeds(coordinator):
    """The upgrade test. Byte for byte the body every deployed agent sends."""
    lease_id, sha = _claim_and_upload(coordinator)

    r = coordinator.post(f"/v1alpha1/attempts/{lease_id}/complete", json={"output_sha256": sha})

    assert r.status_code == 200
    assert r.json()["accepted"] is True


def test_a_completion_carrying_evidence_succeeds_and_the_coordinator_stores_nothing(coordinator):
    """It accepts and ignores: the coordinator has no verifications ledger,
    and inventing a place to put this here would put the runtime in the
    business of judging its own volunteers. The cloud API reads it."""
    lease_id, sha = _claim_and_upload(coordinator)

    r = coordinator.post(
        f"/v1alpha1/attempts/{lease_id}/complete",
        json={
            "output_sha256": sha,
            "evidence": {
                "wall_seconds": 9.13, "cpu_percent_mean": 87.5,
                "gpu_util_percent_mean": None, "image_digest": DIGEST, "exit_code": 0,
            },
        },
    )

    assert r.status_code == 200
    assert r.json() == {"accepted": True}
    # Nothing the coordinator serves has grown an evidence field.
    for path in ("/v1alpha1/jobs/job-a/tasks", "/v1alpha1/nodes"):
        body = coordinator.get(path).text
        assert DIGEST not in body
        assert "evidence" not in body


def test_hostile_evidence_never_costs_the_agent_its_commit(coordinator):
    """A node that reports nonsense is a node to FLAG, not one to refuse.
    Rejecting the commit would requeue honest work over a cosmetic field —
    and hand any agent a way to fail its own attempts on demand."""
    lease_id, sha = _claim_and_upload(coordinator)

    r = coordinator.post(
        f"/v1alpha1/attempts/{lease_id}/complete",
        json={
            "output_sha256": sha,
            "evidence": {
                "wall_seconds": -1.0, "cpu_percent_mean": 999999.0,
                "gpu_util_percent_mean": -5.0, "image_digest": "not-a-digest",
                "exit_code": -9,
            },
        },
    )

    assert r.status_code == 200
    assert r.json()["accepted"] is True
