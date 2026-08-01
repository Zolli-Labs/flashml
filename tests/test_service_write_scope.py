"""A node may write only under the tasks it currently holds."""

import pytest
from fastapi.testclient import TestClient

from flashruntime.service.app import create_app, RuntimeSettings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHML_OPERATOR_TOKENS", "driver:op-tok")
    monkeypatch.setenv("FLASHML_NODE_TOKENS", "node-a:tok-a,node-b:tok-b")
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_SERVICE_AUTOINIT", "1")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "ledger.db"))
    return TestClient(create_app())


def _auth(node_id):
    """The fixture's token for a node — `node-a` → `tok-a`."""
    return {"Authorization": f"Bearer tok-{node_id.rsplit('-', 1)[-1]}"}


def _register(client, node_id, headers=None):
    return client.post("/v1alpha1/nodes/register", json={
        "schema_version": "v1alpha1", "node_id": node_id,
        "kubernetes_node": node_id, "hostname": node_id,
        "capabilities": {"cpu_cores": 1, "memory_bytes": 1 << 30,
                         "gpus": [], "os": "linux", "architecture": "x86_64"},
    }, headers=_auth(node_id) if headers is None else headers)


def _claim(client, node_id, body_node_id=None):
    return client.post("/v1alpha1/leases/claim",
                       json={"node_id": body_node_id or node_id},
                       headers=_auth(node_id))


def _submit_one_task_job(client):
    r = client.post("/v1alpha1/jobs", json={
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": "scope"},
        "spec": {"execution": {"backend": "leases"},
                 "image": {"repository": "local/tier1", "tag": "dev"},
                 "workload": {"type": "hyperparameter_search",
                              "parameters": {"trials": [{"C": 1.0}]}}},
    })
    return r.json()["job_id"]


def test_write_without_a_token_is_401(client):
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json", content=b"{}")
    assert r.status_code == 401


def test_write_with_a_bad_token_is_401(client):
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer nope"})
    assert r.status_code == 401


def test_write_outside_any_live_lease_is_403(client):
    _register(client, "node-a")
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_the_lease_holder_may_write_under_its_own_task(client):
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 200


def test_another_node_may_not_write_under_that_task(client):
    """The core exploit: one volunteer overwriting another's committed result."""
    _register(client, "node-a")
    _register(client, "node-b")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-b"})
    assert r.status_code == 403


def test_the_holder_may_not_write_outside_its_task_prefix(client):
    """Guards the federated-averaging model key, which lives at
    jobs/{job}/round-NNN/weights.json — outside any task prefix."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    _claim(client, "node-a")
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/round-000/weights.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_a_sibling_prefix_does_not_satisfy_the_check(client):
    """`jobs/j/trial-000extra/` must not pass because it starts with
    `jobs/j/trial-000`. The prefix must end at a separator."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}extra/metrics.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_checkpoint_part_outside_a_live_lease_is_403(client):
    # Body shape matches RegisterPartRequest (attempt_id + nested part) —
    # the brief's flat payload doesn't match the pre-existing checkpoint
    # schema and would 422 before authorization ever runs.
    _register(client, "node-a")
    r = client.post(
        "/v1alpha1/jobs/other-job/tasks/trial-000/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": "jobs/other-job/trial-000/ckpt/step-10.json",
                       "sha256": "0" * 64, "size_bytes": 2}},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert r.status_code == 403


def test_checkpoint_part_without_a_token_is_401(client):
    r = client.post(
        "/v1alpha1/jobs/other-job/tasks/trial-000/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": "k", "sha256": "0" * 64, "size_bytes": 2}},
    )
    assert r.status_code == 401


def test_the_lease_holder_may_register_a_checkpoint_part(client):
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    key = f"jobs/{job_id}/{task_id}/ckpt/step-10.json"
    client.put(f"/v1alpha1/artifacts/{key}", content=b"{}",
               headers={"Authorization": "Bearer tok-a"})
    r = client.post(
        f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": key, "sha256": "0" * 64, "size_bytes": 2}},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert r.status_code in (200, 201)


def test_reads_are_not_scoped(client):
    """Drivers read other tasks' outputs and agents download shared inputs."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
               content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert client.get(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json").status_code == 200


# -- 4b-2: a malformed token denies, it does not crash ---------------------


def test_a_non_ascii_token_is_401_not_500(client):
    """`hmac.compare_digest` raises TypeError on non-ASCII. Unhandled that is
    an unauthenticated remote 500, and 500-vs-401 is an oracle.

    Sent as raw bytes because that is how it arrives on the wire: header
    values are bytes, and the ASGI server decodes them latin-1, so `0xF6`
    reaches `authenticate()` as the str `'ö'`. Passing a `str` here would
    only prove httpx refuses to encode it."""
    r = client.put("/v1alpha1/artifacts/jobs/j/trial-000/metrics.json",
                   content=b"{}",
                   headers=[(b"authorization", "Bearer tök-á".encode("latin-1"))])
    assert r.status_code == 401


# -- 4b-1: the authorization re-check after the body is read ---------------


def test_put_artifact_authorizes_twice(client, monkeypatch):
    """Structural: authorization must run BOTH before the body is buffered
    (so a stranger cannot make us read 200MB) and after (so a slow upload
    cannot outlive its own lease). One call is a TOCTOU window whose width
    the attacker chooses."""
    from flashruntime.service import modea

    calls = []
    real = modea._authorize_write

    def counting(state, manager, request, key):
        calls.append(key)
        return real(state, manager, request, key)

    monkeypatch.setattr(modea, "_authorize_write", counting)

    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    key = f"jobs/{job_id}/{task_id}/metrics.json"
    r = client.put(f"/v1alpha1/artifacts/{key}", content=b"{}",
                   headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 200
    assert calls == [key, key]


def test_a_lease_lost_while_the_body_uploads_cannot_write(client, monkeypatch):
    """The exploit itself: authorized at open, expired by the time the bytes
    land. Streaming a slow body in-process is impractical, so the lease is
    withdrawn between the two checks instead — which is exactly what the
    sweeper does to a stalled node."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    task_id = lease["payload"]["task_id"]
    key = f"jobs/{job_id}/{task_id}/metrics.json"

    state = client.app.state.modea
    manager = state.manager
    real = manager.live_leases_for_node
    seen = {"n": 0}

    def expiring(node_id, now=None):
        seen["n"] += 1
        return real(node_id, now) if seen["n"] == 1 else set()

    monkeypatch.setattr(manager, "live_leases_for_node", expiring)

    r = client.put(f"/v1alpha1/artifacts/{key}", content=b"evil",
                   headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403
    assert seen["n"] == 2  # the re-check really ran
    # and nothing was written: the committed result is untouched
    assert client.get(f"/v1alpha1/artifacts/{key}").status_code == 404


# -- 4b-3: the lease lifecycle endpoints ----------------------------------


def test_claim_resolves_node_id_from_the_token_not_the_body(client):
    """§5.2: the API resolves node_id from the token, never from the request
    body. node-b claiming as "node-a" gets a lease as ITSELF."""
    _register(client, "node-a")
    _register(client, "node-b")
    _submit_one_task_job(client)
    lease = _claim(client, "node-b", body_node_id="node-a").json()
    assert lease["node_id"] == "node-b"


def test_claim_without_a_token_is_401(client):
    _register(client, "node-a")
    _submit_one_task_job(client)
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"})
    assert r.status_code == 401


def test_another_node_may_not_fail_your_attempt(client):
    """The bypass one layer down: fail somebody else's attempt until the task
    requeues to you, then write your poison entirely legitimately."""
    _register(client, "node-a")
    _register(client, "node-b")
    _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    r = client.post(f"/v1alpha1/attempts/{lease['lease_id']}/fail",
                    json={"reason": "steal it"}, headers=_auth("node-b"))
    assert r.status_code == 403
    # the victim still holds it
    tasks = client.get(f"/v1alpha1/jobs/{lease['job_id']}/tasks").json()
    assert tasks[0]["state"] == "LEASED"
    assert tasks[0]["node_id"] == "node-a"


def test_another_node_may_not_complete_your_attempt(client):
    _register(client, "node-a")
    _register(client, "node-b")
    _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    r = client.post(f"/v1alpha1/attempts/{lease['lease_id']}/complete",
                    json={"output_sha256": "0" * 64}, headers=_auth("node-b"))
    assert r.status_code == 403


def test_another_node_may_not_heartbeat_your_attempt(client):
    """Holding somebody's lease open stops the sweeper ever requeueing it."""
    _register(client, "node-a")
    _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    r = client.post(f"/v1alpha1/attempts/{lease['lease_id']}/heartbeat",
                    headers=_auth("node-b"))
    assert r.status_code == 403
    assert client.post(f"/v1alpha1/attempts/{lease['lease_id']}/heartbeat",
                       headers=_auth("node-a")).status_code == 200


def test_lifecycle_endpoints_are_401_without_a_token(client):
    _register(client, "node-a")
    _submit_one_task_job(client)
    lease = _claim(client, "node-a").json()
    lid = lease["lease_id"]
    assert client.post(f"/v1alpha1/attempts/{lid}/fail",
                       json={"reason": "x"}).status_code == 401
    assert client.post(f"/v1alpha1/attempts/{lid}/complete",
                       json={"output_sha256": "0" * 64}).status_code == 401
    assert client.post(f"/v1alpha1/attempts/{lid}/heartbeat").status_code == 401


def test_register_resolves_node_id_from_the_token(client):
    """A node must not be able to register — and so be claimable — as another."""
    r = _register(client, "node-impostor", headers=_auth("node-b"))
    assert r.status_code == 200
    assert r.json()["node_id"] == "node-b"
    assert [n["node_id"] for n in client.get("/v1alpha1/nodes").json()] == ["node-b"]


def test_register_without_a_token_is_401(client):
    r = _register(client, "node-a", headers={})
    assert r.status_code == 401


def test_a_node_may_not_heartbeat_another_nodes_registration(client):
    _register(client, "node-a")
    body = {"schema_version": "v1alpha1", "node_id": "node-a",
            "timestamp": "2026-07-31T00:00:00Z"}
    assert client.post("/v1alpha1/nodes/node-a/heartbeat", json=body,
                       headers=_auth("node-b")).status_code == 403
    assert client.post("/v1alpha1/nodes/node-a/heartbeat", json=body,
                       headers=_auth("node-a")).status_code == 200


# -- 4b-4: the operator (driver) credential -------------------------------


def test_an_operator_may_write_the_round_weights_a_node_cannot(client):
    """The driver aggregates a round and writes jobs/{job}/round-NNN/
    weights.json — a key no lease can ever cover. Without a credential class
    for it, enforcing mode makes federated averaging unrunnable."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    _claim(client, "node-a")
    key = f"jobs/{job_id}/round-000/weights.json"
    assert client.put(f"/v1alpha1/artifacts/{key}", content=b"{}",
                      headers={"Authorization": "Bearer tok-a"}).status_code == 403
    assert client.put(f"/v1alpha1/artifacts/{key}", content=b"{}",
                      headers={"Authorization": "Bearer op-tok"}).status_code == 200


def test_an_operator_may_register_a_checkpoint_part_it_holds_no_lease_for(client):
    r = client.post(
        "/v1alpha1/jobs/any-job/tasks/trial-000/checkpoints/parts",
        json={"attempt_id": "at1", "step": 10,
              "part": {"key": "k", "sha256": "0" * 64, "size_bytes": 2}},
        headers={"Authorization": "Bearer op-tok"},
    )
    assert r.status_code in (200, 201)


def test_an_operator_token_is_not_a_node_identity(client):
    """An operator holds no lease, so it must not be able to claim work or
    drive somebody else's attempt — that would re-open the requeue attack."""
    _register(client, "node-a")
    _submit_one_task_job(client)
    op = {"Authorization": "Bearer op-tok"}
    assert client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"},
                       headers=op).status_code == 401
    lease = _claim(client, "node-a").json()
    assert client.post(f"/v1alpha1/attempts/{lease['lease_id']}/fail",
                       json={"reason": "x"}, headers=op).status_code == 401
