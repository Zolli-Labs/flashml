"""Operator-asserted node identity: `X-FlashML-On-Behalf-Of`.

The deployed topology puts this coordinator on a private network behind the
cloud API, so agent traffic is *forwarded*: the API holds one operator token
and speaks for many machines. Forwarding naively — operator token, nothing
else — would erase the lease scoping the write-scope suite proves, because
every write would arrive as an unscoped driver.

So the API names the machine it is acting for in a header, and the
coordinator honours that header **only** from an operator credential. What it
does then is not new authorization: it is exactly the authorization a direct
node call would have received. Delegation therefore only ever *narrows* the
operator's reach.

All three authorization surfaces are pinned here — artifacts (keyed by
prefix), checkpoints (keyed by the `(job_id, task_id)` pair), and the lease
lifecycle (keyed by the lease holder). A rule that held in one and not the
others is the seam-shaped bug this codebase has already been bitten by twice.

The lifecycle half carries the load-bearing negative: `_require_lease_holder`
exists because a node that could `fail` another node's attempt would get the
task requeued to *itself* and then write its poison entirely legitimately.
Delegation must not soften that by one inch for a node token — only an
operator, and only for the machine it names.
"""

import pytest
from fastapi.testclient import TestClient

from flashruntime.service.app import create_app

HEADER = "X-FlashML-On-Behalf-Of"


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
    return {"Authorization": f"Bearer tok-{node_id.rsplit('-', 1)[-1]}"}


OP = {"Authorization": "Bearer op-tok"}


def _register(client, node_id, headers=None, body_node_id=None):
    """Register `node_id`. `headers` overrides the credential (so an operator
    can forward one) and `body_node_id` plants a different id in the body, to
    prove the body is never authoritative."""
    return client.post(
        "/v1alpha1/nodes/register",
        json={
            "schema_version": "v1alpha1",
            "node_id": node_id if body_node_id is None else body_node_id,
            "kubernetes_node": node_id,
            "hostname": node_id,
            "capabilities": {
                "cpu_cores": 1,
                "memory_bytes": 1 << 30,
                "gpus": [],
                "os": "linux",
                "architecture": "x86_64",
            },
        },
        headers=_auth(node_id) if headers is None else headers,
    )


def _submit_one_task_job(client):
    r = client.post(
        "/v1alpha1/jobs",
        json={
            "apiVersion": "flashml.dev/v1alpha1",
            "kind": "Job",
            "metadata": {"name": "delegation"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {
                    "type": "hyperparameter_search",
                    "parameters": {"trials": [{"C": 1.0}]},
                },
            },
        },
    )
    return r.json()["job_id"]


def _claim(client, node_id):
    return client.post(
        "/v1alpha1/leases/claim", json={"node_id": node_id}, headers=_auth(node_id)
    ).json()


@pytest.fixture()
def leased(client):
    """node-a holds a live lease; node-b is registered and holds nothing."""
    _register(client, "node-a")
    _register(client, "node-b")
    job_id = _submit_one_task_job(client)
    lease = _claim(client, "node-a")
    return job_id, lease["payload"]["task_id"]


def _put(client, key, headers, body=b"{}"):
    return client.put(f"/v1alpha1/artifacts/{key}", content=body, headers=headers)


def _part(client, job_id, task_id, headers, key="k"):
    return client.post(
        f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/parts",
        json={
            "attempt_id": "at1",
            "step": 10,
            "part": {"key": key, "sha256": "0" * 64, "size_bytes": 2},
        },
        headers=headers,
    )


def _commit(client, job_id, task_id, headers):
    return client.post(
        f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/commit",
        json={
            "attempt_id": "at1",
            "step": 10,
            "expected_parts": [],
            "storage_prefix": "s",
        },
        headers=headers,
    )


# -- Rule 1: operator + header → authorized as that node -------------------


def test_operator_on_behalf_of_the_holder_may_write_its_artifact(client, leased):
    """The forwarded case: exactly the write node-a could have made itself."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}/metrics.json", {**OP, HEADER: "node-a"})
    assert r.status_code == 200


def test_operator_on_behalf_of_the_holder_may_register_a_checkpoint_part(client, leased):
    job_id, task_id = leased
    r = _part(client, job_id, task_id, {**OP, HEADER: "node-a"})
    assert r.status_code in (200, 201)


def test_operator_on_behalf_of_the_holder_may_commit_a_checkpoint(client, leased):
    """`commit` shares the helper with `parts`; pin it too so the pair cannot
    drift apart."""
    job_id, task_id = leased
    r = _commit(client, job_id, task_id, {**OP, HEADER: "node-a"})
    assert r.status_code != 403


# -- Rule 2: delegation must NARROW, never widen ---------------------------


def test_delegating_to_a_node_with_no_lease_is_403(client, leased):
    """node-b holds nothing. The operator could have written this key with no
    header at all — asserting an identity gives up that reach."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}/metrics.json", {**OP, HEADER: "node-b"})
    assert r.status_code == 403


def test_delegating_outside_the_holders_prefix_is_403(client, leased):
    """The sharpest form: node-a IS a live holder, but this key falls outside
    its lease. The driver's own unscoped key becomes unreachable the moment it
    speaks as a node."""
    job_id, _ = leased
    r = _put(client, f"jobs/{job_id}/round-000/weights.json", {**OP, HEADER: "node-a"})
    assert r.status_code == 403


def test_delegating_to_another_job_is_403(client, leased):
    """Scope is per (job, task), not per node — a lease on one job never
    authorizes a write to another."""
    _, task_id = leased
    r = _put(client, f"jobs/other-job/{task_id}/metrics.json", {**OP, HEADER: "node-a"})
    assert r.status_code == 403


def test_delegating_a_checkpoint_to_a_node_with_no_lease_is_403(client, leased):
    job_id, task_id = leased
    assert _part(client, job_id, task_id, {**OP, HEADER: "node-b"}).status_code == 403
    assert _commit(client, job_id, task_id, {**OP, HEADER: "node-b"}).status_code == 403


def test_delegating_a_checkpoint_to_another_task_is_403(client, leased):
    """The operator's unscoped checkpoint reach (`any-job/trial-000`) is gone
    once it asserts an identity."""
    assert _part(client, "any-job", "trial-000", {**OP, HEADER: "node-a"}).status_code == 403


def test_a_sibling_prefix_still_does_not_satisfy_a_delegated_write(client, leased):
    """`jobs/j/trial-000extra/` must not pass on the delegated path either."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}extra/metrics.json", {**OP, HEADER: "node-a"})
    assert r.status_code == 403


def test_a_delegated_write_dies_with_the_lease(client, leased):
    """Delegation consults the live lease table at request time — it is not a
    grant that outlives the lease it was scoped to."""
    job_id, task_id = leased
    key = f"jobs/{job_id}/{task_id}/metrics.json"
    assert _put(client, key, {**OP, HEADER: "node-a"}).status_code == 200
    manager = client.app.state.modea.manager
    lease = next(r.active_lease for r in manager.records(job_id) if r.spec.task_id == task_id)
    manager.fail(lease.lease_id, "expired")
    assert _put(client, key, {**OP, HEADER: "node-a"}).status_code == 403


# -- Rule 3: a node token never gets the header honoured -------------------


def test_a_node_token_cannot_borrow_another_nodes_identity(client, leased):
    """The whole point of the operator gate. node-b asserting node-a is still
    node-b, and node-b holds no lease here."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}/metrics.json", {**_auth("node-b"), HEADER: "node-a"})
    assert r.status_code == 403


def test_a_node_token_with_a_header_remains_itself(client, leased):
    """Ignored, not rejected: node-a naming node-b still writes its OWN key."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}/metrics.json", {**_auth("node-a"), HEADER: "node-b"})
    assert r.status_code == 200


def test_a_node_token_cannot_borrow_an_identity_for_a_checkpoint(client, leased):
    job_id, task_id = leased
    r = _part(client, job_id, task_id, {**_auth("node-b"), HEADER: "node-a"})
    assert r.status_code == 403


def test_no_token_plus_a_header_is_still_401(client, leased):
    """The header is not a credential. Unauthenticated stays unauthenticated."""
    job_id, task_id = leased
    assert _put(client, f"jobs/{job_id}/{task_id}/m.json", {HEADER: "node-a"}).status_code == 401
    assert _part(client, job_id, task_id, {HEADER: "node-a"}).status_code == 401


def test_a_bad_token_plus_a_header_is_still_401(client, leased):
    job_id, task_id = leased
    bad = {"Authorization": "Bearer nope", HEADER: "node-a"}
    assert _put(client, f"jobs/{job_id}/{task_id}/m.json", bad).status_code == 401
    assert _part(client, job_id, task_id, bad).status_code == 401


# -- Rule 4: no header → unscoped operator, unchanged ----------------------


def test_an_operator_without_the_header_is_still_unscoped(client, leased):
    """Plan 2's behaviour, untouched: the fedavg driver writes the round
    weights, a key no lease can ever cover."""
    job_id, _ = leased
    assert _put(client, f"jobs/{job_id}/round-000/weights.json", OP).status_code == 200
    assert _part(client, "any-job", "trial-000", OP).status_code in (200, 201)


# -- Rule 5: an unknown node --------------------------------------------


def test_delegating_to_an_unknown_node_is_403(client, leased):
    """Nothing was ever leased to `ghost`, so it authorizes nothing. 403 and
    not 404: the answer must not distinguish "no such node" from "no lease",
    which would enumerate the pool."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}/metrics.json", {**OP, HEADER: "ghost"})
    assert r.status_code == 403
    assert _part(client, job_id, task_id, {**OP, HEADER: "ghost"}).status_code == 403


def test_delegating_to_the_operators_own_name_is_403(client, leased):
    """`driver` is an operator NAME, not a node identity — auth.py is explicit
    that an operator is never a node. Naming it must not launder the token
    back into unscoped reach."""
    job_id, task_id = leased
    r = _put(client, f"jobs/{job_id}/{task_id}/metrics.json", {**OP, HEADER: "driver"})
    assert r.status_code == 403


# -- adversarial: the header's own edge cases ------------------------------


def test_an_empty_header_value_fails_closed(client, leased):
    """An API bug that emits an empty header must NOT read as "no header" and
    silently restore unscoped operator reach. Deny."""
    job_id, _ = leased
    r = _put(client, f"jobs/{job_id}/round-000/weights.json", {**OP, HEADER: ""})
    assert r.status_code == 403
    assert _part(client, "any-job", "trial-000", {**OP, HEADER: "   "}).status_code == 403


def test_two_header_values_are_refused_rather_than_picked(client, leased):
    """A forwarding proxy appends; an agent may already have sent one. Which
    value wins would then be decided by header order — attacker-controllable.
    Refuse the request instead of choosing."""
    job_id, task_id = leased
    headers = [
        (b"authorization", b"Bearer op-tok"),
        (HEADER.lower().encode(), b"node-b"),
        (HEADER.lower().encode(), b"node-a"),
    ]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/m.json",
                   content=b"{}", headers=headers)
    assert r.status_code == 400
    assert client.get(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/m.json").status_code == 404


def test_two_header_values_from_a_node_token_are_still_just_ignored(client, leased):
    """For a non-operator the header is not authoritative, so it is not
    parsed either — a volunteer cannot turn its own writes into 400s, and
    more importantly cannot probe the delegation path at all."""
    job_id, task_id = leased
    headers = [
        (b"authorization", b"Bearer tok-a"),
        (HEADER.lower().encode(), b"node-b"),
        (HEADER.lower().encode(), b"ghost"),
    ]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/m.json",
                   content=b"{}", headers=headers)
    assert r.status_code == 200


def test_the_delegated_identity_is_read_identically_on_both_authorize_calls(client, leased):
    """`put_artifact` authorizes twice — before and after buffering the body.
    Both calls must resolve the SAME identity, or the TOCTOU re-check is
    checking a different caller than the first pass admitted."""
    from flashruntime.service import modea

    seen = []
    real = modea._write_identity

    def recording(state, request):
        out = real(state, request)
        seen.append(out)
        return out

    import unittest.mock as _m

    job_id, task_id = leased
    with _m.patch.object(modea, "_write_identity", recording):
        r = _put(client, f"jobs/{job_id}/{task_id}/m.json", {**OP, HEADER: "node-a"})
    assert r.status_code == 200
    assert seen == ["node-a", "node-a"]


def test_delegation_is_inert_when_enforcement_is_off(client, tmp_path, monkeypatch):
    """Self-hosted default (CLAUDE.md rule 4): no credentials, no scoping —
    and so no operator, which means no honoured header. Nothing changes."""
    monkeypatch.delenv("FLASHML_OPERATOR_TOKENS", raising=False)
    monkeypatch.delenv("FLASHML_NODE_TOKENS", raising=False)
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "open-artifacts"))
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_SERVICE_AUTOINIT", "1")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "open-ledger.db"))
    open_client = TestClient(create_app())
    r = _put(open_client, "jobs/j/trial-000/m.json", {HEADER: "whoever"})
    assert r.status_code == 200


# ==========================================================================
# The lease LIFECYCLE: register / claim / node heartbeat / attempts.
#
# Same rule, same reader. The API in front of the coordinator is the only
# component that can authenticate a machine at all (the coordinator has no
# database; machines are enrolled at runtime), so it must be able to proxy
# these. It does so by naming the machine — never by acting as nobody.
# ==========================================================================


def _dup(header_name, first, second):
    """Two values of one header, as a proxy that failed to strip the inner
    copy would produce."""
    return [
        (b"authorization", b"Bearer op-tok"),
        (header_name.lower().encode(), first.encode()),
        (header_name.lower().encode(), second.encode()),
    ]


@pytest.fixture()
def pool(client):
    """node-a and node-b registered, one claimable task, nothing claimed."""
    _register(client, "node-a")
    _register(client, "node-b")
    return _submit_one_task_job(client)


@pytest.fixture()
def attempt(client, pool):
    """node-a holds the only lease. Returns (job_id, task_id, lease_id)."""
    lease = _claim(client, "node-a")
    return pool, lease["task_id"], lease["lease_id"]


def _claim_as(client, headers, body_node_id="whoever"):
    return client.post(
        "/v1alpha1/leases/claim", json={"node_id": body_node_id}, headers=headers
    )


def _node_hb(client, node_id, headers):
    return client.post(
        f"/v1alpha1/nodes/{node_id}/heartbeat",
        json={"schema_version": "v1alpha1", "node_id": node_id},
        headers=headers,
    )


def _att_hb(client, lease_id, headers):
    return client.post(f"/v1alpha1/attempts/{lease_id}/heartbeat", headers=headers)


def _att_complete(client, lease_id, headers, sha="0" * 64):
    return client.post(
        f"/v1alpha1/attempts/{lease_id}/complete",
        json={"output_sha256": sha},
        headers=headers,
    )


def _att_fail(client, lease_id, headers, reason="nope"):
    return client.post(
        f"/v1alpha1/attempts/{lease_id}/fail",
        json={"reason": reason},
        headers=headers,
    )


def _still_leased_to(client, job_id, task_id, node_id):
    """The task is STILL held by `node_id` — i.e. nothing requeued it."""
    tasks = client.get(f"/v1alpha1/jobs/{job_id}/tasks").json()
    row = next(t for t in tasks if t["task_id"] == task_id)
    return row["state"] == "LEASED" and row["node_id"] == node_id


# -- POST /nodes/register --------------------------------------------------


def test_operator_may_register_a_machine_it_names(client):
    """Enrolment is the API's job: it authenticated the machine, the
    coordinator cannot. `node-c` has no token here at all."""
    r = _register(client, "node-c", headers={**OP, HEADER: "node-c"})
    assert r.status_code == 200
    assert r.json()["node_id"] == "node-c"
    assert "node-c" in [n["node_id"] for n in client.get("/v1alpha1/nodes").json()]


def test_a_delegated_registration_still_ignores_the_body_node_id(client):
    """The header decides; the body is as non-authoritative as it has always
    been. A compromised agent that got its body forwarded verbatim must not
    be able to enrol itself as somebody else."""
    r = _register(client, "node-c", headers={**OP, HEADER: "node-c"}, body_node_id="node-a")
    assert r.json()["node_id"] == "node-c"
    assert "node-a" not in [n["node_id"] for n in client.get("/v1alpha1/nodes").json()]


def test_a_node_token_registering_with_a_header_remains_itself(client):
    """The operator gate, on the enrolment path: node-b naming node-a enrols
    node-b."""
    r = _register(client, "node-b", headers={**_auth("node-b"), HEADER: "node-a"})
    assert r.json()["node_id"] == "node-b"
    assert "node-a" not in [n["node_id"] for n in client.get("/v1alpha1/nodes").json()]


def test_an_operator_without_the_header_cannot_register(client):
    """A registration with no machine behind it would create a node named by
    the body — the exact spoof the credential-over-body rule closes. 401, the
    pre-delegation answer: an operator token is simply not a node identity."""
    assert _register(client, "node-c", headers=OP).status_code == 401
    assert client.get("/v1alpha1/nodes").json() == []


def test_duplicate_headers_are_refused_on_register(client):
    r = client.post(
        "/v1alpha1/nodes/register",
        json={
            "schema_version": "v1alpha1",
            "node_id": "node-c",
            "kubernetes_node": "node-c",
            "hostname": "node-c",
            "capabilities": {
                "cpu_cores": 1,
                "memory_bytes": 1 << 30,
                "gpus": [],
                "os": "linux",
                "architecture": "x86_64",
            },
        },
        headers=_dup(HEADER, "node-c", "node-a"),
    )
    assert r.status_code == 400
    assert client.get("/v1alpha1/nodes").json() == []


# -- POST /leases/claim ----------------------------------------------------


def test_operator_may_claim_for_the_machine_it_names(client, pool):
    r = _claim_as(client, {**OP, HEADER: "node-a"})
    assert r.status_code == 200
    assert r.json()["node_id"] == "node-a"


def test_a_delegated_claim_still_overwrites_a_spoofed_body_node_id(client, pool):
    """The body said node-b. The lease must go to the header's node-a —
    otherwise a forwarded body would pick who gets the work."""
    r = _claim_as(client, {**OP, HEADER: "node-a"}, body_node_id="node-b")
    assert r.json()["node_id"] == "node-a"


def test_a_node_token_claiming_with_a_header_remains_itself(client, pool):
    """node-b naming node-a still claims as node-b — so the write scoping
    that follows confines it to node-b."""
    r = _claim_as(client, {**_auth("node-b"), HEADER: "node-a"}, body_node_id="node-a")
    assert r.json()["node_id"] == "node-b"


def test_an_operator_without_the_header_cannot_claim(client, pool):
    """No identity, no claim. Allowing it would hand out a lease to whatever
    the body said — and leases are what authorize writes."""
    assert _claim_as(client, OP, body_node_id="node-a").status_code == 401
    assert _claim_as(client, OP, body_node_id="node-a").json()["detail"]


def test_delegating_a_claim_to_an_unknown_node_is_403(client, pool):
    """`ghost` never registered, so it is not a machine this pool knows —
    the same answer a direct call from an unregistered node gets."""
    assert _claim_as(client, {**OP, HEADER: "ghost"}).status_code == 403


def test_an_empty_header_cannot_claim(client, pool):
    """An API bug emitting a blank header must not read as "no header" — and
    "no header" is itself refused here, so this fails closed twice over."""
    assert _claim_as(client, {**OP, HEADER: ""}).status_code == 403
    assert _claim_as(client, {**OP, HEADER: "   "}).status_code == 403


def test_duplicate_headers_are_refused_on_claim(client, pool):
    r = client.post("/v1alpha1/leases/claim", json={"node_id": "x"},
                    headers=_dup(HEADER, "node-b", "node-a"))
    assert r.status_code == 400
    # ...and nothing was leased on the way out.
    assert client.get(f"/v1alpha1/jobs/{pool}/tasks").json()[0]["state"] == "PENDING"


# -- POST /nodes/{id}/heartbeat -------------------------------------------


def test_operator_may_heartbeat_the_machine_it_names(client, pool):
    assert _node_hb(client, "node-a", {**OP, HEADER: "node-a"}).status_code == 200


def test_operator_cannot_heartbeat_a_node_it_did_not_name(client, pool):
    """"A node may only heartbeat itself" — and under delegation "itself" is
    the asserted machine. A heartbeat marks a node online, so forging one for
    an arbitrary id would falsify the pool view the scheduler reads."""
    assert _node_hb(client, "node-a", {**OP, HEADER: "node-b"}).status_code == 403


def test_an_operator_without_the_header_cannot_heartbeat_a_node(client, pool):
    assert _node_hb(client, "node-a", OP).status_code == 401


def test_a_node_token_heartbeating_with_a_header_remains_itself(client, pool):
    assert _node_hb(client, "node-a", {**_auth("node-b"), HEADER: "node-a"}).status_code == 403
    assert _node_hb(client, "node-b", {**_auth("node-b"), HEADER: "node-a"}).status_code == 200


def test_a_delegated_heartbeat_for_an_unknown_node_matches_the_direct_call(client, pool):
    """Parity, not a new rule: an unregistered node gets 404 "register first"
    whether it calls directly or the API calls for it. No enumeration concern
    — an operator can already GET /nodes."""
    assert _node_hb(client, "ghost", {**OP, HEADER: "ghost"}).status_code == 404


# -- POST /attempts/{lease_id}/{heartbeat,complete,fail} -------------------


def test_operator_may_heartbeat_an_attempt_for_the_holder(client, attempt):
    _, _, lease_id = attempt
    assert _att_hb(client, lease_id, {**OP, HEADER: "node-a"}).status_code == 200


def test_operator_may_fail_an_attempt_for_the_holder(client, attempt):
    """The forwarded worker gave up. Exactly the call node-a could have made
    itself — and the task requeues, as it should."""
    job_id, task_id, lease_id = attempt
    assert _att_fail(client, lease_id, {**OP, HEADER: "node-a"}).status_code == 200
    assert not _still_leased_to(client, job_id, task_id, "node-a")


def test_operator_may_complete_an_attempt_for_the_holder(client, attempt):
    """End to end on the delegated path: the operator writes node-a's output
    under node-a's lease scope, then commits it with the real hash."""
    import hashlib

    job_id, task_id, lease_id = attempt
    body = b'{"loss": 0.1}'
    assert _put(client, f"jobs/{job_id}/{task_id}/metrics.json",
                {**OP, HEADER: "node-a"}, body=body).status_code == 200
    r = _att_complete(client, lease_id, {**OP, HEADER: "node-a"},
                      sha=hashlib.sha256(body).hexdigest())
    assert r.status_code == 200 and r.json()["accepted"] is True


def test_operator_cannot_drive_an_attempt_of_a_node_it_did_not_name(client, attempt):
    """node-b holds nothing. Naming it buys the operator nothing — the whole
    point of "authorized exactly as that node would have been"."""
    job_id, task_id, lease_id = attempt
    hdr = {**OP, HEADER: "node-b"}
    assert _att_hb(client, lease_id, hdr).status_code == 403
    assert _att_complete(client, lease_id, hdr).status_code == 403
    assert _att_fail(client, lease_id, hdr).status_code == 403
    assert _still_leased_to(client, job_id, task_id, "node-a")


def test_an_operator_without_the_header_cannot_drive_an_attempt(client, attempt):
    """The refusal-by-omission case, and the one that matters most. A bare
    operator token has no identity to check the lease against; letting it
    through would give the API the power to fail ANY lease, which is the
    requeue-steal primitive wearing a different hat. 401 is the answer this
    endpoint already gave an operator before delegation existed — pinned by
    tests/test_service_write_scope.py, and unchanged here on purpose."""
    job_id, task_id, lease_id = attempt
    assert _att_hb(client, lease_id, OP).status_code == 401
    assert _att_complete(client, lease_id, OP).status_code == 401
    assert _att_fail(client, lease_id, OP).status_code == 401
    assert _still_leased_to(client, job_id, task_id, "node-a")


def test_node_b_still_cannot_fail_node_as_attempt(client, attempt):
    """THE regression gate. Fail somebody's attempt and the task requeues to
    you; then every poisoned byte you write is inside your own lease scope
    and passes authorization. Delegation must not have opened this by a
    crack — with the header or without it, node-b is node-b."""
    job_id, task_id, lease_id = attempt
    assert _att_fail(client, lease_id, _auth("node-b")).status_code == 403
    assert _att_fail(client, lease_id, {**_auth("node-b"), HEADER: "node-a"}).status_code == 403
    assert _att_complete(client, lease_id, {**_auth("node-b"), HEADER: "node-a"}).status_code == 403
    assert _att_hb(client, lease_id, {**_auth("node-b"), HEADER: "node-a"}).status_code == 403
    assert _still_leased_to(client, job_id, task_id, "node-a")


def test_delegating_an_attempt_to_an_unknown_node_is_403(client, attempt):
    _, _, lease_id = attempt
    assert _att_fail(client, lease_id, {**OP, HEADER: "ghost"}).status_code == 403


def test_delegating_an_attempt_to_the_operators_own_name_is_403(client, attempt):
    """`driver` names an operator, never a machine — it holds no lease and
    must not launder the token back into unscoped lifecycle reach."""
    _, _, lease_id = attempt
    assert _att_fail(client, lease_id, {**OP, HEADER: "driver"}).status_code == 403


def test_an_empty_header_cannot_drive_an_attempt(client, attempt):
    _, _, lease_id = attempt
    assert _att_fail(client, lease_id, {**OP, HEADER: ""}).status_code == 403
    assert _att_fail(client, lease_id, {**OP, HEADER: "  "}).status_code == 403


def test_duplicate_headers_are_refused_on_an_attempt(client, attempt):
    """Refused, not first-wins: header order is exactly what a forwarding bug
    hands to an attacker."""
    job_id, task_id, lease_id = attempt
    r = client.post(f"/v1alpha1/attempts/{lease_id}/fail", json={"reason": "x"},
                    headers=_dup(HEADER, "node-b", "node-a"))
    assert r.status_code == 400
    assert _still_leased_to(client, job_id, task_id, "node-a")


def test_an_unknown_lease_id_is_403_for_a_delegated_caller_too(client, attempt):
    """No 404/403 split on the delegated path either — that difference
    enumerates live lease ids."""
    assert _att_fail(client, "no-such-lease", {**OP, HEADER: "node-a"}).status_code == 403


def test_the_lifecycle_reads_the_header_through_the_same_single_helper(client, attempt):
    """One reader of the header in the whole repo. If a lifecycle endpoint
    ever grew its own parse, this is the test that notices — the rule
    enforced in one place and not another is the seam this codebase has been
    bitten by three times."""
    from flashruntime.service import modea

    seen = []
    real = modea._write_identity

    def recording(state, request):
        out = real(state, request)
        seen.append(out)
        return out

    import unittest.mock as _m

    _, _, lease_id = attempt
    with _m.patch.object(modea, "_write_identity", recording):
        assert _att_hb(client, lease_id, {**OP, HEADER: "node-a"}).status_code == 200
        assert _node_hb(client, "node-a", {**OP, HEADER: "node-a"}).status_code == 200
        assert _claim_as(client, {**OP, HEADER: "node-a"}).status_code in (200, 204)
    assert seen == ["node-a", "node-a", "node-a"]


def test_the_lifecycle_is_untouched_when_enforcement_is_off(client, tmp_path, monkeypatch):
    """Self-hosted default (CLAUDE.md rule 4): no credentials, no operator,
    no honoured header — and no new refusals either. A bare claim with no
    token must still work."""
    monkeypatch.delenv("FLASHML_OPERATOR_TOKENS", raising=False)
    monkeypatch.delenv("FLASHML_NODE_TOKENS", raising=False)
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "open-artifacts"))
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_SERVICE_AUTOINIT", "1")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "open-ledger.db"))
    open_client = TestClient(create_app())
    assert _register(open_client, "node-z", headers={HEADER: "whoever"}).json()["node_id"] == "node-z"
    job_id = _submit_one_task_job(open_client)
    r = _claim_as(open_client, {HEADER: "whoever"}, body_node_id="node-z")
    assert r.status_code == 200 and r.json()["node_id"] == "node-z"
    lease_id = r.json()["lease_id"]
    assert _att_hb(open_client, lease_id, {}).status_code == 200
    assert _att_fail(open_client, lease_id, {}).status_code == 200
    assert job_id
