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

Both write-authorization surfaces are pinned here — artifacts (keyed by
prefix) and checkpoints (keyed by the `(job_id, task_id)` pair). A rule that
held in one and not the other is the seam-shaped bug this codebase has
already been bitten by twice.
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


def _register(client, node_id):
    return client.post(
        "/v1alpha1/nodes/register",
        json={
            "schema_version": "v1alpha1",
            "node_id": node_id,
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
        headers=_auth(node_id),
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
