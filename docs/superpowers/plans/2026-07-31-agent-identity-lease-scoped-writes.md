# Agent Identity and Lease-Scoped Writes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give every worker machine its own revocable credential, and confine what a machine may write to the tasks it currently holds a lease on — so a public coordinator cannot be used by one volunteer to overwrite another job's results.

**Architecture:** The coordinator gains a pluggable `NodeAuthenticator` seam (open by default, so the self-hosted profile is unchanged) that resolves a bearer token to a `node_id`. Every artifact and checkpoint **write** then derives its permitted key prefixes from that node's *live leases* and rejects anything else with 403. flashnode gains a credential store and sends its token on every request.

**Tech Stack:** Python ≥3.10, FastAPI, pydantic, pytest.

This is **Plan 2 of 7** for M1, implementing §5.2 and §6.1–§6.2 of
`flashml-cloud/docs/superpowers/specs/2026-07-31-deployed-multi-user-poc-design.md`.

## Why this plan gates every other one

`HANDOFF.md` risk #2 and the 2026-07-29 deferred follow-up #1: `PUT /v1alpha1/artifacts/{key}` (`service/modea.py:503`) and the checkpoint endpoints accept **any key from any caller**, with no authentication whatsoever — only `_safe_key` path containment and a size cap. The sha256 commit check is no defense, because the attacker supplies both the file and the hash.

The Plan 1 whole-branch review sharpened this: the authoritative federated-averaging model is written to `jobs/{job_id}/round-{r:03d}/weights.json`, a predictable key in that same writable namespace. One volunteer can overwrite **everyone's** model. That is a different and worse thing than the accepted "a node may lie about its own result" gap.

Standing guidance is "do not put the current coordinator on a public IP for longer than a demo." **Nothing in M1 may face the internet until this plan lands.**

## Global Constraints

- **The self-hosted profile must keep working unchanged.** `flashruntime/CLAUDE.md` rule 4: "the runtime must stay useful without the cloud — self-hosted local coordinator is a first-class mode, not a demo shim." With no authenticator configured, behavior is exactly as today. Every existing test must pass untouched.
- **Security fields fail closed** (`CLAUDE.md` rule 3). Where a value is absent or type-confused, deny.
- **Reads stay open; only writes are scoped.** The driver reads other tasks' outputs and agents download shared inputs — scoping reads would break both. Read-side authorization is the cloud API's job (Plan 3), by job ownership.
- flashruntime imports nothing from flashnode or flashml-cloud, ever (`CLAUDE.md` rule 1).
- Never run the coordinator with more than one uvicorn worker (`HANDOFF.md` risk #5).
- **Run tests with the venv on `PATH`**, or `test_examples_e2e.py::test_sklearn_sweep_end_to_end` fails spuriously (`LocalLauncher` spawns a bare `python`):

      cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest

  Inner loop (the torchrun tests take minutes):

      PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q \
        --ignore=tests/test_examples_e2e.py --ignore=tests/test_gpu_e2e.py

- **Baselines — do not reduce:** flashruntime **417 passed**, flashnode **75 passed, 1 skipped, 4 deselected**, workspace e2e **7 passed**.

## File Structure

| File | Responsibility |
|---|---|
| `flashruntime/flashruntime/service/auth.py` (new) | `NodeAuthenticator` protocol, `OpenAuthenticator`, `StaticTokenAuthenticator`, `authenticator_from_env()`. No FastAPI, no lease knowledge — pure token → node_id. |
| `flashruntime/flashruntime/leases/__init__.py` (modify) | `live_leases_for_node(node_id) -> set[tuple[str, str]]` — the authorization *fact* the service asks for. |
| `flashruntime/flashruntime/service/modea.py` (modify) | Resolve the caller, scope `PUT /artifacts`, fail-closed startup guard. |
| `flashruntime/flashruntime/service/checkpoints.py` (modify) | Same scoping for `parts` and `commit`. |
| `flashnode/flashnode/identity/credentials.py` (new) | Read/write `~/.flashnode/credentials.json` at 0600. |
| `flashnode/flashnode/agent/cli.py` (modify) | `flashnode login --token`, `flashnode logout`. |
| `flashnode/flashnode/executor/client.py` (modify) | Send `Authorization: Bearer` when a credential exists. |

Keeping `auth.py` free of lease knowledge is the decomposition that matters: *who is calling* and *what may they write* are separate questions, tested separately, and Plan 3 replaces only the first.

---

### Task 1: The authenticator seam

**Files:**
- Create: `flashruntime/flashruntime/service/auth.py`
- Test: `flashruntime/tests/test_service_auth.py`

**Interfaces:**
- Produces:
  - `NodeAuthenticator` — Protocol with `authenticate(token: str | None) -> str | None` (returns `node_id`, or `None` to deny) and property `enforcing: bool`
  - `OpenAuthenticator()` — `enforcing = False`; `authenticate` always returns `None`
  - `StaticTokenAuthenticator(tokens: dict[str, str])` — `enforcing = True`; maps token → node_id
  - `authenticator_from_env(env: dict[str, str] | None = None) -> NodeAuthenticator`
  - `AuthConfigError(RuntimeError)`

`FLASHML_NODE_TOKENS` holds `node_id:token` pairs, comma-separated. Absent ⇒ `OpenAuthenticator`.

- [ ] **Step 1: Write the failing test**

```python
# flashruntime/tests/test_service_auth.py
import pytest

from flashruntime.service.auth import (
    AuthConfigError,
    OpenAuthenticator,
    StaticTokenAuthenticator,
    authenticator_from_env,
)


def test_open_authenticator_is_not_enforcing():
    a = OpenAuthenticator()
    assert a.enforcing is False
    assert a.authenticate("anything") is None
    assert a.authenticate(None) is None


def test_static_authenticator_maps_token_to_node():
    a = StaticTokenAuthenticator({"tok-a": "node-a", "tok-b": "node-b"})
    assert a.enforcing is True
    assert a.authenticate("tok-a") == "node-a"
    assert a.authenticate("tok-b") == "node-b"


def test_static_authenticator_denies_unknown_and_missing_tokens():
    a = StaticTokenAuthenticator({"tok-a": "node-a"})
    assert a.authenticate("nope") is None
    assert a.authenticate("") is None
    assert a.authenticate(None) is None


def test_static_authenticator_rejects_an_empty_token_at_construction():
    """An empty token would authenticate every caller sending no token."""
    with pytest.raises(AuthConfigError, match="empty token"):
        StaticTokenAuthenticator({"": "node-a"})


def test_token_comparison_is_constant_time():
    """Guard against a timing oracle on token contents. We cannot measure
    timing reliably in a unit test, so we pin the implementation choice."""
    import inspect

    from flashruntime.service import auth

    assert "compare_digest" in inspect.getsource(auth.StaticTokenAuthenticator)


def test_env_without_tokens_yields_open():
    assert authenticator_from_env({}).enforcing is False


def test_env_with_tokens_yields_enforcing():
    a = authenticator_from_env({"FLASHML_NODE_TOKENS": "node-a:tok-a,node-b:tok-b"})
    assert a.enforcing is True
    assert a.authenticate("tok-a") == "node-a"


def test_env_tolerates_whitespace_and_trailing_commas():
    a = authenticator_from_env({"FLASHML_NODE_TOKENS": " node-a:tok-a , node-b:tok-b ,"})
    assert a.authenticate("tok-a") == "node-a"
    assert a.authenticate("tok-b") == "node-b"


def test_env_rejects_a_malformed_pair():
    with pytest.raises(AuthConfigError, match="node_id:token"):
        authenticator_from_env({"FLASHML_NODE_TOKENS": "garbage"})


def test_env_rejects_a_duplicate_token_across_nodes():
    """Two nodes sharing a token makes attribution — and revocation — a lie."""
    with pytest.raises(AuthConfigError, match="duplicate token"):
        authenticator_from_env({"FLASHML_NODE_TOKENS": "node-a:same,node-b:same"})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_auth.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashruntime.service.auth'`

- [ ] **Step 3: Write minimal implementation**

```python
# flashruntime/flashruntime/service/auth.py
"""Who is this caller? — the node-authentication seam.

Deliberately knows nothing about leases or authorization. This module answers
only "which node_id does this token belong to"; what that node may WRITE is a
separate question answered by the lease store (service/modea.py). Keeping them
apart is what lets the cloud replace authentication (Plan 3) without touching
authorization.

Default is OPEN: `flashruntime/CLAUDE.md` rule 4 makes the self-hosted local
coordinator a first-class mode, and requiring credentials on a laptop would
break it. An operator exposing the coordinator turns enforcement on with
FLASHML_NODE_TOKENS, and FLASHML_REQUIRE_NODE_AUTH=1 makes startup fail closed
if they forget (see service/modea.py).
"""

from __future__ import annotations

import hmac
from typing import Protocol, runtime_checkable

__all__ = [
    "AuthConfigError",
    "NodeAuthenticator",
    "OpenAuthenticator",
    "StaticTokenAuthenticator",
    "authenticator_from_env",
]


class AuthConfigError(RuntimeError):
    """The authenticator configuration is unusable. Raised at construction —
    never at request time, so a misconfiguration cannot silently admit
    callers."""


@runtime_checkable
class NodeAuthenticator(Protocol):
    @property
    def enforcing(self) -> bool:
        """True when callers must present a valid token."""

    def authenticate(self, token: str | None) -> str | None:
        """Return the caller's node_id, or None to deny."""


class OpenAuthenticator:
    """Self-hosted default: no credentials, no scoping. Behavior identical to
    the coordinator before this seam existed."""

    enforcing = False

    def authenticate(self, token: str | None) -> str | None:  # noqa: ARG002
        return None


class StaticTokenAuthenticator:
    """Token → node_id from configuration. The self-hosted multi-machine case,
    and the test double for the cloud's authenticator."""

    enforcing = True

    def __init__(self, tokens: dict[str, str]):
        for token, node_id in tokens.items():
            if not token:
                raise AuthConfigError(
                    f"empty token configured for node {node_id!r}: an empty token "
                    "would authenticate every caller that sends none"
                )
        self._tokens = dict(tokens)

    def authenticate(self, token: str | None) -> str | None:
        if not token:
            return None
        # compare_digest against every candidate: a dict lookup leaks token
        # contents through timing, and the candidate set is small.
        for candidate, node_id in self._tokens.items():
            if hmac.compare_digest(candidate, token):
                return node_id
        return None


def authenticator_from_env(env: dict[str, str] | None = None) -> NodeAuthenticator:
    import os

    env = os.environ if env is None else env
    raw = (env.get("FLASHML_NODE_TOKENS") or "").strip()
    if not raw:
        return OpenAuthenticator()

    tokens: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if pair.count(":") != 1:
            raise AuthConfigError(
                f"FLASHML_NODE_TOKENS entry {pair!r} is not 'node_id:token'"
            )
        node_id, token = (p.strip() for p in pair.split(":"))
        if token in tokens:
            raise AuthConfigError(
                f"duplicate token shared by {tokens[token]!r} and {node_id!r}: "
                "shared tokens make attribution and revocation meaningless"
            )
        tokens[token] = node_id
    if not tokens:
        return OpenAuthenticator()
    return StaticTokenAuthenticator(tokens)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_auth.py -v`
Expected: 10 passed

- [ ] **Step 5: Commit**

```bash
git add flashruntime/service/auth.py tests/test_service_auth.py
git commit -m "feat(auth): node authenticator seam, open by default

Answers only 'which node is this token' — what that node may write is a
separate question the lease store answers. Open by default so the
self-hosted profile (CLAUDE.md rule 4) is unchanged. Constant-time token
comparison; rejects empty and shared tokens at construction, since a shared
token makes attribution and revocation meaningless."
```

---

### Task 2: `live_leases_for_node` — the authorization fact

**Files:**
- Modify: `flashruntime/flashruntime/leases/__init__.py`
- Test: `flashruntime/tests/test_leases_scope.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `LeaseManager.live_leases_for_node(node_id: str) -> set[tuple[str, str]]` — `(job_id, task_id)` for every lease this node currently holds that has **not** expired.

Expiry matters: a node whose lease lapsed must lose write access immediately, or a straggler could overwrite the result of whoever reclaimed its task.

- [ ] **Step 1: Write the failing test**

```python
# flashruntime/tests/test_leases_scope.py
"""Which (job, task) pairs may a node write to right now?"""

import pytest

from flashruntime.leases import LeaseManager
from flashruntime.leases.store import InMemoryLeaseStore
from flashruntime.protocol.v1alpha1 import TaskSpec


class FakeClock:
    def __init__(self, t=0.0):
        self.t = t

    def __call__(self):
        return self.t


def _task(job_id: str, task_id: str) -> TaskSpec:
    return TaskSpec(
        task_id=task_id, job_id=job_id,
        commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
        max_attempts=4, lease_seconds=30.0, payload={},
    )


@pytest.fixture()
def manager():
    clock = FakeClock()
    m = LeaseManager(store=InMemoryLeaseStore(), clock=clock)
    m._test_clock = clock          # exposed for the tests below
    return m


def test_no_leases_means_no_scope(manager):
    assert manager.live_leases_for_node("node-a") == set()


def test_a_claimed_task_is_in_scope_for_its_holder(manager):
    manager.submit_tasks([_task("job-1", "task-000")])
    manager.claim("node-a")
    assert manager.live_leases_for_node("node-a") == {("job-1", "task-000")}


def test_another_node_gets_no_scope_from_it(manager):
    manager.submit_tasks([_task("job-1", "task-000")])
    manager.claim("node-a")
    assert manager.live_leases_for_node("node-b") == set()


def test_multiple_live_leases_all_appear(manager):
    manager.submit_tasks([_task("job-1", "task-000"), _task("job-2", "task-000")])
    manager.claim("node-a")
    manager.claim("node-a")
    assert manager.live_leases_for_node("node-a") == {
        ("job-1", "task-000"), ("job-2", "task-000"),
    }


def test_an_expired_lease_leaves_scope(manager):
    """A straggler whose lease lapsed must not be able to overwrite the
    result of whoever reclaimed its task."""
    manager.submit_tasks([_task("job-1", "task-000")])
    manager.claim("node-a")
    assert manager.live_leases_for_node("node-a") == {("job-1", "task-000")}
    manager._test_clock.t += 31.0
    assert manager.live_leases_for_node("node-a") == set()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_leases_scope.py -v`
Expected: FAIL — `AttributeError: 'LeaseManager' object has no attribute 'live_leases_for_node'`

If the fixture itself errors, read `tests/test_leases.py` and match how it constructs `LeaseManager` and submits tasks — that file is the authority on the current constructor and clock-injection shape. Adjust the fixture, not the assertions.

- [ ] **Step 3: Write minimal implementation**

Add to `LeaseManager` in `flashruntime/flashruntime/leases/__init__.py`:

```python
    def live_leases_for_node(self, node_id: str) -> set[tuple[str, str]]:
        """(job_id, task_id) this node may currently write to.

        Expiry is checked here rather than trusting the sweeper: a lease that
        lapsed a millisecond ago must already be out of scope, or a straggler
        could overwrite the result of whoever reclaimed its task. Fails closed
        on any record whose lease is missing or malformed.
        """
        now = self._clock()
        scope: set[tuple[str, str]] = set()
        for record in self._store.leased():
            lease = record.active_lease
            if lease is None or getattr(lease, "node_id", None) != node_id:
                continue
            expires = getattr(lease, "expires_at", None)
            if expires is None or expires <= now:
                continue
            scope.add((record.spec.job_id, record.spec.task_id))
        return scope
```

Read the surrounding class first: the clock attribute may be named `self._clock` or `self.clock`, and `Lease.expires_at` may be a float or a datetime. Match what is actually there — if `expires_at` is a datetime, compare against a datetime, not a float.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_leases_scope.py tests/test_leases.py tests/test_leases_sqlite.py -v`
Expected: 5 new passed; the two existing lease suites unchanged and green.

- [ ] **Step 5: Commit**

```bash
git add flashruntime/leases/__init__.py tests/test_leases_scope.py
git commit -m "feat(leases): live_leases_for_node, the write-authorization fact

Checks expiry inline rather than trusting the sweeper — a lease that lapsed
a millisecond ago must already be out of scope, or a straggler could
overwrite the result of whoever reclaimed its task."
```

---

### Task 3: Scope artifact writes to the caller's leases

**Files:**
- Modify: `flashruntime/flashruntime/service/modea.py` (`ModeAState.__init__` ~line 311, `put_artifact` ~line 503)
- Test: `flashruntime/tests/test_service_write_scope.py`

**Interfaces:**
- Consumes: `authenticator_from_env`, `NodeAuthenticator` (Task 1); `live_leases_for_node` (Task 2).
- Produces: `ModeAState.authenticator`; helper `_authorize_write(state, manager, request, key) -> None` raising `HTTPException(401|403)`.

Rules, exactly:
- authenticator **not** enforcing → allow (self-hosted, unchanged).
- enforcing, no/bad token → **401**.
- enforcing, valid token, key outside every live lease prefix → **403**.
- allowed prefix for lease `(job, task)` is exactly `jobs/{job}/{task}/`.

- [ ] **Step 1: Write the failing test**

```python
# flashruntime/tests/test_service_write_scope.py
"""A node may write only under the tasks it currently holds."""

import pytest
from fastapi.testclient import TestClient

from flashruntime.service.app import create_app, RuntimeSettings


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHML_NODE_TOKENS", "node-a:tok-a,node-b:tok-b")
    monkeypatch.setenv("FLASHML_LOCAL_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_SERVICE_AUTOINIT", "1")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "ledger.db"))
    return TestClient(create_app())


def _register(client, node_id):
    client.post("/v1alpha1/nodes/register", json={
        "schema_version": "v1alpha1", "node_id": node_id,
        "kubernetes_node": node_id, "hostname": node_id,
        "capabilities": {"cpu_cores": 1, "memory_bytes": 1 << 30,
                         "gpus": [], "os": "linux", "architecture": "x86_64"},
    })


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
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
                   content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 200


def test_another_node_may_not_write_under_that_task(client):
    """The core exploit: one volunteer overwriting another's committed result."""
    _register(client, "node-a")
    _register(client, "node-b")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-b"})
    assert r.status_code == 403


def test_the_holder_may_not_write_outside_its_task_prefix(client):
    """Guards the federated-averaging model key, which lives at
    jobs/{job}/round-NNN/weights.json — outside any task prefix."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"})
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/round-000/weights.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_a_sibling_prefix_does_not_satisfy_the_check(client):
    """`jobs/j/trial-000extra/` must not pass because it starts with
    `jobs/j/trial-000`. The prefix must end at a separator."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    r = client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}extra/metrics.json",
                   content=b"evil", headers={"Authorization": "Bearer tok-a"})
    assert r.status_code == 403


def test_reads_are_not_scoped(client):
    """Drivers read other tasks' outputs and agents download shared inputs."""
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    client.put(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json",
               content=b"{}", headers={"Authorization": "Bearer tok-a"})
    assert client.get(f"/v1alpha1/artifacts/jobs/{job_id}/{task_id}/metrics.json").status_code == 200
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_write_scope.py -v`
Expected: the 401/403 tests FAIL (every write returns 200 today).

- [ ] **Step 3: Write minimal implementation**

In `ModeAState.__init__` (`modea.py` ~311), accept and store an authenticator, defaulting to the env-derived one:

```python
        from flashruntime.service.auth import NodeAuthenticator, authenticator_from_env
        self.authenticator: NodeAuthenticator = authenticator or authenticator_from_env()
```

Add beside `_safe_key`:

```python
def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _authorize_write(state, manager, request: Request, key: str) -> None:
    """Confine a write to the tasks this caller currently holds.

    Not enforcing ⇒ allow: the self-hosted profile predates credentials and
    must keep working (CLAUDE.md rule 4). When enforcing, an unknown caller is
    401 and an out-of-scope key is 403 — distinct so an operator can tell a
    misconfigured agent from a misbehaving one.
    """
    if not state.authenticator.enforcing:
        return
    node_id = state.authenticator.authenticate(_bearer(request))
    if node_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing node token")
    # Trailing slash is load-bearing: without it `jobs/j/trial-000extra/...`
    # would satisfy a `jobs/j/trial-000` prefix test.
    allowed = [f"jobs/{job}/{task}/" for job, task in manager.live_leases_for_node(node_id)]
    if not any(key.startswith(p) for p in allowed):
        raise HTTPException(
            status_code=403,
            detail=f"node {node_id} holds no live lease covering {key!r}",
        )
```

Call it first in `put_artifact` (`modea.py:503`), after `_safe_key`:

```python
    @router.put("/artifacts/{key:path}")
    async def put_artifact(key: str, request: Request):
        key = _safe_key(key)
        _authorize_write(state, manager, request, key)
        data = await request.body()
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_write_scope.py tests/test_service_modea.py -v`
Expected: 8 new passed; `test_service_modea.py` unchanged and green (it configures no tokens, so the open path is untouched).

- [ ] **Step 5: Commit**

```bash
git add flashruntime/service/modea.py tests/test_service_write_scope.py
git commit -m "feat(service): scope artifact writes to the caller's live leases

Closes HANDOFF.md risk #2 for artifacts: PUT accepted any key from any
caller, and the sha256 check was no defense because the attacker supplied
both file and hash. 401 for an unknown caller, 403 for an out-of-scope key.
Open when no authenticator is configured, so the self-hosted profile is
unchanged. The prefix carries a trailing slash so a sibling task id cannot
satisfy it."
```

---

### Task 4: Scope checkpoint writes, and fail closed on request

**Files:**
- Modify: `flashruntime/flashruntime/service/checkpoints.py` (`register_part` :47, `commit` :52)
- Modify: `flashruntime/flashruntime/service/app.py` (startup guard)
- Test: extend `flashruntime/tests/test_service_write_scope.py`; `flashruntime/tests/test_service_auth_startup.py`

**Interfaces:**
- Consumes: `_authorize_write` (Task 3). Export it from `modea.py` so `checkpoints.py` can import it — the checkpoint router already receives what it needs at build time; thread `state` and `manager` through `build_router` if they are not already available.

Checkpoint routes carry `job_id` and `task_id` in the path, so authorize `(job_id, task_id)` membership directly rather than by key prefix.

`FLASHML_REQUIRE_NODE_AUTH=1` makes `create_app()` raise at startup when the authenticator is not enforcing. An operator who exposes the coordinator and forgets `FLASHML_NODE_TOKENS` gets a refusal to boot, not a silent open door.

- [ ] **Step 1: Write the failing tests**

```python
# append to flashruntime/tests/test_service_write_scope.py

def test_checkpoint_part_outside_a_live_lease_is_403(client):
    _register(client, "node-a")
    r = client.post(
        "/v1alpha1/jobs/other-job/tasks/trial-000/checkpoints/parts",
        json={"step": 10, "key": "jobs/other-job/trial-000/ckpt/step-10.json",
              "sha256": "0" * 64, "size_bytes": 2},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert r.status_code == 403


def test_checkpoint_part_without_a_token_is_401(client):
    r = client.post(
        "/v1alpha1/jobs/other-job/tasks/trial-000/checkpoints/parts",
        json={"step": 10, "key": "k", "sha256": "0" * 64, "size_bytes": 2},
    )
    assert r.status_code == 401


def test_the_lease_holder_may_register_a_checkpoint_part(client):
    _register(client, "node-a")
    job_id = _submit_one_task_job(client)
    lease = client.post("/v1alpha1/leases/claim", json={"node_id": "node-a"}).json()
    task_id = lease["payload"]["task_id"]
    key = f"jobs/{job_id}/{task_id}/ckpt/step-10.json"
    client.put(f"/v1alpha1/artifacts/{key}", content=b"{}",
               headers={"Authorization": "Bearer tok-a"})
    r = client.post(
        f"/v1alpha1/jobs/{job_id}/tasks/{task_id}/checkpoints/parts",
        json={"step": 10, "key": key, "sha256": "0" * 64, "size_bytes": 2},
        headers={"Authorization": "Bearer tok-a"},
    )
    assert r.status_code in (200, 201)
```

```python
# flashruntime/tests/test_service_auth_startup.py
"""An operator who exposes the coordinator and forgets to configure tokens
must get a refusal to boot, not a silently open door."""

import pytest

from flashruntime.service.app import create_app


def test_require_node_auth_without_tokens_refuses_to_start(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHML_REQUIRE_NODE_AUTH", "1")
    monkeypatch.delenv("FLASHML_NODE_TOKENS", raising=False)
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "l.db"))
    with pytest.raises(RuntimeError, match="FLASHML_NODE_TOKENS"):
        create_app()


def test_require_node_auth_with_tokens_starts(monkeypatch, tmp_path):
    monkeypatch.setenv("FLASHML_REQUIRE_NODE_AUTH", "1")
    monkeypatch.setenv("FLASHML_NODE_TOKENS", "node-a:tok-a")
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "l.db"))
    assert create_app() is not None


def test_default_startup_is_open_and_unchanged(monkeypatch, tmp_path):
    monkeypatch.delenv("FLASHML_REQUIRE_NODE_AUTH", raising=False)
    monkeypatch.delenv("FLASHML_NODE_TOKENS", raising=False)
    monkeypatch.setenv("FLASHML_ENABLE_KUBERAY", "0")
    monkeypatch.setenv("FLASHML_LEDGER_PATH", str(tmp_path / "l.db"))
    assert create_app() is not None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_write_scope.py tests/test_service_auth_startup.py -v`
Expected: the three checkpoint tests and `test_require_node_auth_without_tokens_refuses_to_start` FAIL.

- [ ] **Step 3: Write minimal implementation**

Add a lease-pair authorizer beside `_authorize_write` in `modea.py` and export both:

```python
def authorize_task_write(state, manager, request: Request, job_id: str, task_id: str) -> None:
    """Checkpoint routes name (job, task) in the path, so authorize the pair
    directly instead of reconstructing a key prefix."""
    if not state.authenticator.enforcing:
        return
    node_id = state.authenticator.authenticate(_bearer(request))
    if node_id is None:
        raise HTTPException(status_code=401, detail="invalid or missing node token")
    if (job_id, task_id) not in manager.live_leases_for_node(node_id):
        raise HTTPException(
            status_code=403,
            detail=f"node {node_id} holds no live lease on {job_id}/{task_id}",
        )
```

Call it at the top of `register_part` and `commit` in `checkpoints.py`. Those handlers need `state`, `manager`, and the `Request`; thread them in via `build_router(catalog, state=..., manager=...)` and add `request: Request` to each signature, following how `modea.build_router` already receives its state.

In `app.py`'s `create_app`, after the state is built:

```python
    if os.environ.get("FLASHML_REQUIRE_NODE_AUTH") == "1" and not state.authenticator.enforcing:
        raise RuntimeError(
            "FLASHML_REQUIRE_NODE_AUTH=1 but no node tokens are configured — "
            "set FLASHML_NODE_TOKENS. Refusing to start an internet-exposed "
            "coordinator with unauthenticated writes."
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_service_write_scope.py tests/test_service_auth_startup.py tests/test_service_checkpoints.py -v`
Expected: all new passed; `test_service_checkpoints.py` unchanged and green.

- [ ] **Step 5: Commit**

```bash
git add flashruntime/service/checkpoints.py flashruntime/service/modea.py \
        flashruntime/service/app.py tests/test_service_write_scope.py \
        tests/test_service_auth_startup.py
git commit -m "feat(service): scope checkpoint writes; fail closed on request

Checkpoint parts/commit now require a live lease on the (job, task) named in
the path. FLASHML_REQUIRE_NODE_AUTH=1 refuses to boot when no tokens are
configured, so exposing the coordinator and forgetting to set them is a
startup failure rather than a silent open door."
```

---

### Task 5: flashnode sends a credential

**Files:**
- Create: `flashnode/flashnode/identity/credentials.py`
- Modify: `flashnode/flashnode/executor/client.py` (`CoordinatorClient.__init__` :29)
- Modify: `flashnode/flashnode/agent/cli.py`
- Test: `flashnode/tests/test_credentials.py`

**Interfaces:**
- Produces:
  - `credentials_path() -> Path` — `$FLASHNODE_CREDENTIALS` or `~/.flashnode/credentials.json`
  - `save_token(coordinator: str, token: str) -> Path` — writes 0600
  - `load_token(coordinator: str) -> str | None`
  - `clear_token(coordinator: str) -> bool`
  - `flashnode login --coordinator URL --token TOKEN`, `flashnode logout --coordinator URL`
  - `CoordinatorClient(base_url, ..., token: str | None = None)` sending `Authorization: Bearer`

Tokens are keyed by coordinator URL so one machine can join more than one pool without them overwriting each other.

Note: this is the *manual-token* half. The interactive device flow (`flashnode login` with a browser code) needs a server to talk to and arrives with Plan 3; the credential store and header plumbing built here are what it will reuse.

- [ ] **Step 1: Write the failing test**

```python
# flashnode/tests/test_credentials.py
import json
import stat

import pytest

from flashnode.identity.credentials import (
    clear_token, credentials_path, load_token, save_token,
)


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("FLASHNODE_CREDENTIALS", str(tmp_path / "credentials.json"))


def test_no_token_before_login():
    assert load_token("https://c.example") is None


def test_save_then_load_round_trip():
    save_token("https://c.example", "tok-a")
    assert load_token("https://c.example") == "tok-a"


def test_tokens_are_scoped_per_coordinator():
    """One machine may join several pools; a second login must not clobber
    the first."""
    save_token("https://a.example", "tok-a")
    save_token("https://b.example", "tok-b")
    assert load_token("https://a.example") == "tok-a"
    assert load_token("https://b.example") == "tok-b"


def test_credentials_file_is_not_world_readable():
    save_token("https://c.example", "tok-a")
    mode = credentials_path().stat().st_mode
    assert not mode & stat.S_IRGRP
    assert not mode & stat.S_IROTH


def test_clear_removes_only_that_coordinator():
    save_token("https://a.example", "tok-a")
    save_token("https://b.example", "tok-b")
    assert clear_token("https://a.example") is True
    assert load_token("https://a.example") is None
    assert load_token("https://b.example") == "tok-b"


def test_clear_is_idempotent():
    assert clear_token("https://nothing.example") is False


def test_a_corrupt_credentials_file_does_not_crash_the_agent():
    credentials_path().parent.mkdir(parents=True, exist_ok=True)
    credentials_path().write_text("{not json")
    assert load_token("https://c.example") is None


def test_trailing_slashes_do_not_split_the_identity():
    save_token("https://c.example/", "tok-a")
    assert load_token("https://c.example") == "tok-a"


def test_client_sends_bearer_header_when_a_token_is_present():
    from flashnode.executor.client import CoordinatorClient

    c = CoordinatorClient("http://c.example", token="tok-a")
    assert c._headers().get("Authorization") == "Bearer tok-a"


def test_client_sends_no_auth_header_without_a_token():
    from flashnode.executor.client import CoordinatorClient

    c = CoordinatorClient("http://c.example")
    assert "Authorization" not in c._headers()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd flashnode && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest tests/test_credentials.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashnode.identity.credentials'`

- [ ] **Step 3: Write the implementation**

Create `flashnode/flashnode/identity/credentials.py` implementing the five functions above. Requirements the tests pin: JSON object keyed by normalized coordinator URL (strip trailing `/`); create the parent directory; `os.chmod(path, 0o600)` after writing; return `None` rather than raising on unparseable JSON — an agent must not die because a credentials file got truncated.

Then in `CoordinatorClient` (`flashnode/flashnode/executor/client.py:29`): accept `token: str | None = None`, add a `_headers()` method returning `{"Authorization": f"Bearer {self._token}"}` when set and `{}` otherwise, and merge it into every request the class makes. Read the class first — every existing call site must keep working unchanged.

In `flashnode/flashnode/agent/cli.py`: add `login` and `logout` subcommands alongside the existing `work`, and make `work` load a token for its `--coordinator` URL and pass it to `CoordinatorClient`. Print where the credential was written; **never print the token itself**.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd flashnode && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q`
Expected: 10 new passed on top of the 75-passed baseline; nothing regressed.

- [ ] **Step 5: Commit**

```bash
git add flashnode/identity/credentials.py flashnode/executor/client.py \
        flashnode/agent/cli.py tests/test_credentials.py
git commit -m "feat(identity): per-coordinator credential store and bearer auth

Tokens are keyed by coordinator URL so one machine can join several pools
without clobbering. File is 0600; a corrupt file yields no token rather than
killing the agent. This is the manual-token half — the interactive device
flow needs a server and arrives with the cloud API, reusing this store."
```

---

### Task 6: Prove the exploit is closed end to end

**Files:**
- Create: `../e2e/test_write_scope.py` (workspace repo)

**Interfaces:**
- Consumes: everything above. Uses the `coordinator` fixture in `e2e/conftest.py`.

Unit tests use `TestClient`. This proves it against a real coordinator process with real agents, which is the configuration that will actually be deployed.

- [ ] **Step 1: Write the failing test**

The `coordinator` fixture must start with `FLASHML_NODE_TOKENS` set — read `e2e/conftest.py` and add a parametrized or sibling fixture rather than changing the existing one, so the current e2e tests keep their open profile.

Assert, over real HTTP:
1. A write with no token → 401.
2. `node-b` writing under a task `node-a` holds → **403**. This is the exploit: one volunteer overwriting another's committed result.
3. `node-a` writing under its own held task → 200.
4. After `node-a`'s lease expires, the same write → 403.
5. A write to `jobs/{job}/round-000/weights.json` (the federated-averaging model key, outside any task prefix) → 403 for every node.

- [ ] **Step 2: Run and iterate**

Run: `cd /Users/phongcao/Work/Zolli-Labs && PATH="$PWD/e2e/.venv/bin:$PATH" e2e/.venv/bin/pytest e2e/test_write_scope.py -v`

- [ ] **Step 3: Confirm nothing regressed anywhere**

```
cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q --ignore=tests/test_examples_e2e.py --ignore=tests/test_gpu_e2e.py
cd ../flashnode   && PATH="$PWD/.venv/bin:$PATH" .venv/bin/pytest -q
cd ..             && PATH="$PWD/e2e/.venv/bin:$PATH" e2e/.venv/bin/pytest e2e/ -q
cd flashruntime && PATH="$PWD/.venv/bin:$PATH" .venv/bin/python -u scripts/fedavg_local_demo.py
```

Expected: 417+ / 75+ / 7+ and the demo still converging `0.5361 → 0.1757` at exit 0. **The demo runs the open profile** — if it broke, the default path is no longer backward compatible, which is a Task 3 defect, not a demo problem.

- [ ] **Step 4: Commit**

```bash
cd .. && git add e2e/test_write_scope.py
git commit -m "test(e2e): one volunteer cannot overwrite another's results"
```

---

### Task 7: Documentation and the progress entry

**Files:**
- Modify: `flashruntime/docs/guides/donate-a-machine.md` — it currently documents "one shared join code" as a known gap; update to per-machine tokens and revocation.
- Modify: `flashruntime/docs/site/guides/federated-averaging.md` — the round-weights key is now write-protected.
- Modify: `../PROGRESS.md` — entry per the logging protocol; tick Plan 2 in the M1 checklist.
- Modify: `../M1_DECISIONS.md` — close the D11 note that deferred these two items to Plan 2.

State honestly what is still **not** true: the tokens here are configured statically on the coordinator; self-service enrolment arrives with the cloud API. Result verification (a node lying about its own result) remains M3 and is untouched by this plan.

- [ ] **Step 1: Write the docs and the entry, then commit**

```bash
cd .. && git add PROGRESS.md M1_DECISIONS.md
git -C flashruntime add docs/
git commit -m "docs(progress): agent identity + lease-scoped writes (M1 Plan 2 of 7)"
```

---

## Self-Review

**Spec coverage.** §5.2 (agent credential, bearer token) → Tasks 1, 5. §6.1 (lease-scoped artifact and checkpoint writes) → Tasks 3, 4, 6. §6.2 (shared join code replaced) → Tasks 1, 5, 7. The §5.2 *device flow* is deliberately **not** here: it needs a server, so it belongs with the cloud API in Plan 3 — Task 5 builds the credential store and header plumbing it will reuse. `resolve node_id from the token, never the body` (spec §5.2) is satisfied because `_authorize_write` derives `node_id` from the token alone and never reads the request body.

**Not covered, deliberately:** read-side authorization (Plan 3, by job ownership); result verification (M3); the abandoned-shard zombie tasks from Plan 1's review (needs cooperative cancel, M3).

**Type consistency.** `authenticate(token: str | None) -> str | None` is used identically in Tasks 1, 3, 4. `live_leases_for_node(node_id) -> set[tuple[str, str]]` is defined in Task 2 and consumed as a set of `(job_id, task_id)` pairs in Tasks 3 and 4. `enforcing` is a property on every authenticator and is read in Tasks 3, 4. `_bearer(request)` is defined once in Task 3 and reused in Task 4.

**Known risk to watch during execution.** Task 4 threads `state` and `manager` into `checkpoints.build_router`, changing its signature. Check every call site — `service/app.py` at minimum — and note that `tests/test_service_checkpoints.py` may construct the router directly.
