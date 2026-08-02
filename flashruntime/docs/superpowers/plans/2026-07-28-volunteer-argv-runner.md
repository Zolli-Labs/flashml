# Volunteer argv runner + composite lease key — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let any machine join as an untrusted volunteer compute resource, pull tasks from the coordinator, and execute a submitting user's arbitrary code inside a hardened container — with the isolation tier, not a module allowlist, as the security control.

**Architecture:** Additive. A new `ArgvDockerRunner` in flashnode consumes the `argv` payload `CommandRecipe` already emits; a shared `harden_args()` helper becomes the single source of truth for container security flags. The proven module runners are left intact. In flashruntime, lease records become keyed by `(job_id, task_id)` so concurrent jobs stop colliding on positional task IDs.

**Tech Stack:** Python ≥3.10, pydantic (core), pytest, sqlite3 (stdlib), Docker CLI (agent-side only).

## Global Constraints

- **Repos:** `flashruntime` (protocol owner) and `flashnode`. `flashruntime` imports NOTHING from `flashnode`. Ever.
- **Core stays pydantic-only:** `import flashruntime` + planner/leases/checkpoint/recovery must never require numpy/k8s/minio/fastapi.
- **Security-relevant fields fail closed** (AGENTS.md rule 3). Capability checks accept `is True` only — never truthy stand-ins like `1` or `"true"`.
- **Test baseline before starting:** flashruntime `297 passed, 1 skipped, 20 deselected`. Every task must leave the suite green.
- **`scripts/audit_secrets.sh` must stay CLEAN.**
- **No shell=True, ever.** Argv lists end to end.
- **Spec:** `docs/superpowers/specs/2026-07-28-volunteer-argv-runner-design.md`. Change a contract only with a red test + a note.
- **Venv:** `source .venv/bin/activate` from the repo root (never call `.venv/bin/python` by path — `torchrun` resolves via `PATH`).

## File Structure

| Repo | File | Responsibility |
|---|---|---|
| flashruntime | `flashruntime/leases/store.py` | Protocol + in-memory store — composite key |
| flashruntime | `flashruntime/leases/sqlite_store.py` | Durable store — composite PK + migration |
| flashruntime | `flashruntime/leases/manager.py` | `_require`/`cancel_task`/`claim` job-scoping |
| flashruntime | `flashruntime/protocol/v1alpha1.py` | `NodeRegistration.argv_capable` |
| flashruntime | `flashruntime/scheduler/__init__.py` | argv placement gate |
| flashruntime | `flashruntime/recipes/command.py` | tier validation |
| flashruntime | `flashruntime/service/modea.py` | NodeView wiring |
| flashnode | `flashnode/executor/hardening.py` | **new** — `harden_args()` |
| flashnode | `flashnode/executor/argv_runner.py` | **new** — `ArgvDockerRunner` |
| flashnode | `flashnode/executor/docker_runner.py` | refactor onto `harden_args()` |
| flashnode | `flashnode/executor/runner.py` | argv refusal |
| flashnode | `flashnode/agent/cli.py` | `--runner argv` |
| flashnode | `flashnode/inventory/capabilities.py` | advertise `argv_capable` |

---

### Task 1: Composite `(job_id, task_id)` key — in-memory store + manager

**Files:**
- Modify: `flashruntime/flashruntime/leases/store.py:45-96`
- Modify: `flashruntime/flashruntime/leases/manager.py:76-82,105-110,280-284`
- Modify: `flashruntime/flashruntime/service/app.py:371`
- Test: `flashruntime/tests/test_leases.py`

**Interfaces:**
- Consumes: nothing (first task).
- Produces: `LeaseStore.get(job_id: str, task_id: str) -> TaskRecord | None`; `InMemoryLeaseStore._tasks: dict[tuple[str, str], TaskRecord]`; `LeaseManager.cancel_task(job_id: str, task_id: str, now: datetime | None = None) -> None`. Task 2 implements the same `get` signature for SQLite.

There are **two** collision bugs here, not one. The known bug is the store key. The second is in `claim()`: the policy path re-finds the chosen record by `task_id` alone (`manager.py:107`), so with two jobs it can select a *different job's* record than the policy picked.

- [ ] **Step 1: Write the failing tests**

Add to `flashruntime/tests/test_leases.py`:

```python
from flashruntime.leases.manager import LeaseManager
from flashruntime.leases.store import InMemoryLeaseStore, TaskRecord
from flashruntime.protocol.v1alpha1 import TaskSpec, TaskState


def _spec(job_id: str, task_id: str) -> TaskSpec:
    return TaskSpec(task_id=task_id, job_id=job_id, commit_key=f"{job_id}/{task_id}/m.json")


def test_two_jobs_may_share_a_task_id():
    store = InMemoryLeaseStore()
    store.add(TaskRecord(_spec("job-a", "task-000")))
    store.add(TaskRecord(_spec("job-b", "task-000")))  # must NOT raise
    assert store.get("job-a", "task-000").spec.job_id == "job-a"
    assert store.get("job-b", "task-000").spec.job_id == "job-b"


def test_duplicate_within_one_job_still_rejected():
    store = InMemoryLeaseStore()
    store.add(TaskRecord(_spec("job-a", "task-000")))
    with pytest.raises(ValueError):
        store.add(TaskRecord(_spec("job-a", "task-000")))


def test_get_is_job_scoped():
    store = InMemoryLeaseStore()
    store.add(TaskRecord(_spec("job-a", "task-000")))
    assert store.get("job-b", "task-000") is None


def test_claim_with_policy_picks_the_right_job():
    """The policy path re-finds its chosen spec in `pending`. Matching on
    task_id alone crosses job boundaries when ids collide."""
    class PickJobB:
        def choose(self, specs, node):
            return next(s for s in specs if s.job_id == "job-b")

    mgr = LeaseManager()
    mgr.add_task(_spec("job-a", "task-000"))
    mgr.add_task(_spec("job-b", "task-000"))
    lease = mgr.claim("node-1", policy=PickJobB(), node={"node_id": "node-1"})
    assert lease.job_id == "job-b"


def test_cancel_task_is_job_scoped():
    mgr = LeaseManager()
    mgr.add_task(_spec("job-a", "task-000"))
    mgr.add_task(_spec("job-b", "task-000"))
    mgr.cancel_task("job-a", "task-000")
    assert mgr.tasks("job-a")[0].state == TaskState.CANCELLED
    assert mgr.tasks("job-b")[0].state == TaskState.PENDING
```

Update the two existing calls at `tests/test_leases.py:151-152` from `mgr.cancel_task("t1")` to `mgr.cancel_task(<that task's job_id>, "t1")`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `source .venv/bin/activate && pytest tests/test_leases.py -v -k "two_jobs or job_scoped or right_job or duplicate_within"`
Expected: FAIL — `ValueError: task task-000 already exists`, and `get()` `TypeError: takes 2 positional arguments but 3 were given`.

- [ ] **Step 3: Change the store to a composite key**

In `flashruntime/leases/store.py`, update the Protocol signature and the in-memory implementation:

```python
class LeaseStore(Protocol):
    def add(self, record: TaskRecord) -> None: ...
    def save(self, record: TaskRecord) -> None: ...
    def get(self, job_id: str, task_id: str) -> TaskRecord | None: ...
```

```python
class InMemoryLeaseStore:
    """Reference implementation: insertion-ordered dict, no persistence.

    Keyed by (job_id, task_id): task ids are positional within a job
    (`task-000`), so two jobs routinely produce the same task_id.
    """

    def __init__(self) -> None:
        self._tasks: dict[tuple[str, str], TaskRecord] = {}

    def add(self, record: TaskRecord) -> None:
        key = (record.spec.job_id, record.spec.task_id)
        if key in self._tasks:
            raise ValueError(
                f"task {record.spec.task_id} already exists in job {record.spec.job_id}"
            )
        self._tasks[key] = record

    def get(self, job_id: str, task_id: str) -> TaskRecord | None:
        return self._tasks.get((job_id, task_id))
```

`save`, `next_pending`, `leased`, and `all` iterate `.values()` and need **no change** — insertion order is preserved, so `next_pending()` stays deterministic as its docstring promises.

- [ ] **Step 4: Job-scope the manager**

In `flashruntime/leases/manager.py`:

```python
    def cancel_task(self, job_id: str, task_id: str, now: datetime | None = None) -> None:
        record = self._require(job_id, task_id)
```

```python
    def _require(self, job_id: str, task_id: str) -> TaskRecord:
        record = self._store.get(job_id, task_id)
        if record is None:
            raise LeaseError(f"unknown task {task_id} in job {job_id}")
        return record
```

And fix the second collision bug in `claim()` (around line 107) — match on **both** fields:

```python
                record = next(
                    (
                        r
                        for r in pending
                        if r.spec.task_id == chosen.task_id
                        and r.spec.job_id == chosen.job_id
                    ),
                    None,
                )
```

- [ ] **Step 5: Update the service caller**

In `flashruntime/service/app.py:371`, the loop already holds `record`:

```python
                lease_manager.cancel_task(record.spec.job_id, record.spec.task_id)
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_leases.py -v`
Expected: PASS, including the pre-existing tests.

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: `298+ passed, 1 skipped, 20 deselected`. `tests/test_leases_sqlite.py` will FAIL here — that is expected; Task 2 fixes it. If anything *else* fails, stop and fix before continuing.

- [ ] **Step 8: Commit**

```bash
git add flashruntime/leases/store.py flashruntime/leases/manager.py \
        flashruntime/service/app.py tests/test_leases.py
git commit -m "fix(leases): key tasks by (job_id, task_id), not task_id alone

Task ids are positional within a job (task-000), so two concurrent jobs
collided with 'task already exists'. Also fixes the policy claim path,
which re-found its chosen spec by task_id alone and could cross job
boundaries."
```

---

### Task 2: SQLite composite primary key + migration

**Files:**
- Modify: `flashruntime/flashruntime/leases/sqlite_store.py` (whole file)
- Test: `flashruntime/tests/test_leases_sqlite.py`

**Interfaces:**
- Consumes: `LeaseStore.get(job_id, task_id)` from Task 1.
- Produces: `SqliteLeaseStore` with `PRIMARY KEY (job_id, task_id)` and a `_migrate()` that upgrades an old single-PK database in place.

The migration is the load-bearing part: `SqliteLeaseStore` exists so **in-flight leases survive a coordinator restart**. Dropping and recreating the table would silently destroy exactly the property the class is for.

- [ ] **Step 1: Write the failing tests**

Add to `flashruntime/tests/test_leases_sqlite.py`:

```python
import json
import sqlite3

from flashruntime.leases.sqlite_store import SqliteLeaseStore
from flashruntime.leases.store import TaskRecord
from flashruntime.protocol.v1alpha1 import TaskSpec, TaskState

_OLD_SCHEMA = """
CREATE TABLE lease_tasks (
    task_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts_used INTEGER NOT NULL,
    active_lease_json TEXT,
    accepted_attempt_id TEXT,
    lease_history_json TEXT NOT NULL,
    seq INTEGER
);
"""


def _spec(job_id, task_id):
    return TaskSpec(task_id=task_id, job_id=job_id, commit_key=f"{job_id}/{task_id}/m.json")


def test_two_jobs_share_a_task_id_and_survive_reopen(tmp_path):
    path = tmp_path / "leases.db"
    store = SqliteLeaseStore(path)
    store.add(TaskRecord(_spec("job-a", "task-000")))
    store.add(TaskRecord(_spec("job-b", "task-000")))

    reopened = SqliteLeaseStore(path)
    assert reopened.get("job-a", "task-000").spec.job_id == "job-a"
    assert reopened.get("job-b", "task-000").spec.job_id == "job-b"
    assert len(reopened.all()) == 2


def test_migration_preserves_an_in_flight_lease(tmp_path):
    """The whole point of the durable store: a lease issued before the
    upgrade must still be renewable after it."""
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(_OLD_SCHEMA)
    spec = _spec("job-a", "task-000")
    lease = {
        "schema_version": "v1alpha1", "lease_id": "lease-1", "task_id": "task-000",
        "job_id": "job-a", "node_id": "node-1", "attempt_id": "attempt-1",
        "attempt_number": 1, "deadline": "2099-01-01T00:00:00Z",
    }
    conn.execute(
        "INSERT INTO lease_tasks VALUES (?,?,?,?,?,?,?,?,?)",
        ("task-000", "job-a", spec.model_dump_json(), "leased", 1,
         json.dumps(lease), None, json.dumps({"lease-1": lease}), 1),
    )
    conn.commit()
    conn.close()

    store = SqliteLeaseStore(path)                      # migrates on open
    record = store.get("job-a", "task-000")
    assert record is not None
    assert record.state == TaskState.LEASED
    assert record.active_lease.lease_id == "lease-1"    # in-flight lease survived
    assert "lease-1" in record.lease_history

    cols = sqlite3.connect(path).execute("PRAGMA table_info(lease_tasks)").fetchall()
    pk_cols = sorted(c[1] for c in cols if c[5] > 0)
    assert pk_cols == ["job_id", "task_id"]             # composite PK now

    store.add(TaskRecord(_spec("job-b", "task-000")))   # collision no longer possible
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_leases_sqlite.py -v`
Expected: FAIL — `sqlite3.IntegrityError: UNIQUE constraint failed` and `TypeError` on `get()`.

- [ ] **Step 3: Update the schema and add the migration**

In `flashruntime/leases/sqlite_store.py`, replace `_SCHEMA` and add `_migrate()`:

```python
_SCHEMA = """
CREATE TABLE IF NOT EXISTS lease_tasks (
    task_id TEXT NOT NULL,
    job_id TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    state TEXT NOT NULL,
    attempts_used INTEGER NOT NULL,
    active_lease_json TEXT,
    accepted_attempt_id TEXT,
    lease_history_json TEXT NOT NULL,
    seq INTEGER,
    PRIMARY KEY (job_id, task_id)
);
CREATE INDEX IF NOT EXISTS idx_lease_tasks_job ON lease_tasks (job_id);
"""

_COLUMNS = (
    "task_id, job_id, spec_json, state, attempts_used, active_lease_json,"
    " accepted_attempt_id, lease_history_json, seq"
)
```

```python
    def _migrate(self) -> None:
        """Upgrade a pre-composite-key database in place.

        SQLite cannot alter a primary key, so the table is rebuilt and the
        rows copied — never dropped. In-flight leases must survive the
        upgrade; that durability is the reason this store exists.
        """
        info = self._conn.execute("PRAGMA table_info(lease_tasks)").fetchall()
        if not info:
            return  # fresh database; _SCHEMA already created the right table
        pk_cols = sorted(col[1] for col in info if col[5] > 0)
        if pk_cols == ["job_id", "task_id"]:
            return  # already migrated
        self._conn.executescript(
            "BEGIN;"
            "ALTER TABLE lease_tasks RENAME TO lease_tasks_legacy;"
            + _SCHEMA.replace("IF NOT EXISTS lease_tasks", "lease_tasks")
            + f"INSERT INTO lease_tasks ({_COLUMNS})"
            f" SELECT {_COLUMNS} FROM lease_tasks_legacy;"
            "DROP TABLE lease_tasks_legacy;"
            "COMMIT;"
        )
        self._conn.commit()
```

Call it from `__init__`, **before** `_load()`:

```python
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._migrate()                 # legacy single-PK table → composite
        self._conn.executescript(_SCHEMA)
        self._cache: dict[tuple[str, str], TaskRecord] = {}
        self._seq = 0
        self._load()
```

Note the ordering: `_migrate()` runs first so it sees the legacy table; `_SCHEMA` is then a no-op `IF NOT EXISTS` for the migrated case and the creator for a fresh database.

- [ ] **Step 4: Update cache keys, `get`, `add`, and the upsert**

```python
    def _load(self) -> None:
        ...
            self._cache[(record.spec.job_id, record.spec.task_id)] = record
            self._seq = max(self._seq, seq or 0)

    def add(self, record: TaskRecord) -> None:
        key = (record.spec.job_id, record.spec.task_id)
        if key in self._cache:
            raise ValueError(
                f"task {record.spec.task_id} already exists in job {record.spec.job_id}"
            )
        self._cache[key] = record
        self._seq += 1
        self._persist(record, self._seq)

    def get(self, job_id: str, task_id: str) -> TaskRecord | None:
        return self._cache.get((job_id, task_id))
```

And in `_persist`, both the `seq` sub-select and the conflict target become job-scoped:

```python
        self._conn.execute(
            "INSERT INTO lease_tasks"
            " (task_id, job_id, spec_json, state, attempts_used, active_lease_json,"
            "  accepted_attempt_id, lease_history_json, seq)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?,"
            "   COALESCE(?, (SELECT seq FROM lease_tasks"
            "                 WHERE job_id = ? AND task_id = ?)))"
            " ON CONFLICT(job_id, task_id) DO UPDATE SET"
            "  state=excluded.state, attempts_used=excluded.attempts_used,"
            "  active_lease_json=excluded.active_lease_json,"
            "  accepted_attempt_id=excluded.accepted_attempt_id,"
            "  lease_history_json=excluded.lease_history_json",
            (
                record.spec.task_id,
                record.spec.job_id,
                record.spec.model_dump_json(),
                record.state.value,
                record.attempts_used,
                record.active_lease.model_dump_json() if record.active_lease else None,
                record.accepted_attempt_id,
                history,
                seq,
                record.spec.job_id,
                record.spec.task_id,
            ),
        )
```

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_leases_sqlite.py -v`
Expected: PASS.

- [ ] **Step 6: Run the full suite**

Run: `pytest -q`
Expected: green — `300+ passed, 1 skipped, 20 deselected`.

- [ ] **Step 7: Commit**

```bash
git add flashruntime/leases/sqlite_store.py tests/test_leases_sqlite.py
git commit -m "fix(leases): composite (job_id, task_id) primary key + in-place migration

Rebuilds a legacy single-PK table rather than dropping it, so leases
in flight across the upgrade stay renewable."
```

---

### Task 3: `harden_args()` + refactor `docker_runner` onto it

**Repo: flashnode.**

**Files:**
- Create: `flashnode/flashnode/executor/hardening.py`
- Modify: `flashnode/flashnode/executor/docker_runner.py:70-84`
- Test: `flashnode/tests/test_hardening.py` (new), `flashnode/tests/test_docker_runner.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `harden_args(workdir: Path, *, cpus: float, memory_gb: float, pids_limit: int = 512) -> list[str]` and `CONTAINER_WORKDIR = "/work"`. Task 4 calls it.

- [ ] **Step 1: Write the failing test**

Create `flashnode/tests/test_hardening.py`:

```python
from pathlib import Path

from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args


def test_harden_args_carries_the_full_security_contract(tmp_path):
    args = harden_args(tmp_path, cpus=2.0, memory_gb=4.0)
    joined = " ".join(args)
    assert "--network none" in joined
    assert "--read-only" in joined
    assert "--cap-drop=ALL" in joined
    assert "--security-opt=no-new-privileges" in joined
    assert "--pids-limit=512" in joined
    assert "noexec" in joined and "nosuid" in joined       # tmpfs flags
    assert f"{tmp_path}:{CONTAINER_WORKDIR}" in joined


def test_memory_swap_equals_memory():
    """Without this, --memory is bypassable via swap — the cap is a
    suggestion rather than a limit."""
    args = harden_args(Path("/tmp/x"), cpus=1.0, memory_gb=4.0)
    assert args[args.index("--memory") + 1] == "4.0g"
    assert args[args.index("--memory-swap") + 1] == "4.0g"


def test_runs_as_the_invoking_user_not_root():
    import os
    args = harden_args(Path("/tmp/x"), cpus=1.0, memory_gb=1.0)
    assert args[args.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd ../flashnode && source .venv/bin/activate && pytest tests/test_hardening.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashnode.executor.hardening'`.

- [ ] **Step 3: Write the implementation**

Create `flashnode/flashnode/executor/hardening.py`:

```python
"""The container security contract, in one place.

Every sandboxed runner builds its `docker run` flags here. Keeping this in a
single function is deliberate: two runners maintaining their own flag lists
drift, and the drift is invisible — the runner that quietly lost
`--cap-drop=ALL` still passes all its behavioural tests.

Changing this function changes the guarantee for ALL runners.
"""

from __future__ import annotations

import os
from pathlib import Path

CONTAINER_WORKDIR = "/work"


def harden_args(
    workdir: Path,
    *,
    cpus: float,
    memory_gb: float,
    pids_limit: int = 512,
) -> list[str]:
    """Docker flags common to every sandboxed task."""
    return [
        # the job never reaches the volunteer's LAN or the internet; the
        # agent is the courier for inputs, outputs, and checkpoints
        "--network", "none",
        "--read-only",
        "--tmpfs", "/tmp:rw,noexec,nosuid,size=256m",
        "--user", f"{os.getuid()}:{os.getgid()}",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        f"--pids-limit={pids_limit}",
        "--cpus", str(cpus),
        # equal values: with a larger memory-swap the memory cap is
        # bypassable by swapping
        "--memory", f"{memory_gb}g",
        "--memory-swap", f"{memory_gb}g",
        "--ulimit", "nofile=1024:1024",
        "-v", f"{workdir}:{CONTAINER_WORKDIR}",
        "-w", CONTAINER_WORKDIR,
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_hardening.py -v`
Expected: PASS.

- [ ] **Step 5: Refactor `docker_runner` onto the helper**

In `flashnode/flashnode/executor/docker_runner.py`, replace the inline flags with the helper. Import `CONTAINER_WORKDIR` from `hardening` and delete the local definition:

```python
from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args
```

```python
        argv = [
            "docker", "run", "--rm",
            *harden_args(workdir, cpus=self.cpus, memory_gb=self.memory_gb),
            image,
            "python", "-m", module,
            "--spec", f"{CONTAINER_WORKDIR}/spec.json",
            "--out", f"{CONTAINER_WORKDIR}/out",
        ]
```

This is additive for the module runner — it gains `--cap-drop=ALL`, `--security-opt`, `--pids-limit`, `--memory-swap`, `--ulimit`, and the bounded `noexec` tmpfs it did not have before.

- [ ] **Step 6: Run the docker runner tests**

Run: `pytest tests/test_docker_runner.py tests/test_hardening.py -v`
Expected: PASS. If a test pins the exact argv list, update it to assert *membership* of the required flags rather than exact ordering — an exact-list assertion will fight every future hardening change.

- [ ] **Step 7: Commit**

```bash
git add flashnode/executor/hardening.py flashnode/executor/docker_runner.py \
        tests/test_hardening.py tests/test_docker_runner.py
git commit -m "feat(executor): harden_args() as the single container security contract

Adds cap-drop=ALL, no-new-privileges, pids-limit, memory-swap parity,
nofile ulimit, and a bounded noexec tmpfs; docker_runner now builds its
flags from the shared helper so tiers cannot drift apart."
```

---

### Task 4: `ArgvDockerRunner`

**Repo: flashnode.**

**Files:**
- Create: `flashnode/flashnode/executor/argv_runner.py`
- Test: `flashnode/tests/test_argv_runner.py` (new)

**Interfaces:**
- Consumes: `harden_args`, `CONTAINER_WORKDIR` (Task 3); `TaskExecutionError` from `flashnode.executor.runner`.
- Produces: `ArgvDockerRunner(allowed_images: frozenset[str], cpus: float = 2.0, memory_gb: float = 2.0, timeout_seconds: float = 3600.0, max_output_bytes: int = 2 * 1024**3)` with `run(payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path`. Task 8 constructs it.

- [ ] **Step 1: Write the failing tests**

Create `flashnode/tests/test_argv_runner.py`:

```python
from pathlib import Path
from unittest import mock

import pytest

from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.runner import TaskExecutionError

IMAGES = frozenset({"ghcr.io/zolli/trainer:1.0"})


def _runner(**kw):
    return ArgvDockerRunner(allowed_images=IMAGES, **kw)


def _payload(**over):
    base = {"argv": ["python", "train.py"], "image": "ghcr.io/zolli/trainer:1.0",
            "env": {"LR": "0.05"}, "task_id": "task-000"}
    base.update(over)
    return base


_MISSING = object()   # distinct from every legitimate-but-invalid argv value


@pytest.mark.parametrize("bad", [_MISSING, None, [], "python train.py", [1, 2]])
def test_bad_argv_refused_before_any_subprocess(tmp_path, bad):
    payload = _payload()
    if bad is _MISSING:
        payload.pop("argv")          # payload carrying no argv key at all
    else:
        payload["argv"] = bad        # present but malformed
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError):
            _runner().run(payload, tmp_path, {})
    run.assert_not_called()      # a check that runs after launching is not a check


def test_non_allowlisted_image_refused_before_any_subprocess(tmp_path):
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError, match="not allowlisted"):
            _runner().run(_payload(image="evil/image:1"), tmp_path, {})
    run.assert_not_called()


def test_image_cannot_smuggle_a_docker_flag(tmp_path):
    """A hostile image value must never reach docker's flag parser."""
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError):
            _runner().run(_payload(image="--privileged"), tmp_path, {})
    run.assert_not_called()


def test_bad_env_key_refused(tmp_path):
    with mock.patch("subprocess.run") as run:
        with pytest.raises(TaskExecutionError, match="env"):
            _runner().run(_payload(env={"BAD KEY": "v"}), tmp_path, {})
    run.assert_not_called()


def test_argv_lands_after_the_image_so_flags_are_inert(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics.json").write_text("{}")
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        _runner().run(_payload(argv=["--privileged"]), tmp_path, {})
    cmd = run.call_args[0][0]
    assert cmd.index("--privileged") > cmd.index("ghcr.io/zolli/trainer:1.0")


def test_missing_metrics_json_fails_the_task(tmp_path):
    (tmp_path / "out").mkdir()
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        with pytest.raises(TaskExecutionError, match="metrics.json"):
            _runner().run(_payload(), tmp_path, {})


def test_nonzero_exit_reports_stderr_tail(tmp_path):
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=1, stderr=b"boom")
        with pytest.raises(TaskExecutionError, match="boom"):
            _runner().run(_payload(), tmp_path, {})


def test_output_size_cap_enforced(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "metrics.json").write_text("{}")
    (tmp_path / "out" / "big.bin").write_bytes(b"x" * 2048)
    with mock.patch("subprocess.run") as run:
        run.return_value = mock.Mock(returncode=0, stderr=b"")
        with pytest.raises(TaskExecutionError, match="output"):
            _runner(max_output_bytes=1024).run(_payload(), tmp_path, {})
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_argv_runner.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'flashnode.executor.argv_runner'`.

- [ ] **Step 3: Write the implementation**

Create `flashnode/flashnode/executor/argv_runner.py`:

```python
"""Tier-2 argv runner: run the user's own command inside a hardened container.

The module runners execute an allowlisted `python -m <module>`. This runner
executes whatever argv the job carried, which is what makes an arbitrary
machine a useful compute resource — and is why it is container-only. There
is no unsandboxed argv path here, by default or otherwise.

The security control is the isolation tier plus the operator's IMAGE
allowlist (what this volunteer consents to run), not a code allowlist: the
user's code inside a permitted image is unrestricted.

The task sees exactly one directory: its workdir bound at /work. Inputs are
pre-staged by the agent at /work/inputs; outputs are written to /work/out.
With --network none the job cannot fetch anything itself.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

from flashnode.executor.hardening import CONTAINER_WORKDIR, harden_args
from flashnode.executor.runner import TaskExecutionError

_ENV_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class ArgvDockerRunner:
    def __init__(
        self,
        allowed_images: frozenset[str],
        cpus: float = 2.0,
        memory_gb: float = 2.0,
        timeout_seconds: float = 3600.0,
        max_output_bytes: int = 2 * 1024**3,
    ):
        self.allowed_images = allowed_images
        self.cpus = cpus
        self.memory_gb = memory_gb
        self.timeout_seconds = timeout_seconds
        self.max_output_bytes = max_output_bytes

    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path:
        argv = payload.get("argv")
        if not argv or not isinstance(argv, list) or not all(isinstance(t, str) for t in argv):
            raise TaskExecutionError("payload 'argv' must be a non-empty list of strings")

        # Checked BEFORE any subprocess call, so a hostile value such as
        # "--privileged" can never reach docker's flag parser.
        image = payload.get("image")
        if not image or image not in self.allowed_images:
            raise TaskExecutionError(f"image {image!r} is not allowlisted — refusing to run")

        env_args: list[str] = []
        for key, value in (payload.get("env") or {}).items():
            if not _ENV_KEY.match(str(key)):
                raise TaskExecutionError(f"illegal env key {key!r} — refusing to run")
            env_args += ["--env", f"{key}={value}"]

        workdir = Path(workdir)
        outdir = workdir / "out"
        outdir.mkdir(parents=True, exist_ok=True)

        command = [
            "docker", "run", "--rm",
            *harden_args(workdir, cpus=self.cpus, memory_gb=self.memory_gb),
            *env_args,
            image,          # argv follows the image, where docker treats it
            *argv,          # as the container command: leading '-' is inert
        ]
        try:
            proc = subprocess.run(
                command, capture_output=True, timeout=self.timeout_seconds, check=False
            )
        except subprocess.TimeoutExpired:
            raise TaskExecutionError(f"task exceeded {self.timeout_seconds}s wall clock")
        if proc.returncode != 0:
            tail = proc.stderr.decode(errors="replace")[-800:]
            raise TaskExecutionError(f"task exited {proc.returncode}: {tail}")

        # metrics.json is load-bearing, not a preference: CommandRecipe sets
        # commit_key to <prefix>/metrics.json and the coordinator validates
        # the artifact at that key by sha256. Failing here turns an opaque
        # commit rejection into a clear task error.
        if not (outdir / "metrics.json").is_file():
            raise TaskExecutionError("task produced no metrics.json — nothing to commit")

        total = sum(p.stat().st_size for p in outdir.rglob("*") if p.is_file())
        if total > self.max_output_bytes:
            raise TaskExecutionError(
                f"task output {total} B exceeds the {self.max_output_bytes} B cap"
            )
        return outdir
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_argv_runner.py -v`
Expected: PASS (all 9).

- [ ] **Step 5: Commit**

```bash
git add flashnode/executor/argv_runner.py tests/test_argv_runner.py
git commit -m "feat(executor): ArgvDockerRunner — run user argv in a hardened container

Container-only by construction. Image allowlist, argv shape, and env keys
are all validated before any subprocess call."
```

---

### Task 5: `SubprocessRunner` refuses argv payloads

**Repo: flashnode.**

**Files:**
- Modify: `flashnode/flashnode/executor/runner.py:52-62`
- Test: `flashnode/tests/test_executor.py`

**Interfaces:**
- Consumes: `TaskExecutionError` (same module).
- Produces: no new API — a guard.

This is the single most important guard in the slice. It protects the volunteer who runs `flashnode work` with default flags: Tier 1 is explicitly *not* a security boundary, so it must never gain argv capability by accident.

- [ ] **Step 1: Write the failing test**

Add to `flashnode/tests/test_executor.py`:

```python
def test_subprocess_runner_refuses_argv_payloads(tmp_path):
    """Tier 1 is not a security boundary. An argv payload reaching it would
    be arbitrary code execution on the host with no isolation at all."""
    from flashnode.executor.runner import SubprocessRunner, TaskExecutionError

    with pytest.raises(TaskExecutionError, match="argv"):
        SubprocessRunner().run({"argv": ["python", "evil.py"]}, tmp_path, {})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor.py -k argv -v`
Expected: FAIL — raises about the *module* not being allowlisted, not about argv (wrong error, and the reason is misleading).

- [ ] **Step 3: Add the guard**

In `flashnode/flashnode/executor/runner.py`, at the very top of `SubprocessRunner.run()`, before the module check:

```python
    def run(self, payload: dict, workdir: Path, inputs: dict[str, Path]) -> Path:
        # Tier 1 has no isolation, so it must never execute a caller-supplied
        # command line. Argv workloads are container-only (ArgvDockerRunner);
        # refusing here keeps a misrouted payload from silently running
        # unsandboxed on the host.
        if "argv" in payload:
            raise TaskExecutionError(
                "argv payloads require a sandboxed runner — "
                "start the agent with --runner argv"
            )
        module = payload.get("module", "")
        ...
```

Also update the module docstring's Tier 1 description to name the refusal.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_executor.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add flashnode/executor/runner.py tests/test_executor.py
git commit -m "fix(executor): SubprocessRunner refuses argv payloads

Tier 1 has no isolation; a misrouted argv payload would run unsandboxed
on the volunteer's host."
```

---

### Task 6: `argv_capable` protocol field + placement gate

**Repo: flashruntime.**

**Files:**
- Modify: `flashruntime/flashruntime/protocol/v1alpha1.py` (`NodeRegistration`, ~line 283)
- Modify: `flashruntime/flashruntime/scheduler/__init__.py:116-127`
- Modify: `flashruntime/flashruntime/service/modea.py:295-299` and `212-223`
- Test: `flashruntime/tests/test_scheduler.py`, `flashruntime/tests/test_service_modea.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `NodeRegistration.argv_capable: bool = False`; `NodeView` key `"argv_capable"`. Task 8 sets it from flashnode.

`argv_capable` goes on `NodeRegistration` beside `sandbox_capable` — **not** on `NodeCapabilities`, which carries only hardware facts.

- [ ] **Step 1: Write the failing tests**

Add to `flashruntime/tests/test_scheduler.py`:

```python
from flashruntime.protocol.v1alpha1 import TaskSpec
from flashruntime.scheduler import IsolationAwarePlacement


def _argv_task():
    return TaskSpec(
        task_id="task-000", job_id="job-a", commit_key="job-a/task-000/m.json",
        payload={"argv": ["python", "train.py"],
                 "isolation": {"tier": "sandboxed", "allowFallback": False}},
    )


@pytest.mark.parametrize("value", [None, False, "true", 1, "yes"])
def test_argv_task_ineligible_without_genuine_true(value):
    node = {"node_id": "n1", "sandbox_capable": True}
    if value is not None:
        node["argv_capable"] = value
    assert IsolationAwarePlacement().eligible(_argv_task(), node) is False


def test_argv_task_eligible_on_capable_node():
    node = {"node_id": "n1", "sandbox_capable": True, "argv_capable": True}
    assert IsolationAwarePlacement().eligible(_argv_task(), node) is True


def test_allow_fallback_cannot_bypass_the_argv_gate():
    """allowFallback waives the sandbox capability requirement. It must not
    also waive argv capability, or a submitter could land arbitrary argv on
    a node with no argv runner at all."""
    task = _argv_task()
    task.payload["isolation"]["allowFallback"] = True
    node = {"node_id": "n1", "sandbox_capable": False}
    assert IsolationAwarePlacement().eligible(task, node) is False


def test_non_argv_tasks_are_unaffected():
    task = TaskSpec(task_id="t", job_id="j", commit_key="j/t/m.json",
                    payload={"module": "flashml_workloads.sklearn_trial"})
    assert IsolationAwarePlacement().eligible(task, {"node_id": "n1"}) is True
```

Add to `flashruntime/tests/test_service_modea.py` a check that the claim handler's node view carries the flag:

```python
def test_node_view_exposes_argv_capable(client):
    reg = _registration(node_id="n1")          # existing helper in this file
    reg["argv_capable"] = True
    client.post("/v1alpha1/nodes/register", json=reg)
    nodes = client.get("/v1alpha1/nodes").json()
    assert nodes[0]["argv_capable"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_scheduler.py tests/test_service_modea.py -k "argv or fallback" -v`
Expected: FAIL — `test_argv_task_eligible_on_capable_node` and the allowFallback test fail (the gate does not exist yet).

- [ ] **Step 3: Add the protocol field**

In `flashruntime/flashruntime/protocol/v1alpha1.py`, in `NodeRegistration`, directly after `sandbox_capable`:

```python
    sandbox_capable: bool = False
    #: This node runs an argv-capable sandboxed runner. Defaults False so
    #: every already-deployed agent is excluded from argv work until it is
    #: upgraded and explicitly opted in (security fields fail closed).
    argv_capable: bool = False
```

- [ ] **Step 4: Add the placement gate**

In `flashruntime/flashruntime/scheduler/__init__.py`, at the top of `IsolationAwarePlacement.eligible()`, **before** the isolation block:

```python
    def eligible(self, task: TaskSpec, node: NodeView) -> bool:
        # Checked before the allowFallback waiver below: the waiver relaxes
        # the sandbox-tier requirement, and must never be readable as
        # permission to run argv on a node with no argv runner.
        if "argv" in task.payload and node.get("argv_capable") is not True:
            return False
        isolation = task.payload.get("isolation")
        ...
```

Extend the class docstring with the new rule.

- [ ] **Step 5: Wire it into the node views**

In `flashruntime/flashruntime/service/modea.py`, the claim handler (~line 295):

```python
        node_view = {
            "node_id": req.node_id,
            "sandbox_capable": entry.registration.sandbox_capable,
            "argv_capable": entry.registration.argv_capable,
            "capabilities": entry.registration.capabilities.model_dump(),
        }
```

And the same key in `ModeAState.node_view()` (~line 212-223), so the dashboard and `GET /v1alpha1/nodes` show it.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_scheduler.py tests/test_service_modea.py -v`
Expected: PASS.

- [ ] **Step 7: Run the full suite**

Run: `pytest -q`
Expected: green.

- [ ] **Step 8: Commit**

```bash
git add flashruntime/protocol/v1alpha1.py flashruntime/scheduler/__init__.py \
        flashruntime/service/modea.py tests/test_scheduler.py tests/test_service_modea.py
git commit -m "feat(protocol): argv_capable node field + fail-closed argv placement gate

Checked ahead of the allowFallback waiver so the waiver cannot grant argv
placement on an incapable node."
```

---

### Task 7: `CommandRecipe` tier validation

**Repo: flashruntime.**

**Files:**
- Modify: `flashruntime/flashruntime/recipes/command.py:26-45` (`validate_params`)
- Test: `flashruntime/tests/test_recipes_command.py` (or the existing command-recipe test module)

**Interfaces:**
- Consumes: nothing.
- Produces: no new API — `validate_params()` gains two refusals and reads `FLASHML_ALLOW_UNSANDBOXED_ARGV`.

`validate_params()` currently receives only `params`, but the tier lives on `spec.spec.isolation`. Do the tier check in `expand()`, which has the whole `JobSpec`, and keep `validate_params()` for payload shape.

- [ ] **Step 1: Write the failing tests**

```python
import os
import pytest

from flashruntime.recipes.command import CommandRecipe
from flashruntime.workloads.command import CommandWorkload, to_jobspec
from flashruntime.protocol.v1alpha1 import ImageSpec, IsolationSpec


def _job(tier="sandboxed", allow_fallback=False):
    wl = CommandWorkload(
        command="python train.py",
        image=ImageSpec(reference="ghcr.io/zolli/trainer:1.0"),
        isolation=IsolationSpec(tier=tier, allowFallback=allow_fallback),
    )
    return to_jobspec(wl, name="j")


def test_sandboxed_tier_is_accepted():
    assert CommandRecipe().expand("job-a", _job()) != []


def test_standard_tier_rejected_by_default():
    with pytest.raises(ValueError, match="sandboxed"):
        CommandRecipe().expand("job-a", _job(tier="standard"))


def test_standard_tier_allowed_with_coordinator_opt_in(monkeypatch):
    """Deliberately a coordinator-side env var: a submitter must never be
    able to downgrade the isolation their own code runs under."""
    monkeypatch.setenv("FLASHML_ALLOW_UNSANDBOXED_ARGV", "1")
    assert CommandRecipe().expand("job-a", _job(tier="standard")) != []


def test_allow_fallback_rejected_for_command_jobs():
    with pytest.raises(ValueError, match="allowFallback"):
        CommandRecipe().expand("job-a", _job(allow_fallback=True))
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_recipes_command.py -v`
Expected: FAIL — no exception raised for `standard` or `allowFallback`.

- [ ] **Step 3: Add the validation**

In `flashruntime/flashruntime/recipes/command.py`, add at the top of `expand()`, before the existing `validate_params` call:

```python
import os

...

    def expand(self, job_id: str, spec: JobSpec) -> list[TaskSpec]:
        isolation_spec = spec.spec.isolation
        if isolation_spec.allowFallback:
            # allowFallback waives the sandbox capability requirement at
            # placement time. Honouring it for argv would let a submitter
            # place arbitrary code on an unsandboxed node.
            raise ValueError(
                "command jobs may not set isolation.allowFallback — "
                "argv execution is container-only"
            )
        if isolation_spec.tier != "sandboxed":
            # Coordinator-side opt-in only: the operator running the pool
            # decides, never the submitter.
            if os.environ.get("FLASHML_ALLOW_UNSANDBOXED_ARGV") != "1":
                raise ValueError(
                    f"command jobs require isolation.tier 'sandboxed', got "
                    f"{isolation_spec.tier!r} (set FLASHML_ALLOW_UNSANDBOXED_ARGV=1 "
                    f"on the coordinator to allow a trusted fleet)"
                )
        p = spec.spec.workload.parameters
        ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_recipes_command.py -v`
Expected: PASS.

- [ ] **Step 5: Run the full suite**

Run: `pytest -q`
Expected: green. Existing command-workload tests that submit with the default `standard` tier will now fail — update them to pass `IsolationSpec(tier="sandboxed")`, which is the correct posture for a command job.

- [ ] **Step 6: Commit**

```bash
git add flashruntime/recipes/command.py tests/test_recipes_command.py
git commit -m "feat(recipes): command jobs require sandboxed isolation

Rejects tier 'standard' (coordinator-side FLASHML_ALLOW_UNSANDBOXED_ARGV
opt-in only) and rejects allowFallback, which would otherwise let a
submitter bypass the argv placement gate."
```

---

### Task 8: Agent CLI — `--runner argv` + capability advertisement

**Repo: flashnode.**

**Files:**
- Modify: `flashnode/flashnode/agent/cli.py:51-77,84`
- Modify: `flashnode/flashnode/inventory/capabilities.py:60-100`
- Test: `flashnode/tests/test_capabilities.py`, `flashnode/tests/test_agent.py`

**Interfaces:**
- Consumes: `ArgvDockerRunner` (Task 4); `NodeRegistration.argv_capable` (Task 6).
- Produces: `flashnode work --runner argv`; `discover(..., argv_capable: bool = False)`.

Requires Task 6 to be merged in flashruntime first — `discover()` constructs a `NodeRegistration` from the flashruntime protocol package.

- [ ] **Step 1: Write the failing tests**

Add to `flashnode/tests/test_capabilities.py`:

```python
def test_argv_capable_defaults_false():
    from flashnode.inventory.capabilities import discover
    reg = discover("node-1", kubernetes_node="", node_meta=None)
    assert reg.argv_capable is False


def test_argv_capable_when_requested():
    from flashnode.inventory.capabilities import discover
    reg = discover("node-1", kubernetes_node="", node_meta=None, argv_capable=True)
    assert reg.argv_capable is True
```

Add to `flashnode/tests/test_agent.py`:

```python
def test_argv_runner_requires_an_image_allowlist(monkeypatch, capsys):
    """Refuse to start rather than silently degrade to an unsandboxed tier."""
    from flashnode.agent.cli import main
    monkeypatch.delenv("FLASHNODE_ALLOWED_IMAGES", raising=False)
    assert main(["work", "--runner", "argv"]) == 2
    assert "FLASHNODE_ALLOWED_IMAGES" in capsys.readouterr().err
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_capabilities.py tests/test_agent.py -k argv -v`
Expected: FAIL — `discover()` has no `argv_capable` parameter; `--runner` rejects `argv` as an invalid choice.

- [ ] **Step 3: Add the capability parameter**

In `flashnode/flashnode/inventory/capabilities.py`, add the keyword to `discover()` and pass it through:

```python
def discover(node_id: str, kubernetes_node: str, node_meta=None,
             argv_capable: bool = False) -> NodeRegistration:
    ...
    return NodeRegistration(
        ...
        sandbox_capable=sandbox_capable,
        # Set by the agent when it is actually running an argv-capable
        # runner — never inferred, so the coordinator's fail-closed gate
        # cannot be satisfied by a node that merely has docker installed.
        argv_capable=argv_capable,
        ...
    )
```

- [ ] **Step 4: Wire the CLI**

In `flashnode/flashnode/agent/cli.py`:

```python
    parser.add_argument(
        "--runner",
        choices=["subprocess", "docker", "argv"],
        default=os.environ.get("FLASHNODE_RUNNER", "subprocess"),
        help="task execution tier (docker/argv need FLASHNODE_ALLOWED_IMAGES)",
    )
```

```python
    runner = None
    if opts.runner in ("docker", "argv"):
        images = frozenset(
            i.strip() for i in os.environ.get("FLASHNODE_ALLOWED_IMAGES", "").split(",") if i.strip()
        )
        if not images:
            print(
                f"flashnode work: --runner {opts.runner} requires FLASHNODE_ALLOWED_IMAGES "
                "(comma-separated image references) — refusing to start with an "
                "empty allowlist",
                file=sys.stderr,
            )
            return 2
        if opts.runner == "docker":
            from flashnode.executor.docker_runner import DockerRunner

            runner = DockerRunner(allowed_images=images)
        else:
            from flashnode.executor.argv_runner import ArgvDockerRunner

            runner = ArgvDockerRunner(
                allowed_images=images,
                cpus=float(os.environ.get("FLASHNODE_MAX_CPUS", "2.0")),
                memory_gb=float(os.environ.get("FLASHNODE_MAX_MEMORY_GB", "2.0")),
                timeout_seconds=float(os.environ.get("FLASHNODE_TASK_TIMEOUT_S", "3600")),
                max_output_bytes=int(
                    os.environ.get("FLASHNODE_MAX_OUTPUT_BYTES", str(2 * 1024**3))
                ),
            )
```

And advertise it at registration:

```python
    registration = discover(
        node_id, kubernetes_node="", node_meta=None,
        argv_capable=(opts.runner == "argv"),
    )
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_capabilities.py tests/test_agent.py -v`
Expected: PASS.

- [ ] **Step 6: Run the flashnode suite**

Run: `pytest -q`
Expected: green.

- [ ] **Step 7: Commit**

```bash
git add flashnode/agent/cli.py flashnode/inventory/capabilities.py \
        tests/test_capabilities.py tests/test_agent.py
git commit -m "feat(agent): flashnode work --runner argv + argv_capable advertisement

Capability is set from the runner actually in use, never inferred, and the
agent refuses to start with an empty image allowlist."
```

---

### Task 9: Integration tests — real Docker + volunteer-kill

**Repos: flashnode (runner integration), workspace `e2e/` (kill test).**

**Files:**
- Create: `flashnode/tests/integration/test_argv_runner_docker.py`
- Test: run with `pytest -m integration`

**Interfaces:**
- Consumes: `ArgvDockerRunner` (Task 4), the `--runner argv` CLI (Task 8).
- Produces: nothing new — evidence.

Unit tests assert on *constructed argv*; they cannot prove the kernel actually enforced anything. These prove the guarantees are real.

- [ ] **Step 1: Write the integration tests**

Create `flashnode/tests/integration/test_argv_runner_docker.py`:

```python
"""Real-docker proof that the sandbox flags are enforced, not just passed.

Opt-in: pytest -m integration. Auto-skips without a docker daemon.
"""
import shutil
import subprocess
from pathlib import Path

import pytest

from flashnode.executor.argv_runner import ArgvDockerRunner
from flashnode.executor.runner import TaskExecutionError

pytestmark = pytest.mark.integration

IMAGE = "python:3.11-alpine"


def _docker_available() -> bool:
    if shutil.which("docker") is None:
        return False
    return subprocess.run(["docker", "info"], capture_output=True).returncode == 0


pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not _docker_available(), reason="needs a docker daemon")]


def _runner():
    return ArgvDockerRunner(allowed_images=frozenset({IMAGE}), timeout_seconds=120.0)


def test_argv_task_runs_and_produces_metrics(tmp_path):
    payload = {"image": IMAGE, "task_id": "task-000",
               "argv": ["python", "-c",
                        "open('/work/out/metrics.json','w').write('{\"acc\": 1.0}')"]}
    outdir = _runner().run(payload, tmp_path, {})
    assert (outdir / "metrics.json").read_text() == '{"acc": 1.0}'


def test_network_is_really_off(tmp_path):
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "import socket; socket.create_connection(('1.1.1.1', 53), 5)"]}
    with pytest.raises(TaskExecutionError):
        _runner().run(payload, tmp_path, {})


def test_rootfs_is_really_read_only(tmp_path):
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c", "open('/etc/passwd','a').write('x')"]}
    with pytest.raises(TaskExecutionError):
        _runner().run(payload, tmp_path, {})


def test_inputs_are_visible_at_work_inputs(tmp_path):
    (tmp_path / "inputs").mkdir()
    (tmp_path / "inputs" / "data.txt").write_text("hello")
    payload = {"image": IMAGE, "task_id": "t",
               "argv": ["python", "-c",
                        "d=open('/work/inputs/data.txt').read();"
                        "open('/work/out/metrics.json','w').write('{\"n\": %d}' % len(d))"]}
    outdir = _runner().run(payload, tmp_path, {"data": tmp_path / "inputs" / "data.txt"})
    assert '"n": 5' in (outdir / "metrics.json").read_text()
```

- [ ] **Step 2: Run them**

Run: `cd ../flashnode && pytest tests/integration/test_argv_runner_docker.py -m integration -v`
Expected: PASS (or all SKIP if no docker daemon — start colima/Docker Desktop to actually verify, and do verify at least once).

- [ ] **Step 3: Run the volunteer-kill proof end to end**

This reuses the existing recovery proof over the new argv path. From the workspace root:

```bash
make local-coordinator JOIN_CODE=LOCAL-2026
# in a second terminal:
FLASHNODE_ALLOWED_IMAGES=python:3.11-alpine \
FLASHNODE_JOIN_CODE=LOCAL-2026 \
  .venv/bin/flashnode work --runner argv --coordinator http://localhost:8100
```

Submit a command job with `isolation.tier: "sandboxed"`, kill the agent mid-task (`kill -9`), start a second agent, and confirm the lease expires and the task is completed by the replacement. Record the observed MTTD/MTTR.

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_argv_runner_docker.py
git commit -m "test(integration): prove argv sandbox flags are enforced by the kernel

Network-off, read-only rootfs, and input staging verified against a real
docker daemon rather than asserted on constructed argv."
```

---

### Task 10: Documentation

**Repo: flashruntime.**

**Files:**
- Create: `flashruntime/docs/guides/donate-a-machine.md`
- Modify: `flashruntime/docs/guides/bring-your-code.md`
- Modify: `flashruntime/CLAUDE.md` (status section), workspace `PROGRESS.md`
- Modify: `flashruntime/scripts/build_docs.py` nav if guides are enumerated there

**Interfaces:**
- Consumes: everything above.
- Produces: user-facing docs.

- [ ] **Step 1: Write the volunteer guide**

Create `flashruntime/docs/guides/donate-a-machine.md` covering:
- What joining does, and the honest trust statement: **your machine runs other people's code**; it is confined to a hardened container with no network, a read-only rootfs, no capabilities, and CPU/memory/PID caps — but Docker shares the host kernel, so a container escape is not impossible.
- Quickstart:
  ```bash
  export FLASHNODE_ALLOWED_IMAGES=ghcr.io/zolli/trainer:1.0
  export FLASHNODE_MAX_CPUS=4 FLASHNODE_MAX_MEMORY_GB=8
  flashnode work --runner argv --coordinator https://<coordinator>
  ```
- The consent knobs table: `FLASHNODE_ALLOWED_IMAGES`, `FLASHNODE_MAX_CPUS`, `FLASHNODE_MAX_MEMORY_GB`, `FLASHNODE_TASK_TIMEOUT_S`, `FLASHNODE_MAX_OUTPUT_BYTES`.
- Known limitations, stated plainly: **no result verification yet** (a node that lies is currently believed — slice C), **one shared join code** (slice B), **disk fill is capped only at upload time**, **no GPU work yet** (slice D).

- [ ] **Step 2: Document the no-network constraint for job authors**

Add a prominent section to `flashruntime/docs/guides/bring-your-code.md`:

> **Jobs on volunteer nodes run with no network.** Your command cannot
> `pip install`, download a dataset, or pull from HuggingFace. Everything must
> be baked into the pinned image, or passed as an `artifact://` input that the
> agent stages at `/work/inputs/` before your code starts. Write outputs to
> `/work/out/`; `metrics.json` is required — it is the artifact the coordinator
> validates by sha256 at commit time.

Also state that `mode: "coordinated"` (torchrun/DDP) is not available on volunteer nodes.

- [ ] **Step 3: Verify the docs build**

Run: `source .venv/bin/activate && python scripts/build_docs.py --check`
Expected: OK, no broken links.

- [ ] **Step 4: Update status docs**

- `flashruntime/CLAUDE.md`: move "FlashNode argv runner" out of **Missing** and into the built list; update the test counts.
- Workspace `PROGRESS.md`: add a newest-first entry per the logging protocol — what/why, how verified (real test counts and the observed volunteer-kill result), gotchas (the second collision bug in `claim()`; the `NodeRegistration` vs `NodeCapabilities` placement of `argv_capable`), and Next (slice B: per-node identity).

- [ ] **Step 5: Final verification**

Run from `flashruntime`: `pytest -q && python scripts/build_docs.py --check && scripts/audit_secrets.sh`
Run from `flashnode`: `pytest -q`
Expected: all green, audit CLEAN.

- [ ] **Step 6: Commit**

```bash
git add docs/guides/donate-a-machine.md docs/guides/bring-your-code.md CLAUDE.md
git commit -m "docs: donate-a-machine guide + no-network constraint for job authors

States the volunteer trust model honestly, including what is NOT yet
defended against (lying nodes, shared join code)."
```

---

## Self-Review

**Spec coverage:**

| Spec section | Task |
|---|---|
| `harden_args()` (Arch §1) | 3 |
| `ArgvDockerRunner` (Arch §2) | 4 |
| argv container-only, 3 enforcement points (Arch §3) | 5 (runner), 6 (gate), 7 (recipe) |
| Injection surfaces closed (Arch §3) | 4 |
| `argv_capable` + placement gate (Arch §4) | 6 |
| Agent CLI + env knobs (Arch §5) | 8 |
| Composite lease key + migration (Arch §6) | 1, 2 |
| Error-handling table | 4 (runner rows), 5, 8 (docker-absent → empty-allowlist refusal) |
| Testing: unit flashnode | 3, 4, 5 |
| Testing: unit flashruntime | 1, 2, 6, 7 |
| Testing: integration | 9 |
| Known gaps documented | 10 |

**Gaps found and closed during review:**
- The spec's error table says "`docker` binary absent → agent refuses to start". Task 8 covers the empty-allowlist refusal; a missing `docker` binary surfaces as a `TaskExecutionError` at first task rather than at startup. Acceptable for this slice — noted here so it is a known limitation rather than a silent miss.
- `validate_params()` receives only `params`, but the tier lives on `spec.spec.isolation`; Task 7 therefore places the check in `expand()`. The spec said `validate_params()` — this plan is the more accurate placement.

**Type consistency:** `get(job_id, task_id)` is used identically in Tasks 1, 2. `harden_args(workdir, *, cpus, memory_gb, pids_limit)` is defined in Task 3 and called with the same keywords in Task 4. `CONTAINER_WORKDIR` is defined once in Task 3 and imported in Tasks 3 and 4. `argv_capable` is spelled identically in Tasks 6 and 8.

**Ordering constraint:** Task 8 (flashnode) requires Task 6 (flashruntime) merged first — `discover()` builds a `NodeRegistration` from the flashruntime protocol package.
