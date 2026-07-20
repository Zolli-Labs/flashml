# FlashML Provider Connector Spec

This document is the contract for adding a GPU/compute provider to FlashML. If you
can implement the methods below and pass the conformance requirements, the
provider can execute every `DistributedAlgorithm` without engine changes.

Audience: connector authors (including future-us adding Vast.ai, Lambda, etc.).

---

## Provider Matrix

| Connector | Status | Compute model | Storage backend |
|---|---|---|---|
| `local` | **Implemented** | Thread pool on the host | Temp directory |
| `runpod` | Not implemented (Phase 1) | RunPod Flash serverless endpoints | NetworkVolume (S3-compatible) |
| `ssh` | Planned (Phase 4) | Bring-your-own hosts over SSH | rsync/scp staging dir |
| `vast` | Roadmap | Vast.ai instances | Instance disk + S3 |
| `lambda` | Roadmap | Lambda Cloud instances | Shared filesystem |
| `coreweave` | Roadmap | CoreWeave Kubernetes | Object storage |

---

## Design Principles

1. **Connectors move bytes, not meaning.** A connector never inspects task
   payloads or knows what algorithm is running. It provisions workers, executes
   opaque tasks, moves blobs, and tears down.
2. **Small state over the wire, big data through Storage.** Task payloads are
   assumed to be small (KBs — model state, shard IDs, iteration numbers). Datasets
   and shards go through the `Storage` interface exactly once, at plan time. This
   invariant came from RunPod Flash's 10MB payload cap, but it's enforced for all
   connectors because it's what makes iteration N+1 free.
3. **Teardown is sacred.** Rented GPUs bill by the minute. `teardown()` must be
   idempotent, must work after a *partial* provision (3 of 8 workers came up, then
   an error), and must be safe to call twice.
4. **Fail loud, fail typed.** Raise the shared exception types
   (`ProvisionError`, `TaskError`, `StorageError`, `AuthError`) so the engine can
   retry, re-dispatch, or surface a clean message — never provider-specific
   exceptions across the interface boundary.

---

## The Interface

```python
# flashml/providers/base.py

@dataclass
class ResourceSpec:
    accelerator: str          # normalized tier, see table below
    workers: int
    region: str | None = None

@dataclass
class Offer:
    provider: str
    accelerator: str          # normalized tier this offer satisfies
    native_type: str          # provider's own name, e.g. "ADA_24", "RTX 4090"
    price_per_hour: float | None
    available: bool

@dataclass
class WorkerSpec:
    accelerator: str
    entrypoint: str           # dotted module:function resolved on the worker
    dependencies: list[str]   # pip packages the task code needs
    env: dict[str, str]

@dataclass
class Task:
    task_id: str
    job_id: str
    payload: dict             # opaque to the connector; JSON-serializable; small
    timeout_s: float

@dataclass
class TaskResult:
    task_id: str
    ok: bool
    payload: dict | None      # opaque to the connector
    error: str | None
    duration_ms: int
    worker_id: str


class Provider(ABC):
    """One instance per Cluster. Registered by name in providers/registry.py."""

    name: str

    @abstractmethod
    def offers(self, spec: ResourceSpec) -> list[Offer]:
        """Availability + pricing for a normalized request.
        May return price_per_hour=None if the provider has no pricing API.
        Must not provision anything."""

    @abstractmethod
    def provision(self, spec: WorkerSpec, count: int) -> "WorkerPool":
        """Bring up `count` workers that can execute FlashML tasks.
        Blocks until the pool can accept submissions (workers may still be
        cold-starting; submit() is allowed to queue).
        On any failure, must clean up whatever it started, then raise
        ProvisionError."""

    @abstractmethod
    def storage(self) -> "Storage":
        """A blob store REACHABLE FROM THIS PROVIDER'S WORKERS.
        The engine writes shards here at plan time; tasks reference blobs
        by key. Local disk for `local`, NetworkVolume for `runpod`, etc."""

    @abstractmethod
    def teardown(self, pool: "WorkerPool") -> None:
        """Release all compute. Idempotent. Never raises on already-gone."""


class WorkerPool(ABC):
    @abstractmethod
    def submit(self, task: Task) -> "TaskHandle": ...

    @abstractmethod
    def gather(self, handles: list["TaskHandle"], timeout_s: float) -> list[TaskResult]:
        """Wait for all handles. A failed task returns TaskResult(ok=False),
        it does not raise — the scheduler decides about retries."""

    @property
    @abstractmethod
    def size(self) -> int: ...


class Storage(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes) -> None: ...
    @abstractmethod
    def get(self, key: str) -> bytes: ...
    @abstractmethod
    def exists(self, key: str) -> bool: ...
    @abstractmethod
    def delete_prefix(self, prefix: str) -> None:
        """Cleanup for a finished job's shards (`{job_id}/...`)."""
```

### How workers read shards

Tasks reference data by storage key (`{job_id}/shard_{i}`), never by inline
payload. The connector decides how a worker resolves a key:

- `local`: the key is a file path under the pool's temp dir.
- `runpod`: the key maps to `/runpod-volume/{key}` on the mounted NetworkVolume.
- `ssh`: keys are pre-staged to a fixed directory on each host before the loop starts.

The engine guarantees every key a task references was `put()` before any
`submit()` for that job.

---

## Normalized Accelerator Tiers

Users request tiers; connectors map them to native types. A connector must map
every tier it supports and raise `ProvisionError("unsupported accelerator")` for
the rest — never silently substitute a different class of hardware.

| Tier | Meaning | `runpod` mapping (example) |
|---|---|---|
| `cpu-small` | 1–2 vCPU, ≤4GB | `CpuInstanceType.CPU3C_1_2` |
| `cpu-large` | 4+ vCPU, 8GB+ | `CpuInstanceType.CPU5C_4_8` |
| `gpu-24gb` | 24GB VRAM class | `[GpuGroup.AMPERE_24, GpuGroup.ADA_24]` |
| `gpu-48gb` | 48GB VRAM class | `GpuGroup.ADA_48` |
| `gpu-80gb` | 80GB VRAM class | `GpuGroup.ADA_80` |

Connectors are encouraged to map one tier to *several* native types when the
provider supports fallback (RunPod auto-switches GPU SKUs by live supply when
max workers ≥ 5 — a real availability win; see [TOOLS.md](TOOLS.md)).

---

## Credentials

Convention: each connector reads `{PROVIDER}_API_KEY`-style env vars
(`RUNPOD_API_KEY`, …) and accepts an explicit `credentials=` dict passed to
`flashml.Cluster(...)` that overrides the environment. Connectors must raise
`AuthError` at `provision()`/`offers()` time — not deep inside a task — when
credentials are missing or rejected.

---

## Conformance Suite

Every connector must pass `flashml.testing.provider_conformance` before being
registered. The suite runs against a live instance of the connector (for paid
providers, on the smallest/cheapest tier) and checks:

1. **Lifecycle** — `provision(n=2)` → pool of 2 → `teardown` → second `teardown` is a no-op.
2. **Round trip** — submit an echo task; payload comes back intact with `ok=True`.
3. **Parallelism** — n sleep tasks on n workers finish in ~1× task time, not n×.
4. **Storage visibility** — `put()` a blob, submit a task that reads it by key
   from inside the worker, verify contents.
5. **Failure typing** — a task that raises returns `TaskResult(ok=False, error=...)`;
   bad credentials raise `AuthError`; unsupported tier raises `ProvisionError`.
6. **Partial-provision cleanup** — force a mid-provision failure; assert nothing
   is left running/billing afterward.
7. **Payload discipline** — a >1MB task payload is rejected by the engine before
   it reaches the connector (this one tests the engine, but runs per-connector).

The Phase 4 milestone ("write the `ssh` connector using only this document") is the
standing test that this spec is complete. If the SSH author needs to read engine
source to finish, the spec has a hole — fix the spec.

---

## Writing a New Connector: Checklist

- [ ] `flashml/providers/<name>/provider.py` implementing `Provider` + `WorkerPool`
- [ ] `Storage` implementation reachable from that provider's workers
- [ ] Tier mapping table (every supported normalized tier → native type(s))
- [ ] Credentials via env var + `credentials=` override, `AuthError` on failure
- [ ] Registered in `flashml/providers/registry.py`
- [ ] Conformance suite green (attach the run log to the PR)
- [ ] Row added to the Provider Matrix in this document
