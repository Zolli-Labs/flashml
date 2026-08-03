"""Mode A over HTTP: the lease coordinator, node registry, and local
artifact hosting for the self-hosted profile.

This module makes the pure-library pieces reachable by remote workers:

- **Leases** — HTTP verbs over the `LeaseManager` (claim / heartbeat /
  complete / fail) plus a background sweep. FlashNode's executor is the
  intended client, but anything that speaks the protocol can pull work.
- **Node registry** — minimal register/heartbeat/list so a self-hosted
  coordinator knows its workers (FlashML Cloud fronts this with join codes
  and trust tiers in the managed product; the wire models are the same).
- **Local artifacts** — PUT/GET raw bytes under a local directory, so
  shared data (datasets in, trial outputs back) needs no cloud and no
  MinIO: the coordinator *is* the artifact host for the local loop. Keys
  are sha256-verified on upload; the same `artifact://` URIs used by the
  cloud stores apply, keeping job specs portable.

Job → task expansion lives here too: a JobSpec with
`execution.backend: leases` and `workload.type: hyperparameter_search`
becomes N `TaskSpec`s whose payloads carry the executor contract
(module, params, input artifact keys, output prefix).
"""

from __future__ import annotations

import hashlib
import itertools
import math
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel

import flashruntime.recipes.command  # noqa: F401 — registers the "command" recipe
from flashruntime.leases import LeaseManager
from flashruntime.protocol.v1alpha1 import (
    ArtifactRecord,
    ExecutionEvidence,
    JobSpec,
    NodeHeartbeat,
    NodeRegistration,
    TaskSpec,
    TaskState,
)
from flashruntime.recipes import recipe_for
from flashruntime.scheduler import IsolationAwarePlacement

NODE_OFFLINE_AFTER_S = 15.0

# Task modules the *coordinator* will hand out. The executor enforces its own
# allowlist too — both ends fail closed.
ALLOWED_TASK_MODULES = {
    "flashml_workloads.sklearn_trial",
    "flashml_workloads.kmeans_shard",
    "flashml_workloads.sgd_trainer",
    "flashml_workloads.fedavg_worker",
}


#: Longest lease a submitter may ask for. A lease deadline is the ONLY thing
#: that returns an abandoned task to the queue, so an unbounded
#: `lease_seconds` is a denial-of-service knob: `1e9` pins a task to a
#: machine that closed its laptop for ~31 years, and `float("inf")` does not
#: even survive the claim path (`timedelta(seconds=inf)` raises OverflowError
#: *inside* the coordinator). One hour is far above any real task here and
#: far below "forever".
MAX_LEASE_SECONDS = 3600.0

#: The worker params `_expand_fedavg` must forward. Single source of truth so
#: the expansion and `flashml_workloads.fedavg_worker`'s reads cannot drift —
#: `tests/test_service_fedavg.py` binds this tuple to the worker's actual
#: parameter accesses by parsing its source.
FEDAVG_WORKER_PARAM_KEYS = ("local_steps", "lr", "batch_size", "seed",
                            "in_dim", "hidden", "out_dim", "dataset_size")

#: Params `_expand_fedavg` computes itself per task, so the submitter does
#: not supply them even though the worker reads them.
FEDAVG_DRIVER_SUPPLIED_KEYS = ("round", "shard", "num_shards")


class ExpansionError(ValueError):
    """The JobSpec cannot be expanded into tasks (bad workload/parameters)."""


def _lease_seconds(params: dict, default: float) -> float:
    """Validated, clamped `lease_seconds` from submitter-supplied params."""
    raw = params.get("lease_seconds", default)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        raise ExpansionError(
            f"lease_seconds must be a number, got {raw!r}"
        ) from None
    if not math.isfinite(value):
        raise ExpansionError(
            f"lease_seconds must be finite, got {raw!r} (a non-finite lease "
            "deadline overflows timedelta in the claim path)"
        )
    if value <= 0:
        raise ExpansionError(f"lease_seconds must be > 0, got {value}")
    return min(value, MAX_LEASE_SECONDS)


def expand_tasks(job_id: str, spec: JobSpec) -> list[TaskSpec]:
    """Turn a lease-mode JobSpec into independent TaskSpecs.

    hyperparameter_search parameters:
      trials:      explicit list of param dicts, or
      grid:        {param: [values, ...]} — cartesian product
      module:      task module (allowlisted; default sklearn_trial)
      inputs:      {name: "artifact://key"} shared data, downloaded per task
      lease_seconds: heartbeat window per attempt (default 60)
    """
    workload = spec.spec.workload
    try:
        recipe = recipe_for(workload.type)
    except LookupError:
        recipe = None
    if recipe is not None:
        try:
            return recipe.expand(job_id, spec)
        except ValueError as exc:
            raise ExpansionError(str(exc)) from None
    if workload.type == "sharded_kmeans":
        return _expand_kmeans(job_id, spec)
    if workload.type == "federated_averaging":
        return _expand_fedavg(job_id, spec)
    if workload.type != "hyperparameter_search":
        raise ExpansionError(
            f"lease backend supports workload types 'hyperparameter_search', "
            f"'sharded_kmeans' and 'federated_averaging', got '{workload.type}'"
        )
    p = workload.parameters
    trials: list[dict] = list(p.get("trials") or [])
    if not trials and p.get("grid"):
        grid: dict[str, list] = p["grid"]
        keys = sorted(grid)
        trials = [dict(zip(keys, combo)) for combo in itertools.product(*(grid[k] for k in keys))]
    if not trials:
        raise ExpansionError("hyperparameter_search needs 'trials' (list) or 'grid' (dict of lists)")

    module = p.get("module", "flashml_workloads.sklearn_trial")
    if module not in ALLOWED_TASK_MODULES:
        raise ExpansionError(f"task module '{module}' is not allowlisted")
    inputs = dict(p.get("inputs") or {})
    for name, uri in inputs.items():
        if not str(uri).startswith("artifact://"):
            raise ExpansionError(f"input '{name}' must be an artifact:// URI, got {uri!r}")

    checkpoint = p.get("checkpoint")  # non-None turns the executor's relay on
    # Stamp the isolation requirement so the placement gate can fail closed —
    # a sandboxed job must never lease to a non-sandbox node (mirrors
    # recipes/command.py; the legacy expansions were dropping this).
    isolation = {
        "tier": spec.spec.isolation.tier,
        "allowFallback": spec.spec.isolation.allowFallback,
    }
    tasks = []
    for i, params in enumerate(trials):
        task_id = f"trial-{i:03d}"
        payload = {
            "module": module,
            "params": params,
            "inputs": inputs,
            "output_prefix": f"jobs/{job_id}/{task_id}/",
            "task_id": task_id,
            # the docker-runner tier resolves and allowlists this
            "image": spec.spec.image.reference,
            "isolation": isolation,
        }
        if checkpoint is not None:
            payload["checkpoint"] = checkpoint
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_id=job_id,
                commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                lease_seconds=_lease_seconds(p, 60.0),
                payload=payload,
            )
        )
    return tasks


def _expand_kmeans(job_id: str, spec: JobSpec) -> list[TaskSpec]:
    """One K-means *iteration*: one task per data shard, each computing
    partial sums against the broadcast centroids. The driver
    (`flashml_workloads.kmeans_driver`) reduces and submits the next
    iteration as a new job — stage composition, not a new execution mode."""
    p = spec.spec.workload.parameters
    shards: list[str] = list(p.get("shards") or [])
    centroids = p.get("centroids")
    if not shards or not centroids:
        raise ExpansionError("sharded_kmeans needs 'shards' (artifact:// list) and 'centroids'")
    for uri in shards:
        if not str(uri).startswith("artifact://"):
            raise ExpansionError(f"shard must be an artifact:// URI, got {uri!r}")
    iteration = int(p.get("iteration", 0))
    # Same fail-closed stamp as the hyperparameter_search path (mirrors
    # recipes/command.py) — without it a sandboxed job leases anywhere.
    isolation = {
        "tier": spec.spec.isolation.tier,
        "allowFallback": spec.spec.isolation.allowFallback,
    }

    tasks = []
    for i, shard_uri in enumerate(shards):
        task_id = f"it{iteration:02d}-shard-{i:03d}"
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_id=job_id,
                commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                lease_seconds=_lease_seconds(p, 60.0),
                payload={
                    "module": "flashml_workloads.kmeans_shard",
                    "params": {"centroids": centroids},
                    "inputs": {"shard": shard_uri},
                    "output_prefix": f"jobs/{job_id}/{task_id}/",
                    "task_id": task_id,
                    "image": spec.spec.image.reference,
                    "isolation": isolation,
                },
            )
        )
    return tasks


def _expand_fedavg(job_id: str, spec: JobSpec) -> list[TaskSpec]:
    """One federated-averaging *round*: one task per shard, each training
    locally from the round's broadcast weights. The driver
    (`flashml_workloads.fedavg_driver`) reduces the deltas and submits the
    next round — same stage-composition pattern as `_expand_kmeans`.
    """
    p = spec.spec.workload.parameters
    num_shards = int(p.get("num_shards", 0))
    if num_shards < 1 or num_shards > 999:
        raise ExpansionError(
            f"federated_averaging needs 1 <= num_shards <= 999, got {num_shards} "
            "(task ids are zero-padded to 3 digits and are sorted as strings)"
        )

    inputs: dict[str, str] = {}
    weights = p.get("weights")
    if weights is not None:
        if not str(weights).startswith("artifact://"):
            raise ExpansionError(
                f"input 'weights' must be an artifact:// URI, got {weights!r}"
            )
        inputs["weights"] = weights

    isolation = {
        "tier": spec.spec.isolation.tier,
        "allowFallback": spec.spec.isolation.allowFallback,
    }
    # Every one of these is read unconditionally by fedavg_worker. Dropping a
    # missing key here would defer the failure to a KeyError inside a container
    # on a volunteer's machine, where it burns an attempt and reads as a node
    # fault rather than a bad submission. Fail at expansion instead.
    worker_keys = FEDAVG_WORKER_PARAM_KEYS
    missing = [k for k in worker_keys if k not in p]
    if missing:
        raise ExpansionError(
            f"federated_averaging is missing required parameters: {sorted(missing)}"
        )

    tasks = []
    for shard in range(num_shards):
        task_id = f"shard-{shard:03d}"
        params = {k: p[k] for k in worker_keys}
        params.update({"round": int(p.get("round", 0)),
                       "shard": shard, "num_shards": num_shards})
        tasks.append(
            TaskSpec(
                task_id=task_id,
                job_id=job_id,
                commit_key=f"jobs/{job_id}/{task_id}/metrics.json",
                max_attempts=spec.spec.retryPolicy.maxTaskAttempts,
                lease_seconds=_lease_seconds(p, 120.0),
                payload={
                    "module": "flashml_workloads.fedavg_worker",
                    "params": params,
                    "inputs": inputs,
                    "output_prefix": f"jobs/{job_id}/{task_id}/",
                    "task_id": task_id,
                    "image": spec.spec.image.reference,
                    "isolation": isolation,
                },
            )
        )
    return tasks


class _NodeEntry(BaseModel):
    registration: NodeRegistration
    last_heartbeat: datetime
    accepted_tasks: int = 0


class ModeAState:
    """Shared state behind the router. In-memory by design for the local
    loop (the ledger keeps the durable event history); the Stage-6 upgrade
    swaps the store for Postgres without touching the endpoints."""

    def __init__(
        self,
        manager: LeaseManager,
        artifacts_dir: Path,
        join_code: str | None = None,
        max_artifact_bytes: int = 256 * 1024 * 1024,
        authenticator: NodeAuthenticator | None = None,
    ):
        self.manager = manager
        self.artifacts_dir = artifacts_dir
        self.join_code = join_code  # None = open registration (self-hosted default)
        self.max_artifact_bytes = max_artifact_bytes
        self.nodes: dict[str, _NodeEntry] = {}
        self.lease_jobs: set[str] = set()  # job_ids running on the lease path
        from flashruntime.service.auth import NodeAuthenticator, authenticator_from_env

        self.authenticator: NodeAuthenticator = authenticator or authenticator_from_env()

    def node_view(self) -> list[dict]:
        now = datetime.now(timezone.utc)
        out = []
        for entry in self.nodes.values():
            age = (now - entry.last_heartbeat).total_seconds()
            out.append(
                {
                    "node_id": entry.registration.node_id,
                    "hostname": entry.registration.hostname,
                    "environment": entry.registration.environment,
                    "argv_capable": entry.registration.argv_capable,
                    "module_capable": entry.registration.module_capable,
                    "capabilities": entry.registration.capabilities.model_dump(),
                    "online": age < NODE_OFFLINE_AFTER_S,
                    "last_heartbeat_age_s": round(age, 1),
                    "accepted_tasks": entry.accepted_tasks,
                }
            )
        return sorted(out, key=lambda n: n["node_id"])


class ClaimRequest(BaseModel):
    node_id: str
    job_id: str | None = None


class CompleteRequest(BaseModel):
    output_sha256: str
    #: What the agent measured about the run it is committing. **Optional,
    #: and it must stay optional.** Every agent deployed today predates this
    #: field; making it required would 422 every completion in the fleet the
    #: moment this coordinator shipped, and the fleet is not reachable to
    #: upgrade first. Same fail-safe polarity as `NodeRegistration
    #: .module_capable` and for the same reason — availability, not security:
    #: an absent block costs a verification signal, a required one costs
    #: every task.
    #:
    #: This coordinator ACCEPTS AND IGNORES it. It keeps no verifications
    #: ledger, and inventing somewhere to put this would put the runtime in
    #: the business of judging its own volunteers; the cloud API reads it and
    #: records a verdict. Absent evidence is `unknown` there — never `pass`.
    evidence: ExecutionEvidence | None = None


class FailRequest(BaseModel):
    reason: str


def _output_valid(artifacts_dir: Path, commit_key: str, claimed_sha256: str) -> bool:
    path = artifacts_dir / commit_key
    if not path.is_file():
        return False
    return hashlib.sha256(path.read_bytes()).hexdigest() == claimed_sha256


def _safe_key(key: str) -> str:
    """Artifact keys are relative paths under the artifacts dir — refuse
    anything that could escape it."""
    if key.startswith("/") or ".." in key.split("/"):
        raise HTTPException(status_code=400, detail=f"invalid artifact key: {key}")
    return key


def _bearer(request: Request) -> str | None:
    header = request.headers.get("Authorization") or ""
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token.strip() else None


def _is_operator(state, token: str | None) -> bool:
    """Is this a driver credential (authenticated, attributable, unscoped)?

    Read defensively: the cloud supplies its own authenticator (Plan 3), and
    one written before this concept existed has no `is_operator`. A missing
    or non-bool answer means "no operators here" — the confinement stays on.
    """
    checker = getattr(state.authenticator, "is_operator", None)
    if checker is None:
        return False
    return checker(token) is True


def _authenticated_node(state, token: str | None) -> str:
    """The caller's node_id, or 401. Callers must have checked `enforcing`."""
    node_id = state.authenticator.authenticate(token)
    if not isinstance(node_id, str) or not node_id:
        raise HTTPException(status_code=401, detail="invalid or missing node token")
    return node_id


#: An operator may name the machine it is forwarding for, on writes AND on
#: the lease lifecycle. Honoured ONLY from an operator credential, and read in
#: exactly one place — see `_write_identity`.
DELEGATION_HEADER = "X-FlashML-On-Behalf-Of"


def _delegated_node(request: Request) -> str | None:
    """The node named by the delegation header, or None if absent.

    Never call this for a non-operator caller: for anyone else the header is
    not authoritative, so it must not be parsed either — a volunteer that
    cannot be believed also cannot be allowed to turn its own writes into
    400s, and probing the delegation path should tell it nothing.
    """
    values = request.headers.getlist(DELEGATION_HEADER)
    if not values:
        return None
    if len(values) > 1:
        # Two values means somebody appended one. The API forwarding an
        # agent's request would produce exactly that if it failed to strip
        # the agent's copy, and Starlette would hand us the *first* — an
        # ordering the agent controls. Refuse rather than pick a winner.
        raise HTTPException(
            status_code=400,
            detail=f"ambiguous {DELEGATION_HEADER}: {len(values)} values",
        )
    # An empty value is deliberately returned as "" rather than None: it is a
    # header that was sent, and `_write_identity` fails it closed. Folding it
    # into None would let an API bug that emits a blank header silently
    # restore unscoped operator reach.
    return values[0].strip()


def _write_identity(state, request: Request) -> str | None:
    """Which machine is this request acting as? A node_id, or None for an
    unscoped operator that named nobody. 401 if the caller is neither.

    THE single place delegation is decided — the only reader of
    `DELEGATION_HEADER` in the repo. Every authorization surface funnels
    through here: artifacts key writes by prefix, checkpoints key them by the
    (job, task) pair, and the lease lifecycle keys them by lease holder. If
    each resolved the caller for itself the three would drift, and the drift
    would be a silent hole — this repo has been bitten by exactly that seam
    twice. (The name is historical: it predates the lifecycle endpoints. It
    resolves *identity*, not writes.)

    Delegation can only ever *narrow*. An operator with no header keeps the
    unscoped driver reach it has always had on writes; the moment it asserts
    an identity it is authorized precisely as that node would have been, so a
    driver speaking for node-a loses its own `jobs/{job}/round-NNN/` keys.
    """
    token = _bearer(request)
    if _is_operator(state, token):
        delegated = _delegated_node(request)
        if delegated is None:
            return None  # unscoped driver — the pre-delegation behaviour
        if not delegated:
            raise HTTPException(
                status_code=403, detail=f"empty {DELEGATION_HEADER}"
            )
        # NOT checked against the node registry. Liveness of a lease is the
        # only authority that matters, and it is strictly stronger: a node
        # can only hold one by having registered and claimed. Consulting the
        # in-memory registry too would add a second source of truth about
        # identity that a coordinator restart empties while durable leases
        # survive — refusing writes the lease table still authorizes.
        return delegated
    # Any other caller: the header is ignored entirely. Not an error — it is
    # simply not authoritative, and a volunteer must never act as another
    # machine.
    return _authenticated_node(state, token)


def _acting_node(state, request: Request) -> str:
    """The machine this request acts as, where acting as *nobody* is not a
    valid answer. Same resolution as `_write_identity` — same single header
    reader — with the unscoped-operator case turned into a refusal.

    The lease lifecycle (register / claim / heartbeat / complete / fail) is
    always an act by a specific machine: a claim assigns work to a node, a
    `fail` returns another node's task to the queue. An operator that names
    nobody has no identity to check, and quietly letting it through would
    hand the API the ability to drive *any* lease purely by omitting a
    header — the exact requeue-steal hole `_require_lease_holder` closes,
    re-opened for the credential that is easiest to forward wrongly.

    A node token is unaffected: `_write_identity` never honours the header
    for it, so it is always itself and can never reach another node's lease.
    """
    node_id = _write_identity(state, request)
    if node_id is None:
        # The pre-delegation answer, unchanged and deliberately so: an
        # operator token is not a node identity, so it 401s here exactly as
        # it did before this header existed (pinned by the write-scope
        # suite). Naming a machine is what gives it one.
        raise HTTPException(
            status_code=401,
            detail=(
                "operator credential is not a node identity — name the "
                f"machine in {DELEGATION_HEADER} to act for it"
            ),
        )
    return node_id


def _authorize_write(state, manager, request: Request, key: str) -> None:
    """Confine a write to the tasks this caller currently holds.

    Not enforcing ⇒ allow: the self-hosted profile predates credentials and
    must keep working (CLAUDE.md rule 4). When enforcing, an unknown caller is
    401 and an out-of-scope key is 403 — distinct so an operator can tell a
    misconfigured agent from a misbehaving one.

    An *operator* token passes unscoped: a driver (fedavg/K-means reducer)
    runs inside the trusted API, holds no lease, and must still write
    `jobs/{job}/round-NNN/weights.json`. That is a second credential class,
    not an exemption — the caller is still authenticated and attributable.
    An operator forwarding for a machine names it in `DELEGATION_HEADER` and
    gets that machine's scope instead.
    """
    if not state.authenticator.enforcing:
        return
    node_id = _write_identity(state, request)
    if node_id is None:
        return
    # Trailing slash is load-bearing: without it `jobs/j/trial-000extra/...`
    # would satisfy a `jobs/j/trial-000` prefix test.
    allowed = [f"jobs/{job}/{task}/" for job, task in manager.live_leases_for_node(node_id)]
    if not any(key.startswith(p) for p in allowed):
        raise HTTPException(
            status_code=403,
            detail=f"node {node_id} holds no live lease covering {key!r}",
        )


def authorize_task_write(state, manager, request: Request, job_id: str, task_id: str) -> None:
    """Checkpoint routes name (job, task) in the path, so authorize the pair
    directly instead of reconstructing a key prefix.

    Same caller resolution as `_authorize_write` — including delegation —
    because it is the same question asked about a differently-shaped key.
    """
    if not state.authenticator.enforcing:
        return
    node_id = _write_identity(state, request)
    if node_id is None:
        return
    if (job_id, task_id) not in manager.live_leases_for_node(node_id):
        raise HTTPException(
            status_code=403,
            detail=f"node {node_id} holds no live lease on {job_id}/{task_id}",
        )


def _require_lease_holder(state, manager, request: Request, lease_id: str) -> None:
    """Only the node that was issued a lease may drive its lifecycle.

    Without this, `complete`/`fail`/`heartbeat` take a lease_id and check
    nothing: an attacker fails another node's attempt over and over until the
    task requeues to *him*, then writes his poison entirely within the write
    scoping above. Scoping writes without owning the lifecycle just moves the
    hole down one layer.

    A *bare* operator token is still NOT accepted here. A driver holds no
    lease; letting an unscoped credential fail somebody's attempt would
    re-open exactly the requeue attack this closes. An operator that names a
    machine in `DELEGATION_HEADER` is a different thing: it is checked
    against *that machine's* lease, so the forwarded call is authorized
    exactly as the direct call would have been and no more.
    """
    if not state.authenticator.enforcing:
        return
    node_id = _acting_node(state, request)
    lease = manager.lease_info(lease_id)
    if lease is None or getattr(lease, "node_id", None) != node_id:
        # 403 for unknown as well as not-yours: a 404/403 split would tell an
        # attacker which lease ids exist.
        raise HTTPException(
            status_code=403, detail=f"node {node_id} does not hold lease {lease_id}"
        )


def build_router(state: ModeAState) -> APIRouter:
    router = APIRouter(prefix="/v1alpha1")
    manager = state.manager

    # -- node registry ------------------------------------------------------

    @router.post("/nodes/register")
    async def register_node(reg: NodeRegistration, request: Request):
        if state.join_code is not None:
            supplied = request.headers.get("X-FlashML-Join-Code")
            if supplied != state.join_code:
                raise HTTPException(status_code=403, detail="invalid or missing join code")
        if state.authenticator.enforcing:
            # Identity comes from the credential (or, for an operator, from
            # the machine it names) — never from the body. A node must not be
            # able to register (and so claim work) as another.
            reg = reg.model_copy(update={"node_id": _acting_node(state, request)})
        state.nodes[reg.node_id] = _NodeEntry(
            registration=reg, last_heartbeat=datetime.now(timezone.utc)
        )
        return {"node_id": reg.node_id, "status": "registered"}

    @router.post("/nodes/{node_id}/heartbeat")
    async def node_heartbeat(node_id: str, hb: NodeHeartbeat, request: Request):
        if state.authenticator.enforcing:
            caller = _acting_node(state, request)
            if caller != node_id:
                # "itself" means the asserted machine when an operator is
                # forwarding — a heartbeat keeps a node marked online, so
                # letting the API send one for an arbitrary node_id would
                # falsify the pool view.
                raise HTTPException(
                    status_code=403, detail="a node may only heartbeat itself"
                )
        entry = state.nodes.get(node_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown node {node_id} — register first")
        entry.last_heartbeat = hb.timestamp
        if hb.pools is not None:
            # Server-stamped membership refresh (cloud proxy). A list, even
            # empty, replaces the registration's pools wholesale; None means
            # no statement. Replace the capabilities object rather than
            # mutating it in place so the registration model stays the
            # single source the claim node view dumps from.
            entry.registration = entry.registration.model_copy(
                update={
                    "capabilities": entry.registration.capabilities.model_copy(
                        update={"pools": list(hb.pools)}
                    )
                }
            )
        return {"status": "ok"}

    @router.get("/nodes")
    async def list_nodes():
        return state.node_view()

    # -- leases -------------------------------------------------------------

    @router.post("/leases/claim")
    async def claim(req: ClaimRequest, request: Request):
        if state.authenticator.enforcing:
            # Overwrite rather than validate-and-reject: a body node_id is
            # simply not authoritative, so a disagreement is not an error to
            # report — there is nothing here to disagree with. That holds for
            # the forwarded case too: the operator's header decides, the body
            # is still ignored.
            req.node_id = _acting_node(state, request)
        entry = state.nodes.get(req.node_id)
        if entry is None:
            raise HTTPException(status_code=403, detail="unregistered node — register first")
        node_view = {
            "node_id": req.node_id,
            "sandbox_capable": entry.registration.sandbox_capable,
            "argv_capable": entry.registration.argv_capable,
            "module_capable": entry.registration.module_capable,
            # Names only — the host paths behind them never reach us, and the
            # placement gate needs the names to place local-data work at all.
            "local_datasets": entry.registration.local_datasets,
            "capabilities": entry.registration.capabilities.model_dump(),
        }
        lease = manager.claim(
            req.node_id,
            job_id=req.job_id,
            policy=IsolationAwarePlacement(),
            node=node_view,
        )
        if lease is None:
            return Response(status_code=204)  # nothing claimable right now
        return lease

    @router.post("/attempts/{lease_id}/heartbeat")
    async def attempt_heartbeat(lease_id: str, request: Request):
        from flashruntime.leases import LeaseError

        # Same ownership rule as complete/fail: keeping somebody else's lease
        # alive stops the sweeper from ever requeueing their stalled task.
        _require_lease_holder(state, manager, request, lease_id)
        try:
            return manager.heartbeat(lease_id)
        except LeaseError as exc:
            # 410 Gone: the worker must stop — its lease is dead.
            raise HTTPException(status_code=410, detail=str(exc))

    @router.post("/attempts/{lease_id}/complete")
    async def attempt_complete(lease_id: str, req: CompleteRequest, request: Request):
        from flashruntime.leases import LeaseError

        _require_lease_holder(state, manager, request, lease_id)
        lease = manager.lease_info(lease_id)
        if lease is None:
            raise HTTPException(status_code=404, detail=f"unknown lease {lease_id}")

        # Accepted work = validated output: the artifact at the task's
        # commit_key must exist and hash to what the worker claims. A bad
        # upload fails the attempt (task requeues elsewhere); it never
        # commits. Fault tolerance that accepts wrong results is worse than
        # failure.
        record = next(
            (r for r in manager.records(lease.job_id) if r.spec.task_id == lease.task_id), None
        )
        if record is not None and not _output_valid(
            state.artifacts_dir, record.spec.commit_key, req.output_sha256
        ):
            try:
                manager.fail(
                    lease_id, f"output validation failed for {record.spec.commit_key}"
                )
                return {"accepted": False, "detail": "output validation failed; attempt requeued"}
            except LeaseError:
                pass  # lease already dead → fall through to the late-commit rejection

        try:
            accepted = manager.complete(lease_id, output_sha256=req.output_sha256)
        except LeaseError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        if accepted and lease.node_id in state.nodes:
            # credit accepted work only — the contribution-accounting rule
            state.nodes[lease.node_id].accepted_tasks += 1
        return {"accepted": accepted}

    @router.post("/attempts/{lease_id}/fail")
    async def attempt_fail(lease_id: str, req: FailRequest, request: Request):
        from flashruntime.leases import LeaseError

        _require_lease_holder(state, manager, request, lease_id)
        try:
            manager.fail(lease_id, req.reason)
        except LeaseError as exc:
            raise HTTPException(status_code=410, detail=str(exc))
        return {"status": "requeued-or-exhausted"}

    # -- tasks view ---------------------------------------------------------

    @router.get("/jobs/{job_id}/tasks")
    async def job_tasks(job_id: str):
        out = []
        for r in manager.records(job_id):
            lease = r.active_lease
            last = None
            if r.lease_history:
                last = list(r.lease_history.values())[-1]
            out.append(
                {
                    "task_id": r.spec.task_id,
                    "state": r.state.value,
                    "attempts": r.attempts_used,
                    "max_attempts": r.spec.max_attempts,
                    "node_id": (lease or last).node_id if (lease or last) else None,
                    "deadline": lease.deadline.isoformat() if lease else None,
                }
            )
        return sorted(out, key=lambda t: t["task_id"])

    # -- local artifact hosting --------------------------------------------

    @router.put("/artifacts/{key:path}")
    async def put_artifact(key: str, request: Request):
        key = _safe_key(key)
        # Checked TWICE, on purpose. The first call rejects an unauthorized
        # caller before we buffer a body that may be hundreds of megabytes.
        _authorize_write(state, manager, request, key)
        data = await request.body()
        if len(data) > state.max_artifact_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"artifact exceeds {state.max_artifact_bytes} bytes",
            )
        # ...and the second closes the TOCTOU window between them. Reading the
        # body takes as long as the client wants it to: a node can claim a
        # task, open a chunked PUT, trickle bytes while the sweeper expires
        # its lease and another node completes and commits the task — and
        # then land its body on top of the committed result. That window is
        # attacker-controlled, not bounded by the lease duration, and it
        # defeats revocation entirely. `live_leases_for_node` stops covering
        # a completed or reclaimed task, so re-checking here is the fix.
        _authorize_write(state, manager, request, key)
        path = state.artifacts_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return ArtifactRecord(
            uri=f"artifact://{key}",
            backend="local",
            bucket=str(state.artifacts_dir),
            object_key=key,
            sha256=hashlib.sha256(data).hexdigest(),
            size_bytes=len(data),
        )

    @router.get("/artifacts/{key:path}")
    async def get_artifact(key: str):
        key = _safe_key(key)
        path = state.artifacts_dir / key
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"no artifact at {key}")
        return Response(content=path.read_bytes(), media_type="application/octet-stream")

    @router.get("/jobs/{job_id}/artifacts")
    async def job_artifacts(job_id: str):
        base = state.artifacts_dir / "jobs" / job_id
        if not base.is_dir():
            return []
        out = []
        for path in sorted(base.rglob("*")):
            if path.is_file():
                key = str(path.relative_to(state.artifacts_dir))
                out.append(
                    {
                        "uri": f"artifact://{key}",
                        "key": key,
                        "size_bytes": path.stat().st_size,
                    }
                )
        return out

    return router


def lease_job_state(manager: LeaseManager, job_id: str) -> tuple[str, dict[str, int]]:
    """Derive a JobState name from the task counts (status is never a
    hand-mutated field — it falls out of the lease table)."""
    counts = manager.job_state(job_id)
    total = sum(counts.values())
    done = counts.get(TaskState.COMPLETED.value, 0)
    failed = counts.get(TaskState.FAILED.value, 0)
    active = counts.get(TaskState.PENDING.value, 0) + counts.get(TaskState.LEASED.value, 0)
    if total == 0:
        return "PENDING", counts
    if active > 0:
        return "RUNNING", counts
    if failed > 0:
        return "FAILED", counts
    if done == total:
        return "SUCCEEDED", counts
    return "CANCELLED", counts
