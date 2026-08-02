"""Federated averaging as a sequence of lease jobs.

One round = one Mode A job (N independent shard tasks); the driver reduces
the shard deltas into new weights and submits the next round. Same
stage-composition pattern as `kmeans_driver` — "pipelines are jobs chained
by a driver, not a new execution mode" — so a dead worker costs one shard
retry and a dead driver resumes from the last completed round.

The one deliberate difference from kmeans_driver: it required *every*
shard (`if len(partials) != len(shard_uris): raise`). This driver
aggregates on a QUORUM. Volunteer machines are unequal and unreliable by
definition; requiring all of them would let one closed laptop stall every
participant's round. Deltas arriving after aggregation are DISCARDED, never
carried into a later round — they were computed against weights that no
longer exist, and applying them would silently corrupt the average.

Pure stdlib: this runs inside the cloud API, which must not carry torch.
"""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable, Protocol, Sequence, TypedDict

from flashml_workloads.fedavg_weights import (
    apply_delta,
    reduce_deltas,
    require_finite,
)

_SAFE_DELTA_FILE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

#: Default container image for a round's tasks. Overridable per run — see
#: `run_fedavg(image=...)`.
DEFAULT_IMAGE = "local/tier1:dev"

__all__ = ["ArtifactNotFound", "BuildRound", "Coordinator",
           "CoordinatorUnavailable", "DEFAULT_IMAGE", "HttpCoordinator",
           "QuorumNotMet", "RoundPlan", "RoundResult", "resume_state",
           "run_fedavg"]


class QuorumNotMet(RuntimeError):
    """A round's deadline passed with too few committed shards."""


class CoordinatorUnavailable(RuntimeError):
    """A poll-loop coordinator call kept failing after bounded retries.

    Distinct from `QuorumNotMet`: the round did not run out of participants,
    the driver ran out of coordinator. Bounded on purpose — retrying forever
    would turn an outage into a run that never ends and never reports.
    """


class ArtifactNotFound(LookupError):
    """No artifact exists at that key.

    A named exception, not a bare `Exception` catch: `resume_state` must
    distinguish "this round never completed" (expected, keep looking) from
    "the coordinator is unreachable" (fatal, must not look like round 0).
    """


class RoundResult(TypedDict):
    round: int
    participants: int
    mean_loss: float
    job_id: str


class RoundPlan(TypedDict):
    """What one round is: the job body to submit, and the task ids it will
    produce.

    ``task_ids`` is carried rather than derived because the two things the
    driver needs from a round — "submit this" and "look for these commits" —
    are decided by whoever built the body. The built-in body expands via
    ``service/modea._expand_fedavg`` (``shard-000``, ``shard-001``, …); a
    caller compiling the round as a ``command`` workload gets
    ``CommandRecipe``'s ``task-000``, ``task-001``, … instead. Inferring the
    prefix from the workload type would be a second place that has to know
    every recipe's naming rule, and the artifact key filter is a security
    boundary here (see ``_committed_metrics_keys``) — so it is stated, not
    guessed.
    """

    body: dict
    task_ids: list[str]


#: Build the ``RoundPlan`` for round ``r`` given the round's weights URI
#: (``None`` on round 0, when nothing has been aggregated yet).
BuildRound = Callable[[int, "str | None"], RoundPlan]


class Coordinator(Protocol):
    """The coordinator operations the driver needs.

    Declared as a Protocol so tests substitute a fake without HTTP, and so
    the cloud API can pass an implementation that adds auth headers.
    """

    def submit(self, body: dict) -> dict: ...
    def job_state(self, job_id: str) -> str: ...
    def artifacts(self, job_id: str) -> list[dict]: ...
    def get_artifact(self, key: str) -> Any: ...
    def put_artifact(self, key: str, body: Any) -> None: ...

    # Optional. When present it is the authoritative participant count:
    # artifacts prove only that *something was uploaded*, tasks prove that
    # the coordinator ACCEPTED a commit. A Coordinator without it still
    # works (the expected-key filter alone bounds the count at num_shards),
    # so it is probed with getattr rather than being a hard requirement.
    def tasks(self, job_id: str) -> list[dict]: ...


def _round_body(round_idx: int, num_shards: int, worker_params: dict,
                weights_uri: str | None, lease_seconds: float,
                image: str, isolation_tier: str, allow_fallback: bool) -> dict:
    params: dict[str, Any] = dict(worker_params)
    params.update({"round": round_idx, "num_shards": num_shards,
                   "lease_seconds": lease_seconds})
    if weights_uri is not None:
        params["weights"] = weights_uri
    repository, _, tag = image.rpartition(":")
    if not repository or not tag:
        raise ValueError(
            f"image must be 'repository:tag' with a pinned tag, got {image!r}"
        )
    return {
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": f"fedavg-r{round_idx:03d}"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": repository, "tag": tag},
            "isolation": {"tier": isolation_tier, "allowFallback": allow_fallback},
            "workload": {"type": "federated_averaging", "parameters": params},
        },
    }


def _default_task_ids(num_shards: int) -> list[str]:
    """Task ids `service/modea._expand_fedavg` produces for a round."""
    return [f"shard-{i:03d}" for i in range(num_shards)]


def _expected_metrics_keys(job_id: str, task_ids: Sequence[str]) -> dict[str, str]:
    """`{artifact key: task_id}` for exactly the tasks this round dispatched.

    Every expansion anchors a task's commit at
    `jobs/{job_id}/{task_id}/metrics.json`; which task ids exist is the
    round's `RoundPlan.task_ids`.
    """
    return {f"jobs/{job_id}/{task_id}/metrics.json": task_id
            for task_id in task_ids}


def _committed_metrics_keys(coord: Coordinator, job_id: str,
                            task_ids: Sequence[str]) -> list[str]:
    """Keys of the round's committed metrics.json artifacts.

    Two filters, because a participant count is a security boundary here —
    it decides how much weight one machine gets in the average, and a
    volunteer that mints extra "participants" both dilutes everyone else
    and inflates its own share.

    1. An EXACT match against the round's expected task set, never a
       `endswith("metrics.json")` suffix test. The agent uploads a task's
       whole output tree recursively, so a worker that writes
       `out/a/metrics.json` and `out/b/metrics.json` would otherwise mint
       two extra participants out of a single lease — and a key naming a
       task id outside the round's own set would mint one out of nothing.
    2. Cross-checked against the coordinator's task states when the
       Coordinator exposes them. Artifact PUTs happen BEFORE the commit is
       offered, so an attempt the coordinator went on to REJECT (lost
       lease, sha256 mismatch, attempts exhausted) still leaves its
       metrics.json sitting in the bucket. Only a task the coordinator
       reports COMPLETED had its commit accepted.

    Cheap: one or two listing calls. Kept separate from `_fetch` so the
    quorum poll does not re-download every delta on every tick — deltas are
    megabytes, and polling re-fetching them would dominate the round's
    transfer cost.
    """
    expected = _expected_metrics_keys(job_id, task_ids)
    present = {a["key"] for a in coord.artifacts(job_id)} & expected.keys()

    list_tasks = getattr(coord, "tasks", None)
    if list_tasks is not None:
        completed = {t["task_id"] for t in list_tasks(job_id)
                     if t.get("state") == "COMPLETED"}
        present = {k for k in present if expected[k] in completed}
    return sorted(present)


def _safe_delta_key(metrics_key: str, delta_file: str) -> str:
    """Resolve a task's declared delta filename inside its own output prefix.

    `delta_file` comes from metrics.json, which is written by an UNTRUSTED
    volunteer node. Without this check a malicious node could name
    `../../other-job/weights.json` and make the driver read — and average
    in — an artifact belonging to somebody else's job. Result verification
    is M3; this is not that, it is basic path containment and belongs here.

    This is an ALLOWLIST, not a denylist: only a plain filename made of
    ASCII letters/digits/`._-`, not starting with `.`, passes. A denylist of
    specific bad substrings (`/`, `\\`, `..`) would still let a URL-encoded
    `%2F`, a leading `~`, or an embedded NUL through — this function's
    docstring claims it *is* the containment layer, so it must not be a
    partial list of things we happened to think of.
    """
    if delta_file in (".", "..") or not _SAFE_DELTA_FILE.match(delta_file):
        raise ValueError(
            f"task declared an unsafe delta_file {delta_file!r}: "
            "must be a plain filename in the task's own output prefix"
        )
    return metrics_key.rsplit("/", 1)[0] + "/" + delta_file


def _fetch(coord: Coordinator, metrics_keys: list[str]) -> list[tuple[dict, int, float]]:
    """Download (delta, samples, loss) for the keys that met quorum."""
    out = []
    for key in metrics_keys:
        metrics = coord.get_artifact(key)
        delta_key = _safe_delta_key(key, metrics.get("delta_file", "delta.json"))
        out.append((coord.get_artifact(delta_key),
                    int(metrics["samples"]), float(metrics["loss"])))
    return out


class HttpCoordinator:
    """`Coordinator` over the coordinator's HTTP API.

    `headers` carries the caller's credentials — the cloud API passes the
    machine/service token here rather than the driver knowing anything
    about auth.
    """

    def __init__(self, base_url: str, headers: dict[str, str] | None = None):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})

    def _request(self, method: str, url: str, data: bytes | None = None,
                 headers: dict | None = None, timeout: float | None = 60.0):
        req = urllib.request.Request(url, data=data, method=method)
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        if data is not None:
            req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def submit(self, body: dict) -> dict:
        return self._request("POST", f"{self.base_url}/v1alpha1/jobs",
                             data=json.dumps(body).encode(), headers=self.headers)

    def job_state(self, job_id: str) -> str:
        job_id_q = urllib.parse.quote(job_id, safe="/")
        return self._request("GET", f"{self.base_url}/v1alpha1/jobs/{job_id_q}",
                             headers=self.headers)["state"]

    def artifacts(self, job_id: str) -> list[dict]:
        job_id_q = urllib.parse.quote(job_id, safe="/")
        return self._request("GET", f"{self.base_url}/v1alpha1/jobs/{job_id_q}/artifacts",
                             headers=self.headers)

    def tasks(self, job_id: str) -> list[dict]:
        """`[{"task_id": ..., "state": "COMPLETED"|...}, ...]` for the job.

        The coordinator's task view (`GET /v1alpha1/jobs/{id}/tasks`) is the
        only place that knows whether a commit was ACCEPTED; the artifact
        listing only knows something was uploaded.
        """
        job_id_q = urllib.parse.quote(job_id, safe="/")
        return self._request("GET", f"{self.base_url}/v1alpha1/jobs/{job_id_q}/tasks",
                             headers=self.headers)

    def get_artifact(self, key: str):
        key_q = urllib.parse.quote(key, safe="/")
        try:
            return self._request("GET", f"{self.base_url}/v1alpha1/artifacts/{key_q}",
                                 headers=self.headers)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                raise ArtifactNotFound(key) from None
            raise   # 5xx / auth failures are NOT "round never completed"

    def put_artifact(self, key: str, body) -> None:
        key_q = urllib.parse.quote(key, safe="/")
        self._request("PUT", f"{self.base_url}/v1alpha1/artifacts/{key_q}",
                      data=json.dumps(body).encode(), headers=self.headers)


def resume_state(coord: Coordinator,
                 job_ids: Sequence[tuple[int, str]]) -> tuple[int, dict, str | None]:
    """Where to restart after a driver crash.

    `job_ids` is a sequence of `(round, job_id)` pairs — the round number is
    CARRIED, never inferred from the list position. Position-as-round is
    only true for a run that started at round 0: after resuming at round 5
    the list holds round 5 at index 0, so a second crash would probe
    `round-000` under the round-5 job, get `ArtifactNotFound`, and silently
    restart training from scratch. The tuple removes the ambiguity rather
    than documenting it.

    Rounds are idempotent: the weights artifact is written only AFTER a
    round aggregates, so the newest one that exists names the last round
    that fully completed.

    Only ArtifactNotFound is swallowed. A transport error must propagate:
    silently treating an unreachable coordinator as "no rounds done" would
    restart a finished run from scratch.

    An artifact that EXISTS (no ArtifactNotFound) but is falsy — `{}`, or
    `None` from an empty-body 200 — is a different situation from "this
    round never completed": something committed a weights key with no
    usable content. Treating that identically to "keep searching" would
    silently walk past a corrupt commit and redo already-completed work
    (or worse, resume from stale weights further back). Surface it instead
    of guessing.
    """
    pairs: list[tuple[int, str]] = []
    for entry in job_ids:
        if isinstance(entry, str) or len(entry) != 2:
            raise TypeError(
                "resume_state expects (round, job_id) pairs, got "
                f"{entry!r}; a bare job-id list re-introduces the "
                "position-is-the-round bug that breaks on a second resume"
            )
        pairs.append((int(entry[0]), str(entry[1])))

    for r, job_id in sorted(pairs, reverse=True):
        key = f"jobs/{job_id}/round-{r:03d}/weights.json"
        try:
            weights = coord.get_artifact(key)
        except ArtifactNotFound:
            continue
        if not weights:
            raise RuntimeError(
                f"weights artifact at {key!r} exists but is empty; "
                "cannot distinguish a corrupt commit from a round that "
                "never completed"
            )
        # This is the READ path: the weights artifact was written by a PUT
        # that is currently unauthenticated, so a corrupted or
        # attacker-written weights.json must not be handed back to the
        # caller un-gated. Without this, a NaN here resumes training from
        # NaN and the run still reports success.
        require_finite(weights, f"resume_state: weights artifact {key!r}")
        return r + 1, weights, f"artifact://{key}"

    if pairs and min(r for r, _ in pairs) > 0:
        # Every carried job is from a resumed run and none of them
        # aggregated, so rounds before the earliest carried one may well
        # have completed under jobs this list does not mention. "Start over
        # from round 0" would throw that work away silently; say so instead.
        raise RuntimeError(
            "no completed round found, but the carried history starts at "
            f"round {min(r for r, _ in pairs)}: earlier rounds ran under job "
            "ids not present here, so 'restart from scratch' cannot be "
            "concluded. Pass the full (round, job_id) history."
        )
    return 0, {}, None


def _retrying(call: Callable[[], Any], *, what: str, deadline: float,
              attempts: int, backoff_s: float) -> Any:
    """Call `call()`, retrying transient failures with capped backoff.

    A multi-round run is hours long; one blip from the coordinator (a
    restart, a dropped connection, a 502 from a proxy) must not end it —
    flashnode's own executor loop backs off and keeps going, and a driver
    that does not is the weakest link in the pair. Bounded, though: after
    `attempts` tries it fails with context rather than looping forever, and
    it never sleeps past the round deadline.
    """
    last: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001 - re-raised with context below
            last = exc
            remaining = deadline - time.monotonic()
            if attempt == attempts or remaining <= 0:
                break
            time.sleep(min(backoff_s * (2 ** (attempt - 1)), 5.0, remaining))
    raise CoordinatorUnavailable(
        f"{what}: coordinator call failed after {attempt} attempt(s): {last!r}"
    ) from last


def run_fedavg(
    coord: Coordinator,
    *,
    rounds: int,
    num_shards: int,
    min_participants: int,
    worker_params: dict,
    initial_weights: dict,
    round_timeout_s: float = 600.0,
    poll_seconds: float = 1.0,
    lease_seconds: float = 120.0,
    on_round: Callable[[RoundResult], None] | None = None,
    start_round: int = 0,
    weights_uri: str | None = None,
    image: str = DEFAULT_IMAGE,
    isolation_tier: str = "standard",
    allow_fallback: bool = False,
    poll_attempts: int = 4,
    poll_backoff_s: float = 0.5,
    prior_job_ids: Sequence[tuple[int, str]] | None = None,
    build_round: BuildRound | None = None,
) -> dict:
    """Drive `rounds` federated-averaging rounds and return the final weights.

    `image` and `isolation_tier` are caller-settable rather than hardcoded:
    the default `local/tier1:dev` exists only in this repo's e2e fixtures,
    and it is inert today only because `SubprocessRunner` ignores `image`
    entirely. The moment a round is served by a docker-tier volunteer, a
    hardcoded image is an unpullable reference on somebody else's machine —
    the same "two places, each correct in isolation" shape as the task-module
    allowlist drift that already caused an outage here.

    `build_round` replaces how a round becomes a job. The default builds the
    built-in `federated_averaging` body, whose tasks run
    `flashml_workloads.fedavg_worker`. A caller that wants the *user's own*
    code to be the round worker — the cloud API compiling a repo's
    entrypoint into a `command` job per round — passes its own builder
    instead; everything downstream (quorum, reduce, weights artifact,
    resume) is unchanged, because none of it depends on what ran inside the
    round, only on the task ids it produced and the `metrics.json` /
    `delta.json` pair each one committed. `worker_params`, `image`,
    `isolation_tier`, `allow_fallback` and `lease_seconds` are inputs to the
    *default* builder and are ignored when `build_round` is supplied — the
    builder already knows all of it.

    `initial_weights` may be `{}`, and that is not the same as "start from
    zeros": it means the driver holds no weights yet, so round 0's reduced
    contribution IS the first set of weights rather than a delta applied to
    something. That is the only reading consistent with the worker contract
    ("`delta.json` is the change from the weights you were given") when a
    worker was given no weights — and it is the case that matters for
    arbitrary user code, where the API cannot construct the model to
    initialise from. Passing a real `initial_weights` (as
    `fedavg_worker`-based callers do, seeding from the model) keeps the
    previous behaviour exactly.

    Returns `{"weights", "history", "job_ids"}` where `job_ids` is a list of
    `(round, job_id)` pairs suitable for feeding straight back into
    `resume_state` (and into `prior_job_ids` on the next resume).
    """
    if min_participants < 1:
        raise ValueError("min_participants must be >= 1")
    if min_participants > num_shards:
        raise ValueError(
            f"min_participants {min_participants} exceeds num_shards {num_shards}"
        )
    if poll_attempts < 1:
        raise ValueError("poll_attempts must be >= 1")

    weights = initial_weights
    history: list[RoundResult] = []
    # (round, job_id), never a bare list whose position implies the round:
    # a resumed run's first entry is round `start_round`, not round 0.
    job_ids: list[tuple[int, str]] = [(int(r), str(j))
                                      for r, j in (prior_job_ids or [])]

    for r in range(start_round, rounds):
        if build_round is None:
            plan: RoundPlan = {
                "body": _round_body(r, num_shards, worker_params, weights_uri,
                                    lease_seconds, image, isolation_tier,
                                    allow_fallback),
                "task_ids": _default_task_ids(num_shards),
            }
        else:
            plan = build_round(r, weights_uri)
        task_ids = list(plan["task_ids"])
        if len(task_ids) != len(set(task_ids)):
            # Duplicate ids would collapse two expected keys into one and
            # silently lower the achievable participant count below
            # min_participants — a round that can never reach quorum.
            raise ValueError(
                f"round {r}: build_round returned duplicate task ids {task_ids!r}"
            )
        if len(task_ids) < min_participants:
            raise ValueError(
                f"round {r}: build_round returned {len(task_ids)} task(s), "
                f"fewer than min_participants {min_participants} — quorum "
                "could never be reached"
            )
        job_id = coord.submit(plan["body"])["job_id"]
        job_ids.append((r, job_id))

        deadline = time.monotonic() + round_timeout_s
        keys: list[str] = []
        while True:
            keys = _retrying(
                lambda: _committed_metrics_keys(coord, job_id, task_ids),
                what=f"round {r}: listing committed shards",
                deadline=deadline, attempts=poll_attempts,
                backoff_s=poll_backoff_s,
            )
            if len(keys) >= min_participants:
                break
            state = _retrying(
                lambda: coord.job_state(job_id),
                what=f"round {r}: reading job state",
                deadline=deadline, attempts=poll_attempts,
                backoff_s=poll_backoff_s,
            )
            if state in ("FAILED", "CANCELLED"):
                raise QuorumNotMet(
                    f"round {r}: job {job_id} ended {state} with "
                    f"{len(keys)} of {min_participants} needed"
                )
            if time.monotonic() > deadline:
                raise QuorumNotMet(
                    f"round {r}: timed out with {len(keys)} of "
                    f"{min_participants} needed ({len(task_ids)} shards dispatched)"
                )
            # Clamp to the time remaining: if round_timeout_s < poll_seconds
            # a full un-clamped sleep would overrun the deadline by up to
            # one poll tick before the loop gets a chance to re-check it.
            time.sleep(min(poll_seconds, max(0.0, deadline - time.monotonic())))

        # Freeze the participant set at the moment quorum was reached, then
        # download. Anything committing from here on is discarded by
        # construction: we never re-read this job after aggregating.
        collected = _fetch(coord, keys)
        reduced = reduce_deltas([(d, n) for d, n, _ in collected])
        # No weights yet (`initial_weights={}` and nothing aggregated): the
        # round's workers were handed nothing, so what they reported as
        # "the change from what you were given" is the weights themselves.
        # `apply_delta` would refuse here — an empty base and a populated
        # delta are, correctly, not the same parameter set.
        # `require_finite` on the bootstrap branch because `apply_delta` —
        # the only other way out of here — checks its own result, and the
        # weighted sum of finite contributions can still overflow to inf.
        weights = (require_finite(reduced, f"round {r}: bootstrap weights")
                   if not weights else apply_delta(weights, reduced))

        weights_key = f"jobs/{job_id}/round-{r:03d}/weights.json"
        coord.put_artifact(weights_key, weights)
        weights_uri = f"artifact://{weights_key}"

        # Sample-weighted, consistent with the delta reduce: an unweighted
        # mean would let a low-sample straggler with high loss skew the
        # reported metric out of proportion to its actual contribution to
        # the aggregate weights.
        total_n = sum(n for _, n, _ in collected)
        result: RoundResult = {
            "round": r,
            "participants": len(collected),
            "mean_loss": sum(loss * n for _, n, loss in collected) / total_n,
            "job_id": job_id,
        }
        history.append(result)
        if on_round is not None:
            on_round(result)

    return {"weights": weights, "history": history, "job_ids": job_ids}
