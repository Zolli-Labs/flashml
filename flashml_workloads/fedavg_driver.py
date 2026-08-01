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

import time
from typing import Any, Callable, Protocol, TypedDict

from flashml_workloads.fedavg_weights import apply_delta, reduce_deltas

__all__ = ["ArtifactNotFound", "Coordinator", "QuorumNotMet", "RoundResult",
           "run_fedavg"]


class QuorumNotMet(RuntimeError):
    """A round's deadline passed with too few committed shards."""


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


def _round_body(round_idx: int, num_shards: int, worker_params: dict,
                weights_uri: str | None, lease_seconds: float) -> dict:
    params: dict[str, Any] = dict(worker_params)
    params.update({"round": round_idx, "num_shards": num_shards,
                   "lease_seconds": lease_seconds})
    if weights_uri is not None:
        params["weights"] = weights_uri
    return {
        "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
        "metadata": {"name": f"fedavg-r{round_idx:03d}"},
        "spec": {
            "execution": {"backend": "leases"},
            "image": {"repository": "local/tier1", "tag": "dev"},
            "workload": {"type": "federated_averaging", "parameters": params},
        },
    }


def _committed_metrics_keys(coord: Coordinator, job_id: str) -> list[str]:
    """Keys of the round's committed metrics.json artifacts.

    Cheap: one listing call. Kept separate from `_fetch` so the quorum poll
    does not re-download every delta on every tick — deltas are megabytes,
    and polling re-fetching them would dominate the round's transfer cost.
    """
    return sorted(a["key"] for a in coord.artifacts(job_id)
                  if a["key"].endswith("metrics.json"))


def _safe_delta_key(metrics_key: str, delta_file: str) -> str:
    """Resolve a task's declared delta filename inside its own output prefix.

    `delta_file` comes from metrics.json, which is written by an UNTRUSTED
    volunteer node. Without this check a malicious node could name
    `../../other-job/weights.json` and make the driver read — and average
    in — an artifact belonging to somebody else's job. Result verification
    is M3; this is not that, it is basic path containment and belongs here.
    """
    if "/" in delta_file or "\\" in delta_file or delta_file in ("", ".", ".."):
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
) -> dict:
    if min_participants < 1:
        raise ValueError("min_participants must be >= 1")
    if min_participants > num_shards:
        raise ValueError(
            f"min_participants {min_participants} exceeds num_shards {num_shards}"
        )

    weights = initial_weights
    weights_uri: str | None = None
    history: list[RoundResult] = []
    job_ids: list[str] = []

    for r in range(rounds):
        job_id = coord.submit(
            _round_body(r, num_shards, worker_params, weights_uri, lease_seconds)
        )["job_id"]
        job_ids.append(job_id)

        deadline = time.monotonic() + round_timeout_s
        keys: list[str] = []
        while True:
            keys = _committed_metrics_keys(coord, job_id)
            if len(keys) >= min_participants:
                break
            state = coord.job_state(job_id)
            if state in ("FAILED", "CANCELLED"):
                raise QuorumNotMet(
                    f"round {r}: job {job_id} ended {state} with "
                    f"{len(keys)} of {min_participants} needed"
                )
            if time.monotonic() > deadline:
                raise QuorumNotMet(
                    f"round {r}: timed out with {len(keys)} of "
                    f"{min_participants} needed ({num_shards} shards dispatched)"
                )
            time.sleep(poll_seconds)

        # Freeze the participant set at the moment quorum was reached, then
        # download. Anything committing from here on is discarded by
        # construction: we never re-read this job after aggregating.
        collected = _fetch(coord, keys)
        reduced = reduce_deltas([(d, n) for d, n, _ in collected])
        weights = apply_delta(weights, reduced)

        weights_key = f"jobs/{job_id}/round-{r:03d}/weights.json"
        coord.put_artifact(weights_key, weights)
        weights_uri = f"artifact://{weights_key}"

        result: RoundResult = {
            "round": r,
            "participants": len(collected),
            "mean_loss": sum(loss for _, _, loss in collected) / len(collected),
            "job_id": job_id,
        }
        history.append(result)
        if on_round is not None:
            on_round(result)

    return {"weights": weights, "history": history, "job_ids": job_ids}
