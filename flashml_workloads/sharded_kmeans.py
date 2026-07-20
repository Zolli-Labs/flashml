"""Iterative sharded K-Means on Ray tasks — the POC's real distributed job.

Shape (per docs/SYSTEM_OVERVIEW and the POC brief):
- deterministic synthetic dataset, generated per shard from (seed, shard_id)
  so shards are reproducible anywhere without shipping data;
- each iteration maps one retriable Ray task per shard (many more shards
  than workers → dynamic scheduling), reduces partial sums on the driver,
  updates centroids;
- every task reports which Ray node / Kubernetes node / pod it ran on;
- retry evidence is *measured*, not fabricated: after the run the driver
  queries Ray's task state API for tasks whose attempt count exceeds 1 and
  correlates in-run worker-set changes;
- final artifacts (centroids, metrics, execution summary, node
  contributions, recovery events) are uploaded to the configured artifact
  store (MinIO locally, OSS on Alibaba) — durable outside any worker.

Parameters (FLASHML_WORKLOAD_PARAMS JSON): samples, dimensions, clusters,
shards, iterations, seed, task_delay_seconds.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# `ray` is imported lazily inside the functions that need it: the pure data
# functions (make_shard, _true_centers) must stay importable in environments
# without Ray (unit tests, the runtime service image).


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _params() -> dict:
    p = json.loads(os.environ.get("FLASHML_WORKLOAD_PARAMS", "{}"))
    return {
        "samples": int(p.get("samples", 150_000)),
        "dimensions": int(p.get("dimensions", 24)),
        "clusters": int(p.get("clusters", 6)),
        "shards": int(p.get("shards", 36)),
        "iterations": int(p.get("iterations", 12)),
        "seed": int(p.get("seed", 42)),
        "task_delay_seconds": float(p.get("task_delay_seconds", 0.4)),
    }


def _true_centers(seed: int, clusters: int, dimensions: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(-10, 10, size=(clusters, dimensions))


def make_shard(seed: int, shard_id: int, samples_per_shard: int,
               clusters: int, dimensions: int) -> np.ndarray:
    """Deterministic shard: same (seed, shard_id) always yields the same
    points, so a retried task recomputes identical data."""
    centers = _true_centers(seed, clusters, dimensions)
    rng = np.random.default_rng(seed * 1_000_003 + shard_id)
    assignment = rng.integers(0, clusters, size=samples_per_shard)
    noise = rng.normal(0.0, 1.5, size=(samples_per_shard, dimensions))
    return centers[assignment] + noise


def assign_shard(shard_id: int, centroids: np.ndarray, cfg: dict) -> dict:
    """One map task: assign a shard's points to the current centroids and
    return partial sums/counts plus the identity of the node that did it."""
    import ray

    t0 = time.monotonic()
    X = make_shard(cfg["seed"], shard_id, cfg["samples_per_shard"],
                   cfg["clusters"], cfg["dimensions"])
    if cfg["task_delay_seconds"] > 0:
        # Only to stretch the run long enough to inject a failure reliably.
        time.sleep(cfg["task_delay_seconds"])

    # Nearest-centroid assignment.
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    inertia = float(d2[np.arange(len(X)), labels].sum())

    k = centroids.shape[0]
    sums = np.zeros_like(centroids)
    counts = np.zeros(k, dtype=np.int64)
    for c in range(k):
        mask = labels == c
        counts[c] = int(mask.sum())
        if counts[c]:
            sums[c] = X[mask].sum(axis=0)

    ctx = ray.get_runtime_context()
    return {
        "shard_id": shard_id,
        "partial_sums": sums,
        "partial_counts": counts,
        "inertia": inertia,
        "n_samples": int(len(X)),
        "duration_ms": int((time.monotonic() - t0) * 1000),
        "ray_node_id": ctx.get_node_id(),
        "pod": os.environ.get("K8S_POD_NAME", socket.gethostname()),
        "k8s_node": os.environ.get("K8S_NODE_NAME", "unknown"),
    }


def _worker_set() -> dict[str, str]:
    """Alive Ray nodes (node_id -> node name/ip), head included."""
    import ray

    return {n["NodeID"]: n.get("NodeManagerHostname", n.get("NodeManagerAddress", "?"))
            for n in ray.nodes() if n["Alive"]}


def _collect_retry_evidence(job_id: str) -> list[dict]:
    """Real per-task attempt metadata from Ray's state API. A task with
    attempt_number > 0 was genuinely retried by Ray."""
    events: list[dict] = []
    try:
        from ray.util.state import list_tasks

        tasks = list_tasks(filters=[("name", "=", "assign_shard")],
                           limit=10_000, detail=True)
        for t in tasks:
            attempt = getattr(t, "attempt_number", 0) or 0
            if attempt > 0:
                events.append({
                    "schema_version": "v1alpha1",
                    "job_id": job_id,
                    "type": "TASK_ATTEMPT_RETRIED",
                    "timestamp": _now_iso(),
                    "source": "ray.task-state",
                    "message": f"task {t.task_id} retried (attempt {attempt + 1})",
                    "data": {
                        "task_id": t.task_id,
                        "attempt_number": attempt,
                        "state": str(getattr(t, "state", "")),
                        "node_id": getattr(t, "node_id", None),
                        "error_type": str(getattr(t, "error_type", "") or ""),
                    },
                })
    except Exception as exc:  # state API is best-effort; never fake evidence
        print(json.dumps({"msg": "could not query Ray task state API",
                          "error": str(exc)}))
    return events


def run() -> None:
    import ray

    job_id = os.environ.get("FLASHML_JOB_ID", "local-dev")
    params = _params()
    max_attempts = int(os.environ.get("FLASHML_MAX_TASK_ATTEMPTS", "4"))

    ray.init()  # inside the RayJob-managed cluster

    cfg = {
        "seed": params["seed"],
        "clusters": params["clusters"],
        "dimensions": params["dimensions"],
        "samples_per_shard": max(1, params["samples"] // params["shards"]),
        "task_delay_seconds": params["task_delay_seconds"],
    }

    # max_retries counts *re*-executions: attempts = 1 + max_retries.
    remote_assign = ray.remote(max_retries=max_attempts - 1, num_cpus=1)(assign_shard)

    rng = np.random.default_rng(params["seed"])
    centers = _true_centers(params["seed"], params["clusters"], params["dimensions"])
    centroids = centers + rng.normal(0, 4.0, size=centers.shape)

    print(json.dumps({"msg": "kmeans starting", "job_id": job_id, **params}))
    initial_workers = _worker_set()
    print(json.dumps({"msg": "ray nodes online", "nodes": list(initial_workers.values()),
                      "count": len(initial_workers)}))

    metrics_history: list[dict] = []
    recovery_events: list[dict] = []
    node_contributions: dict[str, dict] = {}
    seen_workers = dict(initial_workers)
    t_start = time.monotonic()

    for iteration in range(params["iterations"]):
        it0 = time.monotonic()
        refs = [remote_assign.remote(s, centroids, cfg) for s in range(params["shards"])]
        results = ray.get(refs)

        # Reduce.
        sums = np.zeros_like(centroids)
        counts = np.zeros(params["clusters"], dtype=np.int64)
        inertia = 0.0
        for r in results:
            sums += r["partial_sums"]
            counts += r["partial_counts"]
            inertia += r["inertia"]
            node_key = f'{r["k8s_node"]}|{r["pod"]}'
            entry = node_contributions.setdefault(node_key, {
                "k8s_node": r["k8s_node"], "pod": r["pod"],
                "ray_node_id": r["ray_node_id"], "tasks": 0, "samples": 0,
                "busy_ms": 0,
            })
            entry["tasks"] += 1
            entry["samples"] += r["n_samples"]
            entry["busy_ms"] += r["duration_ms"]

        nonzero = counts > 0
        old = centroids.copy()
        centroids[nonzero] = sums[nonzero] / counts[nonzero, None]
        movement = float(np.linalg.norm(centroids - old))

        # Worker-set diffs are observed facts from ray.nodes().
        current_workers = _worker_set()
        for node_id, name in seen_workers.items():
            if node_id not in current_workers:
                recovery_events.append({
                    "schema_version": "v1alpha1", "job_id": job_id,
                    "type": "RAY_WORKER_LOST", "timestamp": _now_iso(),
                    "source": "ray.nodes",
                    "message": f"Ray node {name} left the cluster during iteration {iteration}",
                    "data": {"ray_node_id": node_id, "node": name,
                             "iteration": iteration},
                })
        for node_id, name in current_workers.items():
            if node_id not in seen_workers:
                recovery_events.append({
                    "schema_version": "v1alpha1", "job_id": job_id,
                    "type": "RAY_WORKER_REPLACED", "timestamp": _now_iso(),
                    "source": "ray.nodes",
                    "message": f"Ray node {name} joined during iteration {iteration}",
                    "data": {"ray_node_id": node_id, "node": name,
                             "iteration": iteration},
                })
        seen_workers = current_workers

        entry = {
            "iteration": iteration,
            "inertia": inertia,
            "movement": movement,
            "duration_s": round(time.monotonic() - it0, 3),
            "tasks": params["shards"],
            "workers_online": len(current_workers),
            "timestamp": _now_iso(),
        }
        metrics_history.append(entry)
        print(json.dumps({"msg": "iteration complete", **entry}))
        recovery_events.append({
            "schema_version": "v1alpha1", "job_id": job_id,
            "type": "ITERATION_COMPLETED", "timestamp": _now_iso(),
            "source": "workload.driver",
            "message": f"iteration {iteration} complete "
                       f"(inertia={inertia:.1f}, movement={movement:.4f})",
            "data": entry,
        })

    total_s = round(time.monotonic() - t_start, 2)
    recovery_events.extend(_collect_retry_evidence(job_id))

    summary = {
        "job_id": job_id,
        "parameters": params,
        "total_duration_s": total_s,
        "iterations_run": len(metrics_history),
        "final_inertia": metrics_history[-1]["inertia"],
        "final_movement": metrics_history[-1]["movement"],
        "tasks_total": params["shards"] * params["iterations"],
        "max_task_attempts": max_attempts,
        "retried_tasks": sum(1 for e in recovery_events
                             if e["type"] == "TASK_ATTEMPT_RETRIED"),
        "workers_lost": sum(1 for e in recovery_events
                            if e["type"] == "RAY_WORKER_LOST"),
        "nodes_contributing": len(node_contributions),
        "finished_at": _now_iso(),
    }
    print(json.dumps({"msg": "kmeans finished", **summary}))

    _upload_artifacts(job_id, {
        "centroids.json": {"centroids": centroids.tolist(),
                           "clusters": params["clusters"],
                           "dimensions": params["dimensions"]},
        "metrics.json": {"history": metrics_history},
        "execution-summary.json": summary,
        "node-contributions.json": {"nodes": list(node_contributions.values())},
        "recovery-events.json": recovery_events,
    })


def _upload_artifacts(job_id: str, artifacts: dict[str, object]) -> None:
    prefix = os.environ.get("FLASHML_ARTIFACT_PREFIX", f"artifact://jobs/{job_id}/")
    try:
        from flashruntime.artifacts import artifact_uri_to_key, store_from_env

        store = store_from_env()
    except Exception as exc:
        print(json.dumps({"msg": "artifact store unavailable; writing locally",
                          "error": str(exc)}))
        out = Path("/tmp/flashml-artifacts") / job_id
        out.mkdir(parents=True, exist_ok=True)
        for name, payload in artifacts.items():
            (out / name).write_text(json.dumps(payload, indent=2))
        return

    key_prefix = artifact_uri_to_key(prefix)

    async def _put_all():
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            for name, payload in artifacts.items():
                path = Path(tmp) / name
                path.write_text(json.dumps(payload, indent=2))
                record = await store.put_file(path, f"{key_prefix}{name}")
                print(json.dumps({"msg": "artifact committed", "uri": record.uri,
                                  "etag": record.etag,
                                  "size_bytes": record.size_bytes}))

    asyncio.run(_put_all())


if __name__ == "__main__":
    run()
