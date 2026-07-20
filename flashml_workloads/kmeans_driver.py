"""K-means driver: composes lease jobs into Lloyd's algorithm.

One iteration = one Mode A job (N independent shard tasks); the driver
reduces the shard partials into new centroids and submits the next
iteration. This is the stage-composition pattern: pipelines are jobs
chained by a driver, not a new execution mode — a dead worker inside an
iteration costs one shard retry, and a dead *driver* can resume from the
last completed iteration's artifacts.

The same broadcast → partial-sums → reduce shape is, conceptually, what
gradient synchronization does with gradients instead of cluster sums.

Pure stdlib (urllib); usable as a library (`run_kmeans`) or CLI.
"""

from __future__ import annotations

import json
import random
import time
import urllib.request


def init_centroids(rows: list[list[float]], k: int, seed: int = 0) -> list[list[float]]:
    """Deterministic sample of k distinct starting points."""
    rng = random.Random(seed)
    picks = rng.sample(range(len(rows)), k)
    return [list(rows[i]) for i in sorted(picks)]


def reduce_partials(
    partials: list[dict], prev_centroids: list[list[float]]
) -> tuple[list[list[float]], float]:
    """Sum the shard partials; new centroid = sums/counts. An empty cluster
    keeps its previous centroid (the standard degenerate-cluster rule).
    Returns (new_centroids, total_inertia)."""
    k = len(prev_centroids)
    dims = len(prev_centroids[0])
    sums = [[0.0] * dims for _ in range(k)]
    counts = [0] * k
    inertia = 0.0
    for p in partials:
        for j in range(k):
            counts[j] += p["counts"][j]
            for d in range(dims):
                sums[j][d] += p["sums"][j][d]
        inertia += p["inertia"]
    centroids = [
        [s / counts[j] for s in sums[j]] if counts[j] else list(prev_centroids[j])
        for j in range(k)
    ]
    return centroids, inertia


# ---------------------------------------------------------------------------
# Coordinator plumbing (stdlib HTTP)
# ---------------------------------------------------------------------------


def _http(method: str, url: str, data: bytes | None = None, ctype="application/json"):
    req = urllib.request.Request(
        url, data=data, method=method, headers={"Content-Type": ctype} if data else {}
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        body = r.read()
        return json.loads(body) if body else None


def shard_and_upload(
    base_url: str, rows: list[list[float]], n_shards: int, prefix: str
) -> list[str]:
    """Round-robin the rows into shards, upload each as a CSV artifact to
    the coordinator; returns the artifact:// URIs."""
    uris = []
    for s in range(n_shards):
        shard_rows = rows[s::n_shards]
        csv_bytes = ("\n".join(",".join(f"{v:.6f}" for v in r) for r in shard_rows) + "\n").encode()
        key = f"{prefix}/shard-{s:03d}.csv"
        _http("PUT", f"{base_url}/v1alpha1/artifacts/{key}", csv_bytes,
              ctype="application/octet-stream")
        uris.append(f"artifact://{key}")
    return uris


def run_kmeans(
    base_url: str,
    shard_uris: list[str],
    centroids: list[list[float]],
    iterations: int = 5,
    lease_seconds: float = 30.0,
    poll_seconds: float = 0.5,
    timeout_per_iteration_s: float = 300.0,
) -> dict:
    """Run Lloyd's algorithm as a sequence of lease jobs. Returns
    {centroids, inertia_history, job_ids}."""
    history: list[float] = []
    job_ids: list[str] = []
    for it in range(iterations):
        job = _http("POST", f"{base_url}/v1alpha1/jobs", json.dumps({
            "apiVersion": "flashml.dev/v1alpha1", "kind": "Job",
            "metadata": {"name": f"kmeans-it{it:02d}"},
            "spec": {
                "execution": {"backend": "leases"},
                "image": {"repository": "local/tier1", "tag": "dev"},
                "workload": {"type": "sharded_kmeans", "parameters": {
                    "shards": shard_uris, "centroids": centroids,
                    "iteration": it, "lease_seconds": lease_seconds}},
            }}).encode())
        job_id = job["job_id"]
        job_ids.append(job_id)

        deadline = time.monotonic() + timeout_per_iteration_s
        while True:
            state = _http("GET", f"{base_url}/v1alpha1/jobs/{job_id}")["state"]
            if state == "SUCCEEDED":
                break
            if state in ("FAILED", "CANCELLED"):
                raise RuntimeError(f"kmeans iteration {it} ended {state} (job {job_id})")
            if time.monotonic() > deadline:
                raise TimeoutError(f"kmeans iteration {it} timed out (job {job_id})")
            time.sleep(poll_seconds)

        partials = []
        for a in _http("GET", f"{base_url}/v1alpha1/jobs/{job_id}/artifacts"):
            if a["key"].endswith("metrics.json"):
                partials.append(_http("GET", f"{base_url}/v1alpha1/artifacts/{a['key']}"))
        if len(partials) != len(shard_uris):
            raise RuntimeError(
                f"iteration {it}: expected {len(shard_uris)} partials, got {len(partials)}"
            )
        centroids, inertia = reduce_partials(partials, centroids)
        history.append(inertia)

    return {"centroids": centroids, "inertia_history": history, "job_ids": job_ids}
