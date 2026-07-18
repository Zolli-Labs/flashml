# FlashML Tooling & Runpod Architecture

> **Note:** FlashML is now a provider-agnostic distributed training library
> ([README.md](../../README.md)); Runpod is its first real connector. This document is
> the technical foundation for `flashml/providers/runpod/` — the connector spec
> it implements lives in [PROVIDERS.md](PROVIDERS.md), and the build
> plan in [ROADMAP.md](ROADMAP.md) (see also [IMPLEMENTATION.md](IMPLEMENTATION.md) for the original design rationale).
> Full documentation map: [INDEX.md](INDEX.md).

This document catalogs every tool in the Runpod-facing stack and goes deep on how
FlashML maps its distributed K-Means pipeline onto **Runpod Flash** primitives —
not just "serverless GPUs," but the specific SDK constructs that make the
MapReduce loop fast, cheap, and simple to operate.

---

## Stack at a Glance

| Layer | Tool | Role |
|---|---|---|
| Frontend | React / Next.js (or similar SPA) | Upload dataset, configure job, live dashboard |
| Coordinator | Python (FastAPI) | Owns the training loop, dispatches to Flash endpoints |
| Compute | **Runpod Flash SDK** (`runpod-flash`) | Embedding, Map, and Reduce execution on remote GPU/CPU workers |
| Compute ops | **runpodctl** | GPU/cost visibility, volume + hub management, debugging |
| Storage | Runpod **NetworkVolume** + S3-compatible API | Dataset shards, centroids, iteration state |
| Models | **HuggingFace CLI (`hf`)** | Pre-fetch/cache embedding models |
| Packaging | **Docker** + **GitHub CLI (`gh`)** | Optional custom worker images, Hub publishing |
| Visualization | Plotly / Recharts + PCA/t-SNE | Centroid movement, cluster scatter, worker timeline |

The throughline: **the coordinator never touches a GPU directly.** Every
compute-heavy step — embedding generation and the K-Means map step — is a
`runpod_flash.Endpoint`. The coordinator just orchestrates HTTP calls and runs the
(cheap) reduce step itself.

---

## Runpod Flash: Core Compute Engine

### Why Flash specifically (not raw Runpod pods/serverless)

Flash gives FlashML three things a hand-rolled serverless setup wouldn't:

1. **`flash dev`** — write the map/reduce function locally, it executes on a real
   remote GPU with hot reload and live worker logs streamed to the terminal. This
   *is* the hackathon dev loop: no build/push/deploy cycle while iterating on the
   K-Means kernel.
2. **The `Endpoint` decorator** — turns a plain async Python function into a
   fully autoscaled GPU/CPU endpoint with no Dockerfile, no provisioning code.
3. **Mixed CPU/GPU pipelines in one SDK** — embedding (GPU), map (GPU or CPU
   depending on dataset size), and reduce (CPU/in-process) are all `Endpoint`
   objects with independent worker pools, autoscaling rules, and GPU tiers.

### Endpoint design for FlashML

FlashML uses Flash's **Mode 1 (queue-based decorator)** for both compute stages —
each stage is its own endpoint with its own worker pool, matching the job's
user-selected worker count (4 / 8 / 16).

**Embedding service** (text/image → vectors, embarrassingly parallel):

```python
from runpod_flash import Endpoint, GpuGroup

@Endpoint(
    name="flashml-embed",
    gpu=GpuGroup.ADA_24,        # RTX 4090 tier — cheap, plenty for sentence/vision encoders
    workers=(1, 5),
    idle_timeout=180,
    dependencies=["sentence-transformers", "torch"],
)
async def embed(data: dict):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return {"embeddings": model.encode(data["texts"]).tolist()}
```

**K-Means map step** (the hot loop — runs once per iteration, per shard):

```python
from runpod_flash import Endpoint, GpuGroup, NetworkVolume

shards = NetworkVolume(name="flashml-shards", size=100)

@Endpoint(
    name="flashml-kmeans-map",
    gpu=[GpuGroup.AMPERE_24, GpuGroup.ADA_24],  # list -> auto-switches by supply
    workers=(0, 16),                             # caps at the user's worker count
    volume=shards,
    idle_timeout=120,
    dependencies=["torch"],
)
async def kmeans_map(data: dict):
    import torch
    vectors = torch.load(f"/runpod-volume/{data['job_id']}/shard_{data['shard_id']}.pt")
    centroids = torch.tensor(data["centroids"], device="cuda")
    dists = torch.cdist(vectors.to("cuda"), centroids)
    assignments = dists.argmin(dim=1)

    partial_sums, partial_counts = {}, {}
    for c in assignments.unique().tolist():
        mask = assignments == c
        partial_sums[c] = vectors[mask.cpu()].sum(dim=0).tolist()
        partial_counts[c] = int(mask.sum())

    return {"partial_sums": partial_sums, "partial_counts": partial_counts}
```

Two details here are load-bearing, not stylistic:

- **`gpu=[GpuGroup.AMPERE_24, GpuGroup.ADA_24]` with `workers=(0, 16)`** — Flash
  only auto-switches GPU types based on live supply when max workers ≥ 5. Passing
  a list instead of a single tier means a `k=16` job won't queue behind whichever
  single GPU SKU happens to be scarce that hour. This is a real cost/availability
  win for free.
- **The vectors never travel over HTTP after partition time.** Flash enforces a
  **10MB request payload limit**. A 1M-row embedding shard blows past that
  instantly. So shards are written to a `NetworkVolume` once, at partition time,
  and every subsequent map call ships only `job_id`, `shard_id`, and the current
  centroids (a few KB) — the worker reads its shard straight off the mounted
  volume. This also means **iteration N+1 doesn't re-upload the dataset**, which
  is the difference between a K-Means job that scales to millions of vectors and
  one that times out on iteration 2.

**Reduce step** stays cheap (summing small per-cluster arrays) and runs **inside
the coordinator process**, not as a separate endpoint — there's no GPU work and
no payload-size concern once you're aggregating partial sums/counts instead of
raw vectors.

### CPU tier for non-GPU shards

For numeric CSV datasets where vectors are used directly (no embedding step),
the map step doesn't need a GPU at all:

```python
from runpod_flash import Endpoint, CpuInstanceType

@Endpoint(name="flashml-kmeans-map-cpu", cpu=CpuInstanceType.CPU5C_4_8,
          workers=(0, 16), volume=shards, dependencies=["numpy"])
async def kmeans_map_cpu(data: dict):
    ...
```

Letting the user's "k workers" selector pick CPU vs. GPU tiers per dataset type
is a direct cost lever to expose in the UI — CPU workers are dramatically
cheaper for small numeric data, GPU workers pay for themselves on large
embedding matrices.

### Dev loop

```bash
flash dev > /tmp/flash-dev.log 2>&1 &
until grep -q "flash dev  localhost:" /tmp/flash-dev.log; do sleep 2; done
curl -s localhost:8888/main/kmeans_map -d '{"data": {...}}'
```

Iterate on the map/reduce kernel against a real remote GPU with hot reload and
live worker logs — no Docker build, no redeploy, just save and re-curl. Switch to
`flash deploy` only once the kernel is correct, to get a stable endpoint for the
live demo.

---

## runpodctl: Operations & Debugging

`runpodctl` covers everything outside the Flash SDK's scope — infrastructure
visibility and one-off setup that the demo and cost story depend on:

```bash
runpodctl gpu list                          # live GPU availability — informs which tier to default the UI to
runpodctl billing serverless                # actual spend per endpoint — powers the "cost estimate" dashboard panel
runpodctl network-volume create --name flashml-shards --size 100 --data-center-id US-CA-2
runpodctl network-volume get <volume-id>    # confirm datacenter placement matches the worker pool
runpodctl hub search sentence-transformers  # discover a prebuilt embedding worker instead of building one
runpodctl ssh info <pod-id>                 # debug a stuck worker if running a pod-mode embedding service
```

`runpodctl gpu list` and `billing serverless` are what turn "Cost estimate" and
"worker utilization" from the IMPLEMENTATION.md nice-to-haves into something
backed by real numbers instead of guesses.

---

## Companion CLIs

| CLI | Use in FlashML |
|---|---|
| **`hf`** (HuggingFace) | Pre-download/cache the embedding model (`all-MiniLM-L6-v2`, CLIP, etc.) so cold starts don't pay the download cost on first call |
| **`docker`** | Only needed if the embedding service moves to Flash's Mode 3 (external image) for a heavier model that exceeds the 1500MB decorator-mode artifact limit |
| **`gh`** | Publish K-Means (and future algorithm) workers to the Runpod Hub post-hackathon — the Hub indexes releases, so every update needs a `gh release create` |
| **`aws`** (S3-compatible) | Power-user path for huge datasets: `aws s3 cp` straight into the `flashml-shards` NetworkVolume, bypassing the web upload entirely. Directly serves the README's "developer has a large dataset" use case — no frontend upload limit applies |

```bash
# Bulk dataset upload bypassing the app entirely
aws s3 cp my-1m-vectors.parquet \
  --region US-CA-2 \
  --endpoint-url https://s3api-US-CA-2.runpod.io/ \
  s3://<flashml-shards-volume-id>/raw/
```

---

## End-to-End Flow (tools view)

```text
flash dev              -> iterate on embed/map kernels against real GPUs
flash deploy           -> ship stable embed + map endpoints for the demo
runpodctl nv create    -> provision the shared NetworkVolume for shards/centroids
aws s3 cp / coordinator-> dataset lands on the NetworkVolume
embed Endpoint (GPU)   -> vectors written back to the volume
partition (coordinator)-> shard files written to the volume, once
for each iteration:
    kmeans_map Endpoint (GPU/CPU, autoscaled to worker count) -> partial sums/counts only
    reduce (coordinator, in-process)                          -> new centroids
runpodctl billing       -> cost shown on the results dashboard
```

Infrastructure stays invisible to the end user the whole way through — every
Runpod-specific decision (GPU tier, volume placement, autoscale bounds) is a
config value the coordinator picks on the user's behalf from `k`, dataset size,
and dataset type.
