"""Determinism tests for the sharded K-Means workload (no Ray needed)."""

import numpy as np

from flashml_workloads.sharded_kmeans import _true_centers, make_shard


def test_shards_are_deterministic():
    a = make_shard(seed=42, shard_id=7, samples_per_shard=500, clusters=6, dimensions=24)
    b = make_shard(seed=42, shard_id=7, samples_per_shard=500, clusters=6, dimensions=24)
    assert np.array_equal(a, b)  # a retried task recomputes identical data


def test_shards_differ_by_id_and_seed():
    base = make_shard(seed=42, shard_id=0, samples_per_shard=100, clusters=3, dimensions=4)
    other_shard = make_shard(seed=42, shard_id=1, samples_per_shard=100, clusters=3, dimensions=4)
    other_seed = make_shard(seed=43, shard_id=0, samples_per_shard=100, clusters=3, dimensions=4)
    assert not np.array_equal(base, other_shard)
    assert not np.array_equal(base, other_seed)


def test_true_centers_deterministic_and_shaped():
    c1 = _true_centers(42, 6, 24)
    c2 = _true_centers(42, 6, 24)
    assert np.array_equal(c1, c2)
    assert c1.shape == (6, 24)


def test_full_dataset_reduce_matches_direct_kmeans_step():
    """One driver iteration over shards == the same step over the full data."""
    seed, clusters, dims, n_shards, per_shard = 7, 3, 5, 6, 200
    shards = [make_shard(seed, s, per_shard, clusters, dims) for s in range(n_shards)]
    X = np.vstack(shards)
    centroids = _true_centers(seed, clusters, dims)

    # Reference: single-machine assignment + mean update.
    d2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
    labels = d2.argmin(axis=1)
    expected = np.vstack([
        X[labels == c].mean(axis=0) if (labels == c).any() else centroids[c]
        for c in range(clusters)
    ])

    # Sharded: partial sums/counts reduced like the driver does.
    sums = np.zeros_like(centroids)
    counts = np.zeros(clusters)
    for shard in shards:
        sd2 = ((shard[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
        slabels = sd2.argmin(axis=1)
        for c in range(clusters):
            mask = slabels == c
            counts[c] += mask.sum()
            if mask.any():
                sums[c] += shard[mask].sum(axis=0)
    reduced = centroids.copy()
    nz = counts > 0
    reduced[nz] = sums[nz] / counts[nz, None]

    assert np.allclose(reduced, expected)
