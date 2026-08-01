"""Weight/delta encoding and the federated-averaging reduce.

Weights cross the wire as JSON so the driver never imports torch: it runs
inside the cloud API (spec §5.4.5), which must stay a light service. Only
`fedavg_worker` needs torch, and it converts at the boundary.

    {"<param>": {"shape": [int, ...], "data": [float, ...]}}

`data` is the flattened tensor in row-major order; `shape` restores it.
Pure stdlib on purpose — same rule as kmeans_shard and sgd_trainer, so
this module runs on any device, including inside a --network none
container.
"""

from __future__ import annotations

__all__ = [
    "WeightShapeMismatch",
    "apply_delta",
    "decode",
    "encode",
    "reduce_deltas",
    "subtract",
]


class WeightShapeMismatch(ValueError):
    """Two weight blobs do not describe the same parameter set.

    Never coerce past this: averaging mismatched blobs would emit weights
    that load fine and train to nonsense.
    """


def encode(state: dict[str, tuple[list[int], list[float]]]) -> dict:
    return {name: {"shape": list(shape), "data": list(data)}
            for name, (shape, data) in state.items()}


def decode(blob: dict) -> dict[str, tuple[list[int], list[float]]]:
    return {name: (list(p["shape"]), list(p["data"])) for name, p in blob.items()}


def _require_same_params(a: dict, b: dict) -> None:
    if a.keys() != b.keys():
        raise WeightShapeMismatch(
            f"parameter names differ: {sorted(a.keys())} vs {sorted(b.keys())}"
        )
    for name in a:
        if a[name]["shape"] != b[name]["shape"]:
            raise WeightShapeMismatch(
                f"parameter {name!r} shape {a[name]['shape']} vs {b[name]['shape']}"
            )


def subtract(new: dict, base: dict) -> dict:
    _require_same_params(new, base)
    return {
        name: {
            "shape": list(new[name]["shape"]),
            "data": [x - y for x, y in zip(new[name]["data"], base[name]["data"])],
        }
        for name in new
    }


def apply_delta(base: dict, delta: dict, scale: float = 1.0) -> dict:
    _require_same_params(base, delta)
    return {
        name: {
            "shape": list(base[name]["shape"]),
            "data": [b + scale * d
                     for b, d in zip(base[name]["data"], delta[name]["data"])],
        }
        for name in base
    }


def reduce_deltas(contributions: list[tuple[dict, int]]) -> dict:
    """Sample-weighted mean of per-worker deltas (FedAvg).

    Weighting by sample count, not by worker, is what keeps the result
    equal to centralized training on the union of the shards when the
    shards are unequal — which they always are once machines differ.
    """
    if not contributions:
        raise ValueError("reduce_deltas: no contributions")
    total = sum(n for _, n in contributions)
    if total <= 0:
        raise ValueError("reduce_deltas: zero total samples")

    first = contributions[0][0]
    for blob, _ in contributions[1:]:
        _require_same_params(first, blob)

    out: dict = {}
    for name in first:
        acc = [0.0] * len(first[name]["data"])
        for blob, n in contributions:
            w = n / total
            for i, v in enumerate(blob[name]["data"]):
                acc[i] += w * v
        out[name] = {"shape": list(first[name]["shape"]), "data": acc}
    return out
