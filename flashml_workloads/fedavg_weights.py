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

import math

__all__ = [
    "NonFiniteWeights",
    "WeightShapeMismatch",
    "apply_delta",
    "decode",
    "encode",
    "reduce_deltas",
    "require_finite",
    "subtract",
]


class WeightShapeMismatch(ValueError):
    """Two weight blobs do not describe the same parameter set.

    Never coerce past this: averaging mismatched blobs would emit weights
    that load fine and train to nonsense.
    """


class NonFiniteWeights(ValueError):
    """A weight/delta blob contains NaN or +/-Infinity.

    This is not a paranoid check, and it is not primarily about attackers:
    Python's `json` module both EMITS and PARSES the non-standard `NaN`,
    `Infinity` and `-Infinity` literals, so a single shard whose learning
    rate diverged writes `NaN` into its delta, the reduce turns every
    weight into `NaN`, every later round trains from `NaN` weights, and the
    run still reports success. There is no recovering from it afterwards —
    NaN is absorbing — so it has to fail at the boundary where it enters.
    """


def require_finite(blob: dict, where: str) -> dict:
    """Raise `NonFiniteWeights` if any value in `blob` is NaN or Inf.

    Named (not `_private`) because it is the containment boundary every
    caller of this module is entitled to use; the message names the
    parameter and the flat index so an operator can point at the shard that
    diverged rather than bisecting a megabyte of JSON.
    """
    for name, param in blob.items():
        for i, v in enumerate(param["data"]):
            try:
                finite = math.isfinite(v)
            except TypeError:  # a node sent a string/null where a float belongs
                finite = False
            if not finite:
                raise NonFiniteWeights(
                    f"{where}: parameter {name!r} index {i} is not a finite "
                    f"number ({v!r})"
                )
    return blob


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
        shape_a = a[name]["shape"]
        shape_b = b[name]["shape"]

        if shape_a != shape_b:
            raise WeightShapeMismatch(
                f"parameter {name!r} shape {shape_a} vs {shape_b}"
            )

        # Both shapes are the same; compute expected data length.
        # Empty shape (scalar) has product 1.
        expected_len = math.prod(shape_a) if shape_a else 1

        # Validate that data length matches declared shape in both blobs.
        data_len_a = len(a[name]["data"])
        if data_len_a != expected_len:
            raise WeightShapeMismatch(
                f"parameter {name!r} declared shape {shape_a} (product {expected_len}) but data has length {data_len_a}"
            )

        data_len_b = len(b[name]["data"])
        if data_len_b != expected_len:
            raise WeightShapeMismatch(
                f"parameter {name!r} declared shape {shape_b} (product {expected_len}) but data has length {data_len_b}"
            )


def subtract(new: dict, base: dict) -> dict:
    _require_same_params(new, base)
    # Checked on the RESULT, not the inputs: it is the cheapest single place
    # that catches a diverged local step (NaN weights) before the delta is
    # uploaded, and NaN/Inf in either input propagates into the difference.
    return require_finite({
        name: {
            "shape": list(new[name]["shape"]),
            "data": [x - y for x, y in zip(new[name]["data"], base[name]["data"])],
        }
        for name in new
    }, "subtract")


def apply_delta(base: dict, delta: dict, scale: float = 1.0) -> dict:
    _require_same_params(base, delta)
    return require_finite({
        name: {
            "shape": list(base[name]["shape"]),
            "data": [b + scale * d
                     for b, d in zip(base[name]["data"], delta[name]["data"])],
        }
        for name in base
    }, "apply_delta")


def reduce_deltas(contributions: list[tuple[dict, int]]) -> dict:
    """Sample-weighted mean of per-worker deltas (FedAvg).

    Weighting by sample count, not by worker, is what keeps the result
    equal to centralized training on the union of the shards when the
    shards are unequal — which they always are once machines differ.
    """
    if not contributions:
        raise ValueError("reduce_deltas: no contributions")
    total = sum(n for _, n in contributions)
    if total < 0:
        raise ValueError("reduce_deltas: negative total samples")
    if total == 0:
        raise ValueError("reduce_deltas: zero total samples")

    # Validating only the TOTAL is not enough, and the gap is a model-
    # poisoning primitive rather than a hygiene nit. `samples` is chosen by
    # an untrusted volunteer node. With contributions (delta=-999, n=-999)
    # and (delta=1.0, n=1000) the total is a healthy 1, but the weights
    # w = n/total are -999 and 1000, so the "average" of two updates of
    # magnitude ~1 is 999001.0 — six orders of magnitude outside the convex
    # hull the average is supposed to stay inside. A weight is only a convex
    # combination when every count is positive, so reject non-positive
    # counts outright. (Ordered AFTER the total guard so an all-zero
    # contribution set still reports the clearer "zero total samples".)
    for i, (_, n) in enumerate(contributions):
        if n <= 0:
            raise ValueError(
                f"reduce_deltas: contribution {i} has non-positive sample "
                f"count {n!r}; sample counts must be > 0 (a negative or zero "
                "count makes the sample weight fall outside [0, 1] and lets "
                "one shard amplify its delta arbitrarily)"
            )

    first = contributions[0][0]
    # Validate first blob's internal consistency and all subsequent blobs,
    # and reject NaN/Inf on the way IN: one NaN anywhere in one shard's delta
    # makes every output weight NaN, and every subsequent round then trains
    # from NaN. Rejecting the round is recoverable; poisoning the model is not.
    for i, (blob, _) in enumerate(contributions):
        # i == 0 validates the first blob against itself (internal
        # shape/data-length consistency), the rest against the first.
        _require_same_params(first, blob)
        require_finite(blob, f"reduce_deltas: contribution {i}")

    out: dict = {}
    for name in first:
        acc = [0.0] * len(first[name]["data"])
        for blob, n in contributions:
            w = n / total
            for i, v in enumerate(blob[name]["data"]):
                acc[i] += w * v
        out[name] = {"shape": list(first[name]["shape"]), "data": acc}
    return out
