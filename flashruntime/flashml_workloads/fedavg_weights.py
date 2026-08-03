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
from typing import NamedTuple

__all__ = [
    "CLIP_FACTOR",
    "ClipEvent",
    "NonFiniteWeights",
    "WeightShapeMismatch",
    "apply_delta",
    "decode",
    "encode",
    "reduce_deltas",
    "reduce_deltas_with_report",
    "require_finite",
    "subtract",
]

#: How many times the round's MEDIAN contribution norm a single contribution
#: may be before it is scaled back to that bound.
#:
#: 3.0, and the number is load-bearing. The governing property of the cap is
#: that an honest round is bit-identical to a round with no cap at all:
#: honest per-shard variation sits well inside 3x the median, so nothing
#: fires and the arithmetic below is untouched. A factor of 1.0 would clip
#: roughly half of every honest round and silently alter results that are
#: correct today — a behaviour change wearing a safety net's clothes.
CLIP_FACTOR: float = 3.0


class ClipEvent(NamedTuple):
    """One contribution that exceeded the round's cap, and by how much.

    `index` is POSITIONAL into the `contributions` list the caller passed,
    not a node id: this module is pure stdlib and knows nothing about
    machines. Attribution is the driver's job — it holds the per-task
    provenance and maps an index back to whoever sent it.
    """

    index: int
    norm: float
    cap: float
    scale: float


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


def _l2_norm(blob: dict) -> float:
    """L2 norm of a delta, flattened across every parameter.

    Only ever called on a blob `require_finite` has already accepted: the
    multiplication below is a `TypeError` on the `None` a volunteer can put
    in `data`, and a NaN anywhere would make the norm NaN, `norm > cap`
    False, and the contribution sail through unscaled.
    """
    return math.sqrt(sum(v * v for p in blob.values() for v in p["data"]))


def _median(values: list[float]) -> float:
    """Median, with the even-length case spelled out: the mean of the two
    middles.

    Which is precisely why the cap is weak at two contributions — the
    median of two values is their mean, and an attacker moves a mean
    directly. Robust statistics need a majority to be honest, and with
    `min_participants = 2` there is no majority to have. Documented, not
    papered over: this does not fail closed at that quorum and must not be
    described as protection there.
    """
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2


def _scale_blob(blob: dict, scale: float) -> dict:
    return {name: {"shape": list(p["shape"]),
                   "data": [scale * v for v in p["data"]]}
            for name, p in blob.items()}


def reduce_deltas(contributions: list[tuple[dict, int]],
                  *, clip_factor: float = CLIP_FACTOR) -> dict:
    """Sample-weighted mean of per-worker deltas (FedAvg).

    Weighting by sample count, not by worker, is what keeps the result
    equal to centralized training on the union of the shards when the
    shards are unequal — which they always are once machines differ.

    A thin wrapper over `reduce_deltas_with_report`, kept because this name
    has one production caller and 30+ tests pinning it. The clip report is
    additive; callers that want it ask for it by name.
    """
    return reduce_deltas_with_report(contributions, clip_factor=clip_factor)[0]


def reduce_deltas_with_report(
    contributions: list[tuple[dict, int]],
    *,
    clip_factor: float = CLIP_FACTOR,
) -> tuple[dict, list[ClipEvent]]:
    """`reduce_deltas`, plus the list of contributions the cap bound.

    Everything MALFORMED is rejected below — non-positive, non-finite and
    non-integer sample counts, mismatched shapes, NaN/Inf weights. What none
    of those guards catch is a contribution that is perfectly well-formed
    and adversarial: `delta = 1e6`, `n = 500`. Every check passes and the
    sample-weighted mean moves the model by whatever the sender chose.

    So, after validation and before the mean, each contribution's L2 norm is
    compared against `C = median(norms) * clip_factor` and anything above it
    is scaled to `C`. Median-anchored rather than a fixed constant because
    the right magnitude depends on the model, the learning rate and the
    round number, none of which this module knows — and because with a
    majority of honest contributors the median is an honest value, which an
    attacker-chosen mean is not.

    Bounds MAGNITUDE, not direction. A small, consistently-biased delta
    every round is unaffected, a node returning zeros still earns credit,
    and a colluding majority defeats it by construction. It is a cap on how
    far one contributor can move the model, not result verification.

    Nothing is enforced here beyond the scaling: the events are returned so
    the caller can record them. Revocation is a human decision.
    """
    # First, because it is the function's own configuration rather than
    # untrusted input, and because the failure mode of getting it wrong is
    # the worst one available: a non-positive or non-finite factor would
    # disable the cap silently, the round would still reduce, and it would
    # still report an empty clip list that an operator reads as "nobody
    # tried". `clip_factor=0` would additionally zero every contribution.
    try:
        usable = math.isfinite(clip_factor) and clip_factor > 0
    except TypeError:  # not a number at all
        usable = False
    if not usable:
        raise ValueError(
            f"reduce_deltas: clip_factor must be a finite number > 0, got "
            f"{clip_factor!r}; a non-positive or non-finite value would "
            "silently disable the influence cap rather than widening it"
        )

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
        # A NaN/Inf sample count slips BOTH the total guard above and the
        # `n <= 0` guard below, because every comparison against NaN is
        # False: `total <= 0`, `total == 0` and `n <= 0` are all False for a
        # NaN contribution. It happens to be caught downstream by the
        # finiteness check on the reduced result, but only by accident, not
        # by design, so it is rejected explicitly here, by index, before it
        # can reach that accidental safety net.
        if isinstance(n, float) and not math.isfinite(n):
            raise ValueError(
                f"reduce_deltas: contribution {i} has a non-finite sample "
                f"count {n!r}; sample counts must be a finite positive "
                "integer"
            )
        # `bool` is a subclass of `int` (`True == 1`, `False == 0`), so a
        # bool sample count would otherwise slide through silently. A
        # sample count is a count of training examples, not a flag, and
        # this boundary receives untrusted JSON from a volunteer node where
        # `true`/`false` is a plausible malformed value for a field that
        # should be an integer — reject it rather than coerce it. Likewise
        # a non-integer float (e.g. `2.5`) does not correspond to any real
        # shard size; it happens not to break the convex-combination math
        # below, but silently accepting it is the same kind of
        # accidental-safety gap the NaN/Inf case above is about.
        if isinstance(n, bool) or not isinstance(n, int):
            raise ValueError(
                f"reduce_deltas: contribution {i} has a non-integer sample "
                f"count {n!r} (type {type(n).__name__}); sample counts must "
                "be a plain positive int, not a bool or a fractional float"
            )
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

    # -- bounded influence, and ONLY here: after every guard above, because
    # a malformed contribution must raise its own error rather than be
    # quietly scaled into something plausible. Clipping caps a delta's
    # magnitude and does nothing at all about a negative sample weight.
    #
    # `reduced` deliberately reuses the caller's own blob objects for every
    # contribution that is not clipped, so an honest round accumulates the
    # exact same float objects in the exact same order as it did before this
    # existed. Byte-identical is the governing property; rebuilding every
    # blob "harmlessly" would be the easiest way to lose it.
    # Computed once, not once per use: deltas are megabytes, and this walks
    # every weight in every contribution.
    norms = [_l2_norm(blob) for blob, _ in contributions]
    cap = _median(norms) * clip_factor
    clipped: list[tuple[dict, int]] = list(contributions)
    events: list[ClipEvent] = []
    for i, ((blob, n), norm) in enumerate(zip(contributions, norms)):
        # Strict `>` against a cap that is never negative — so a zero-norm
        # contribution (a converged shard, or a lazy node returning zeros)
        # is never the one being scaled, and `cap / norm` never divides by
        # zero. An all-zero round puts the cap at 0.0 too and clips nothing.
        if norm > cap:
            scale = cap / norm
            clipped[i] = (_scale_blob(blob, scale), n)
            events.append(ClipEvent(index=i, norm=norm, cap=cap, scale=scale))

    out: dict = {}
    for name in first:
        acc = [0.0] * len(first[name]["data"])
        for blob, n in clipped:
            w = n / total
            for i, v in enumerate(blob[name]["data"]):
                acc[i] += w * v
        out[name] = {"shape": list(first[name]["shape"]), "data": acc}
    return out, events
