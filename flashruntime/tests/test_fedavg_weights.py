import json
import math

import pytest

from flashml_workloads.fedavg_weights import (
    CLIP_FACTOR,
    ClipEvent,
    NonFiniteWeights,
    WeightShapeMismatch,
    apply_delta,
    decode,
    encode,
    reduce_deltas,
    reduce_deltas_with_report,
    subtract,
)


def _blob(**params):
    """{'w': [1.0, 2.0]} -> encoded blob with shape [len]."""
    return {k: {"shape": [len(v)], "data": list(v)} for k, v in params.items()}


def test_encode_decode_round_trip():
    state = {"w": ([2, 1], [1.5, -2.5]), "b": ([1], [0.25])}
    assert decode(encode(state)) == state


def test_subtract_is_elementwise():
    new, base = _blob(w=[3.0, 5.0]), _blob(w=[1.0, 2.0])
    assert subtract(new, base) == _blob(w=[2.0, 3.0])


def test_apply_delta_adds_scaled_delta():
    base, delta = _blob(w=[1.0, 1.0]), _blob(w=[2.0, 4.0])
    assert apply_delta(base, delta, scale=0.5) == _blob(w=[2.0, 3.0])


def test_reduce_deltas_weights_by_sample_count():
    # 100 samples say +1.0, 300 samples say +5.0 -> (100*1 + 300*5)/400 = 4.0
    got = reduce_deltas([(_blob(w=[1.0]), 100), (_blob(w=[5.0]), 300)])
    assert got["w"]["data"] == [pytest.approx(4.0)]


def test_reduce_deltas_single_contribution_is_identity():
    assert reduce_deltas([(_blob(w=[2.0, -3.0]), 7)]) == _blob(w=[2.0, -3.0])


def test_reduce_deltas_rejects_empty():
    with pytest.raises(ValueError, match="no contributions"):
        reduce_deltas([])


def test_reduce_deltas_rejects_zero_total_samples():
    # Would divide by zero and silently emit garbage weights.
    with pytest.raises(ValueError, match="zero total samples"):
        reduce_deltas([(_blob(w=[1.0]), 0)])


def test_reduce_deltas_rejects_mismatched_shapes():
    with pytest.raises(WeightShapeMismatch):
        reduce_deltas([(_blob(w=[1.0]), 1), (_blob(w=[1.0, 2.0]), 1)])


def test_reduce_deltas_rejects_mismatched_param_names():
    with pytest.raises(WeightShapeMismatch):
        reduce_deltas([(_blob(w=[1.0]), 1), (_blob(bias=[1.0]), 1)])


def test_subtract_rejects_mismatched_data_length():
    """Data length must match declared shape product; silent truncation is a bug."""
    new = {"w": {"shape": [2], "data": [1.0, 2.0, 3.0]}}  # 3 elements but shape says 2
    base = {"w": {"shape": [2], "data": [5.0, 6.0]}}
    with pytest.raises(WeightShapeMismatch):
        subtract(new, base)


def test_apply_delta_rejects_mismatched_data_length():
    """Data length must match declared shape product."""
    base = {"w": {"shape": [2], "data": [1.0, 1.0]}}
    delta = {"w": {"shape": [2], "data": [2.0, 4.0, 6.0]}}  # 3 elements but shape says 2
    with pytest.raises(WeightShapeMismatch):
        apply_delta(base, delta, scale=0.5)


def test_reduce_deltas_rejects_internal_data_length_mismatch():
    """Each contribution's data length must match its declared shape."""
    # Second blob has 2 data elements but declares shape [3]
    with pytest.raises(WeightShapeMismatch):
        reduce_deltas([
            ({"w": {"shape": [3], "data": [1.0, 1.0, 1.0]}}, 1),
            ({"w": {"shape": [3], "data": [9.0, 9.0]}}, 1)
        ])


# -- C2: sample count is an unbounded model-poisoning primitive -------------


def test_reduce_deltas_rejects_a_negative_sample_count():
    """Validating only the TOTAL is not enough.

    (delta=-999, n=-999) and (delta=1.0, n=1000) sum to a perfectly healthy
    total of 1, so the old zero/negative-total guard passed — but the sample
    weights are then -999 and 1000, and the "average" of two updates of
    magnitude ~1 comes out at 999001.0. That is a ~10^6x amplified step far
    outside the convex hull of the honest updates, bought with one integer
    from one untrusted volunteer. A weight is only a convex combination when
    every count is positive.
    """
    with pytest.raises(ValueError, match="non-positive sample count"):
        reduce_deltas([(_blob(w=[-999.0]), -999), (_blob(w=[1.0]), 1000)])


def test_reduce_deltas_names_the_offending_contribution():
    with pytest.raises(ValueError, match=r"contribution 1 .*-5"):
        reduce_deltas([(_blob(w=[1.0]), 10), (_blob(w=[1.0]), -5)])


def test_reduce_deltas_rejects_a_zero_sample_count_among_positive_ones():
    with pytest.raises(ValueError, match="non-positive sample count"):
        reduce_deltas([(_blob(w=[1.0]), 10), (_blob(w=[1.0]), 0)])


def test_reduce_deltas_still_reports_the_zero_total_guard():
    """The per-contribution check must not shadow the existing total guard."""
    with pytest.raises(ValueError, match="zero total samples"):
        reduce_deltas([(_blob(w=[1.0]), 0)])


# -- C3: NaN/Inf silently and permanently destroy the model ------------------


def test_json_really_does_round_trip_nan_and_inf():
    """The premise of C3, pinned so nobody 'simplifies' the guard away:
    Python's json both EMITS and PARSES NaN/Infinity, so a diverged shard's
    delta arrives at the driver as a genuine float('nan'), not a parse error.
    """
    raw = json.dumps({"w": {"shape": [2], "data": [float("nan"), float("inf")]}})
    assert "NaN" in raw and "Infinity" in raw
    back = json.loads(raw)["w"]["data"]
    assert back[0] != back[0] and back[1] == float("inf")


def test_reduce_deltas_rejects_a_nan_contribution():
    """One NaN anywhere makes EVERY output weight NaN, every later round
    trains from NaN, and nothing reports a failure. No attacker required —
    a learning rate that diverges on one shard does it."""
    with pytest.raises(NonFiniteWeights, match=r"contribution 1.*'w' index 1"):
        reduce_deltas([(_blob(w=[1.0, 1.0]), 10),
                       (_blob(w=[1.0, float("nan")]), 10)])


def test_reduce_deltas_rejects_an_inf_contribution():
    with pytest.raises(NonFiniteWeights, match=r"contribution 0.*'w' index 0"):
        reduce_deltas([(_blob(w=[float("inf")]), 10), (_blob(w=[1.0]), 10)])


def test_reduce_deltas_rejects_a_non_numeric_contribution():
    """A volunteer can put anything JSON-encodable in `data`; a null must
    fail as a typed weights error, not a TypeError from deep in the loop."""
    with pytest.raises(NonFiniteWeights):
        reduce_deltas([({"w": {"shape": [1], "data": [None]}}, 10)])


def test_apply_delta_rejects_a_non_finite_result():
    with pytest.raises(NonFiniteWeights, match=r"apply_delta.*'w' index 0"):
        apply_delta(_blob(w=[1.0]), _blob(w=[float("inf")]))


def test_apply_delta_rejects_a_nan_produced_by_inf_minus_inf():
    """Inf in the base and -Inf in the delta multiply out to NaN: the check
    is on the RESULT precisely so arithmetic that manufactures NaN from
    individually 'valid-looking' inputs cannot slip through."""
    with pytest.raises(NonFiniteWeights):
        apply_delta(_blob(w=[float("inf")]), _blob(w=[float("-inf")]))


def test_subtract_rejects_a_non_finite_result():
    """Catches a diverged local step at the worker, before the delta is
    ever uploaded."""
    with pytest.raises(NonFiniteWeights, match="subtract"):
        subtract(_blob(w=[float("nan")]), _blob(w=[1.0]))


# -- C5: a NaN sample count slips both the total and per-contribution ------
# -- guards by accident, because every comparison with NaN is False --------


def test_reduce_deltas_rejects_a_nan_sample_count_explicitly():
    """`n = float('nan')` fails neither `total <= 0` (nan compares False
    against everything) nor the old `n <= 0` per-contribution guard, for the
    same reason. It happened to be caught downstream by `require_finite` on
    the reduced result, but only by accident — the per-contribution guard
    must name it explicitly, by index, rather than relying on that."""
    with pytest.raises(ValueError, match=r"contribution 1.*non-finite sample count"):
        reduce_deltas([(_blob(w=[1.0]), 10), (_blob(w=[1.0]), float("nan"))])


def test_reduce_deltas_rejects_an_infinite_sample_count_explicitly():
    with pytest.raises(ValueError, match=r"contribution 0.*non-finite sample count"):
        reduce_deltas([(_blob(w=[1.0]), float("inf")), (_blob(w=[1.0]), 10)])


def test_reduce_deltas_rejects_a_bool_sample_count():
    """`True == 1` and `False == 0` in Python, so a bool sample count would
    silently behave as a real count. Decision: reject it anyway — a sample
    count is a count of training examples, not a flag, and this boundary
    receives untrusted JSON where `true`/`false` is a plausible malformed
    payload for a field that should be an integer."""
    with pytest.raises(ValueError, match=r"contribution 1.*non-integer sample count"):
        reduce_deltas([(_blob(w=[1.0]), 10), (_blob(w=[1.0]), True)])


def test_reduce_deltas_rejects_a_non_integer_float_sample_count():
    """A fractional sample count (2.5 "examples") does not correspond to any
    real shard size. It happens not to break the convex-combination math
    (still positive, still sums correctly), but silently accepting it is the
    same kind of accidental-safety gap C3/C5 are about: reject it by design."""
    with pytest.raises(ValueError, match=r"contribution 1.*non-integer sample count"):
        reduce_deltas([(_blob(w=[1.0]), 10), (_blob(w=[1.0]), 2.5)])


# -- C6: bounded influence — median-anchored L2 clipping --------------------
#
# Everything malformed is already rejected above. What is not rejected is a
# contribution that is perfectly WELL-FORMED and adversarial: delta = 1e6,
# n = 500. Every guard passes and the sample-weighted mean moves the model
# by whatever the attacker chose. These tests pin the cap that closes it —
# and, first and above all else, pin that an honest round is untouched.


def _norm(blob):
    return sum(v * v for p in blob.values() for v in p["data"]) ** 0.5


def test_clip_factor_defaults_to_three():
    """3.0, not 1.0, and the reason is the test directly below this one.

    Honest per-shard variation sits well inside 3x the median. A factor of
    1.0 would clip roughly half of every honest round and silently alter
    results that are correct today — a behaviour change wearing a safety
    net's clothes.
    """
    assert CLIP_FACTOR == 3.0


def test_an_honest_round_is_byte_identical_to_the_unclipped_reduce():
    """THE governing property: no clip fires, so the arithmetic is the old
    arithmetic, float for float.

    The expected value is a LITERAL captured from the implementation as it
    stood before clipping existed, not recomputed by the test, so it cannot
    drift with the code it is supposed to pin. Exact `==`, never
    `pytest.approx`: "similar" is what a behaviour change looks like.

    Norms here are 0.3775 / 0.3089 / 0.3803; the median is 0.3775 and the
    cap 1.132, so nothing is near it — which is the point.
    """
    contributions = [
        ({"w": {"shape": [3], "data": [0.10, -0.20, 0.30]},
          "b": {"shape": [1], "data": [0.05]}}, 100),
        ({"w": {"shape": [3], "data": [0.15, -0.10, 0.25]},
          "b": {"shape": [1], "data": [-0.02]}}, 250),
        ({"w": {"shape": [3], "data": [-0.05, 0.30, 0.20]},
          "b": {"shape": [1], "data": [0.11]}}, 175),
    ]
    expected = {
        "w": {"shape": [3], "data": [0.07380952380952381,
                                     0.014285714285714277,
                                     0.24285714285714283]},
        "b": {"shape": [1], "data": [0.03666666666666667]},
    }
    assert reduce_deltas(contributions) == expected

    reduced, events = reduce_deltas_with_report(contributions)
    assert reduced == expected
    assert events == []


def test_a_1e6_contribution_is_scaled_to_the_cap():
    """The attack from section 1 of the design, and the bound that answers it.

    Note what is NOT claimed. The design's definition of done says the
    reduced result "stays within the honest convex hull"; with four
    equal-weight contributions that is arithmetically false and no clip
    factor makes it true — a contribution admitted at the cap (3x the
    median) necessarily pulls the mean past the honest maximum. Here the
    honest hull is [0.9, 1.1] and the reduced result is 1.5375.

    The property the mechanism actually guarantees, and the one asserted,
    is bounded influence: after clipping every contribution has norm <= C,
    so the mean does too. Without the cap this round reduces to 250000.75.
    """
    contributions = [(_blob(w=[0.9]), 100), (_blob(w=[1.0]), 100),
                     (_blob(w=[1.1]), 100), (_blob(w=[1e6]), 100)]
    # sorted norms 0.9, 1.0, 1.1, 1e6 -> median (1.0 + 1.1) / 2 = 1.05
    cap = 1.05 * CLIP_FACTOR

    reduced, events = reduce_deltas_with_report(contributions)

    assert len(events) == 1
    assert events[0].index == 3
    assert events[0].norm == 1e6
    assert events[0].cap == pytest.approx(cap)
    assert events[0].scale == pytest.approx(cap / 1e6)

    # The adversarial contribution now enters the mean at the cap, not at 1e6.
    assert reduced["w"]["data"][0] == pytest.approx((0.9 + 1.0 + 1.1 + cap) / 4)
    # Bounded influence, stated as the bound: |mean| <= C.
    assert _norm(reduced) <= cap
    # And six orders of magnitude below what the attacker asked for.
    assert reduced["w"]["data"][0] < 2.0


def test_a_single_contribution_is_never_clipped():
    """Documented honest limit (design 2.3): the median of one value is that
    value, so `norm <= C` holds trivially and nothing is scaled. Correct —
    an outlier cannot be identified from a single sample — and it is why
    the existing single-contribution identity tests above still pass.
    """
    huge = _blob(w=[1e9, -1e9])
    reduced, events = reduce_deltas_with_report([(huge, 7)])
    assert events == []
    assert reduced == huge


def test_a_zero_norm_contribution_does_not_divide_by_zero():
    """A shard that converged (or a lazy node) sends an all-zero delta. Its
    norm is 0.0, and the scale factor is `cap / norm` — so the guard has to
    be written such that a zero-norm contribution is never the one being
    scaled.
    """
    contributions = [(_blob(w=[0.0, 0.0]), 10), (_blob(w=[1.0, 0.0]), 10),
                     (_blob(w=[0.0, 1.0]), 10), (_blob(w=[1e6, 0.0]), 10)]
    reduced, events = reduce_deltas_with_report(contributions)
    assert [e.index for e in events] == [3]
    assert all(math.isfinite(v) for v in reduced["w"]["data"])


def test_an_all_zero_round_clips_nothing_and_yields_no_nan():
    """Every norm 0.0 makes the cap 0.0 as well. `norm > cap` must be false
    there, or the round divides 0.0 by 0.0 and poisons the model with NaN —
    the exact outcome `require_finite` exists to prevent.
    """
    reduced, events = reduce_deltas_with_report(
        [(_blob(w=[0.0]), 10), (_blob(w=[0.0]), 20)])
    assert events == []
    assert reduced == _blob(w=[0.0])


def test_a_zero_cap_scales_an_outlier_to_zero_rather_than_to_nan():
    """A majority of zero-norm deltas puts the median — and so the cap — at
    0.0. The lone mover is then scaled by `0.0 / 5.0`, i.e. removed. That is
    the median-anchored rule working at its extreme, not a bug, and the
    assertion here is only that it produces finite zeros rather than NaN.
    """
    reduced, events = reduce_deltas_with_report(
        [(_blob(w=[0.0]), 10), (_blob(w=[0.0]), 10), (_blob(w=[5.0]), 10)])
    assert [(e.index, e.cap, e.scale) for e in events] == [(2, 0.0, 0.0)]
    assert reduced == _blob(w=[0.0])


def test_clip_events_identify_the_right_contributions_by_index():
    """`ClipEvent.index` is positional into the caller's own list — this
    module is pure stdlib and knows nothing about nodes, so the driver is
    what turns an index back into a machine. If the index is off by one the
    wrong volunteer gets named, which is worse than naming nobody.
    """
    contributions = [(_blob(w=[1.0]), 10), (_blob(w=[1000.0]), 10),
                     (_blob(w=[1.0]), 10), (_blob(w=[-2000.0]), 10),
                     (_blob(w=[1.0]), 10)]
    # sorted norms 1, 1, 1, 1000, 2000 -> median 1.0 -> cap 3.0
    _, events = reduce_deltas_with_report(contributions)

    assert [e.index for e in events] == [1, 3]
    assert all(isinstance(e, ClipEvent) for e in events)
    assert events[0].norm == 1000.0
    assert events[0].cap == 3.0
    assert events[0].scale == pytest.approx(3.0 / 1000.0)
    assert events[1].norm == 2000.0   # magnitude, so the sign is gone
    assert events[1].cap == 3.0
    assert events[1].scale == pytest.approx(3.0 / 2000.0)


def test_reduce_deltas_hides_the_report_from_its_existing_caller():
    """`reduce_deltas` keeps its signature and its return type. It has one
    production caller and 30+ tests above; the report is additive.
    """
    contributions = [(_blob(w=[1.0]), 10), (_blob(w=[1000.0]), 10),
                     (_blob(w=[1.0]), 10)]
    assert reduce_deltas(contributions) == \
        reduce_deltas_with_report(contributions)[0]


def test_a_larger_clip_factor_admits_what_a_smaller_one_clips():
    contributions = [(_blob(w=[1.0]), 10), (_blob(w=[4.0]), 10),
                     (_blob(w=[1.0]), 10)]
    # median 1.0: cap 3.0 clips the 4.0, cap 5.0 does not.
    assert [e.index for e in reduce_deltas_with_report(
        contributions, clip_factor=3.0)[1]] == [1]
    assert reduce_deltas_with_report(contributions, clip_factor=5.0)[1] == []


@pytest.mark.parametrize("bad", [0, 0.0, -1.0, float("nan"), float("inf"),
                                 float("-inf")])
def test_a_non_positive_or_non_finite_clip_factor_is_rejected(bad):
    """Silently disabling the cap is the worst outcome available: the round
    still reduces, still reports `clipped: []`, and the operator reads that
    as "nobody tried". `clip_factor=0` would additionally set the cap to
    zero and delete every contribution. Fail loudly instead.

    Both entry points, because `reduce_deltas` is the one with the callers.
    """
    contributions = [(_blob(w=[1.0]), 10), (_blob(w=[1.0]), 10)]
    with pytest.raises(ValueError, match="clip_factor"):
        reduce_deltas_with_report(contributions, clip_factor=bad)
    with pytest.raises(ValueError, match="clip_factor"):
        reduce_deltas(contributions, clip_factor=bad)


# -- order matters: validate, THEN clip -------------------------------------


def test_a_nan_weight_still_raises_rather_than_being_scaled_first():
    """Ordering, not politeness. Clipping reads every `data` value to take a
    norm; a NaN there makes the norm NaN, `norm > cap` false, and the
    contribution sails through unscaled to `require_finite` — or, with a NaN
    median, makes the cap NaN and disables the round's whole cap. The
    validation loop must have already rejected it.
    """
    with pytest.raises(NonFiniteWeights, match=r"contribution 1.*'w' index 1"):
        reduce_deltas_with_report([(_blob(w=[1.0, 1.0]), 10),
                                   (_blob(w=[1.0, float("nan")]), 10)])


def test_a_non_numeric_weight_still_raises_rather_than_reaching_the_norm():
    """`None * None` is a TypeError from inside the norm, not a typed
    weights error. The validation loop has to run first for the caller to
    get `NonFiniteWeights`.
    """
    with pytest.raises(NonFiniteWeights):
        reduce_deltas_with_report([({"w": {"shape": [1], "data": [None]}}, 10),
                                   (_blob(w=[1.0]), 10)])


def test_the_sample_count_guards_still_fire_before_any_clipping():
    """The documented attack — (delta=-999, n=-999) and (delta=1.0, n=1000)
    — must still be REJECTED, not clipped into something plausible. Clipping
    bounds a delta's magnitude and does nothing whatsoever about a negative
    sample weight, so a round that silently continued here would be a
    regression dressed as a defence.
    """
    with pytest.raises(ValueError, match="non-positive sample count"):
        reduce_deltas_with_report([(_blob(w=[-999.0]), -999),
                                   (_blob(w=[1.0]), 1000)])
    with pytest.raises(ValueError, match=r"contribution 1.*non-finite sample count"):
        reduce_deltas_with_report([(_blob(w=[1.0]), 10),
                                   (_blob(w=[1e9]), float("nan"))])


def test_a_shape_mismatch_still_raises_rather_than_being_clipped():
    with pytest.raises(WeightShapeMismatch):
        reduce_deltas_with_report([(_blob(w=[1.0]), 1),
                                   (_blob(w=[1.0, 2.0]), 1)])


def test_scalar_parameter_with_empty_shape():
    """A parameter with shape [] (scalar) should have exactly 1 data element."""
    # Scalar: shape [] has product 1
    scalar = {"s": {"shape": [], "data": [42.0]}}
    assert decode(encode({"s": ([], [42.0])})) == {"s": ([], [42.0])}
    assert subtract({"s": {"shape": [], "data": [5.0]}}, {"s": {"shape": [], "data": [2.0]}}) == {"s": {"shape": [], "data": [3.0]}}
    assert apply_delta({"s": {"shape": [], "data": [10.0]}}, {"s": {"shape": [], "data": [3.0]}}, scale=2.0) == {"s": {"shape": [], "data": [16.0]}}
    assert reduce_deltas([({"s": {"shape": [], "data": [7.0]}}, 5)]) == {"s": {"shape": [], "data": [7.0]}}
