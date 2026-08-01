import json

import pytest

from flashml_workloads.fedavg_weights import (
    NonFiniteWeights,
    WeightShapeMismatch,
    apply_delta,
    decode,
    encode,
    reduce_deltas,
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


def test_scalar_parameter_with_empty_shape():
    """A parameter with shape [] (scalar) should have exactly 1 data element."""
    # Scalar: shape [] has product 1
    scalar = {"s": {"shape": [], "data": [42.0]}}
    assert decode(encode({"s": ([], [42.0])})) == {"s": ([], [42.0])}
    assert subtract({"s": {"shape": [], "data": [5.0]}}, {"s": {"shape": [], "data": [2.0]}}) == {"s": {"shape": [], "data": [3.0]}}
    assert apply_delta({"s": {"shape": [], "data": [10.0]}}, {"s": {"shape": [], "data": [3.0]}}, scale=2.0) == {"s": {"shape": [], "data": [16.0]}}
    assert reduce_deltas([({"s": {"shape": [], "data": [7.0]}}, 5)]) == {"s": {"shape": [], "data": [7.0]}}
