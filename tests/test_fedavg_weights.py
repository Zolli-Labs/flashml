import pytest

from flashml_workloads.fedavg_weights import (
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


def test_scalar_parameter_with_empty_shape():
    """A parameter with shape [] (scalar) should have exactly 1 data element."""
    # Scalar: shape [] has product 1
    scalar = {"s": {"shape": [], "data": [42.0]}}
    assert decode(encode({"s": ([], [42.0])})) == {"s": ([], [42.0])}
    assert subtract({"s": {"shape": [], "data": [5.0]}}, {"s": {"shape": [], "data": [2.0]}}) == {"s": {"shape": [], "data": [3.0]}}
    assert apply_delta({"s": {"shape": [], "data": [10.0]}}, {"s": {"shape": [], "data": [3.0]}}, scale=2.0) == {"s": {"shape": [], "data": [16.0]}}
    assert reduce_deltas([({"s": {"shape": [], "data": [7.0]}}, 5)]) == {"s": {"shape": [], "data": [7.0]}}
