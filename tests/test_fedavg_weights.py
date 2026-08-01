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
