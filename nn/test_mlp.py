import math

import pytest

from nn.mlp import MLP, init_params, param_count


def test_param_count():
    assert param_count([3, 4, 2]) == 3 * 4 + 4 + 4 * 2 + 2
    assert param_count([1, 1]) == 2


def test_hand_set_weights():
    # 2 -> 1, linear: y = 2*x0 - x1 + 0.5
    net = MLP([2, 1], [2.0, -1.0, 0.5], output="linear")
    assert net([3.0, 1.0]) == [5.5]


def test_two_layer_known_output():
    # 1 -> 2 (relu) -> 1 (linear)
    net = MLP([1, 2, 1], [1.0, -1.0, 0.0, 0.0, 1.0, 1.0, 0.0], hidden="relu", output="linear")
    assert net([2.0]) == [2.0]
    assert net([-3.0]) == [3.0]   # |x|


def test_tanh_output_is_bounded(rng):
    net = MLP.random([4, 16, 2], rng)
    big = [rng.uniform(-100, 100) for _ in range(4)]
    assert all(-1.0 <= v <= 1.0 for v in net(big))


def test_params_round_trip(rng):
    p = init_params([5, 3, 2], rng)
    assert MLP([5, 3, 2], p).params == p


def test_seeded_init_is_deterministic():
    import random
    a = init_params([4, 8, 2], random.Random(7))
    b = init_params([4, 8, 2], random.Random(7))
    c = init_params([4, 8, 2], random.Random(8))
    assert a == b != c


def test_init_scale(rng):
    p = init_params([50, 50], rng)
    weights = p[:2500]
    std = math.sqrt(sum(w * w for w in weights) / len(weights))
    assert std == pytest.approx(math.sqrt(2 / 100), rel=0.1)
    assert p[2500:] == [0.0] * 50


def test_save_load(tmp_path, rng):
    net = MLP.random([3, 4, 2], rng, hidden="relu")
    path = tmp_path / "w.json"
    net.save(path, note="hi")
    back = MLP.load(path)
    assert back.params == net.params and back.hidden == "relu"
    assert back([0.1, 0.2, 0.3]) == net([0.1, 0.2, 0.3])


@pytest.mark.parametrize("sizes,n", [([2], 0), ([2, 0], 0), ([2, 1], 2)])
def test_rejects_bad_shapes(sizes, n):
    with pytest.raises(ValueError):
        MLP(sizes, [0.0] * n if n != 2 else [0.0] * 5)


def test_rejects_bad_input_length_and_activation():
    net = MLP([2, 1], [1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        net([1.0])
    with pytest.raises(ValueError):
        MLP([2, 1], [1.0, 1.0, 0.0], hidden="sigmoid")
