import math

import pytest

from physics.gravity import System


def test_two_body_force_is_inverse_square():
    s = System((1.0, 2.0))
    acc = s.accelerations([0.0, 0.0, 2.0, 0.0])
    assert acc == pytest.approx([2.0 / 4.0, 0.0, -1.0 / 4.0, 0.0])


def test_newtons_third_law_random_configuration(rng):
    s = System(tuple(rng.uniform(0.5, 3.0) for _ in range(5)))
    pos = [rng.uniform(-5, 5) for _ in range(10)]
    acc = s.accelerations(pos)
    fx = math.fsum(m * acc[2 * i] for i, m in enumerate(s.masses))
    fy = math.fsum(m * acc[2 * i + 1] for i, m in enumerate(s.masses))
    assert abs(fx) < 1e-12 and abs(fy) < 1e-12


def test_field_matches_body_acceleration(rng):
    # A body's acceleration is the field of all *other* bodies at its spot.
    s = System((1.0, 2.0, 3.0))
    pos = [rng.uniform(-3, 3) for _ in range(6)]
    acc = s.accelerations(pos)
    others = System((2.0, 3.0))
    ax, ay = others.field_at(pos[0], pos[1], pos[2:])
    assert (ax, ay) == pytest.approx((acc[0], acc[1]), rel=1e-12)


def test_force_is_minus_gradient_of_potential(rng):
    s = System((1.0, 1.5, 0.7), softening=0.05)
    pos = [rng.uniform(-2, 2) for _ in range(6)]
    acc = s.accelerations(pos)
    h = 1e-6
    for k in range(6):
        up, dn = list(pos), list(pos)
        up[k] += h
        dn[k] -= h
        grad = (s.potential_energy(up) - s.potential_energy(dn)) / (2 * h)
        m = s.masses[k // 2]
        assert -grad / m == pytest.approx(acc[k], rel=1e-6, abs=1e-8)


def test_softening_caps_close_encounter_force():
    hard = System((1.0, 1.0))
    soft = System((1.0, 1.0), softening=0.1)
    pos = [0.0, 0.0, 1e-4, 0.0]
    assert abs(soft.accelerations(pos)[0]) < 1e-1 < abs(hard.accelerations(pos)[0])


def test_invariant_helpers():
    s = System((1.0, 3.0))
    pos, vel = [-3.0, 0.0, 1.0, 0.0], [0.0, -1.5, 0.0, 0.5]
    assert s.center_of_mass(pos) == (0.0, 0.0)
    assert s.momentum(vel) == (0.0, 0.0)
    assert s.kinetic_energy(vel) == pytest.approx(0.5 * (2.25 + 0.75))
    assert s.potential_energy(pos) == pytest.approx(-3.0 / 4.0)
    assert s.angular_momentum(pos, vel) == pytest.approx(4.5 + 1.5)


def test_derivative_layout():
    s = System((1.0, 1.0))
    d = s.derivative(0.0, [0.0, 0.0, 1.0, 0.0, 0.1, 0.2, 0.3, 0.4])
    assert d[:4] == [0.1, 0.2, 0.3, 0.4]
    assert d[4:] == pytest.approx([1.0, 0.0, -1.0, 0.0])


@pytest.mark.parametrize("bad", [dict(masses=(1.0, 0.0)), dict(masses=(1.0,), softening=-1)])
def test_rejects_bad_parameters(bad):
    with pytest.raises(ValueError):
        System(**bad)
