"""Integrator correctness on the harmonic oscillator, where the exact
answer is known: x(t) = cos t, v(t) = -sin t."""

import math

import pytest

from physics.integrators import leapfrog_integrate, leapfrog_step, rk4_integrate


def sho(t, y):
    return [y[1], -y[0]]


def rk4_error(h):
    T = 2.0
    steps = round(T / h)
    y = rk4_integrate(sho, 0.0, [1.0, 0.0], h, steps)
    return abs(y[0] - math.cos(T))


def leapfrog_error(h):
    T = 2.0
    steps = round(T / h)
    x, _ = leapfrog_integrate(lambda x: [-x[0]], [1.0], [0.0], h, steps)
    return abs(x[0] - math.cos(T))


def observed_order(err, h=0.02):
    return math.log2(err(h) / err(h / 2))


def test_rk4_is_fourth_order():
    assert observed_order(rk4_error) == pytest.approx(4.0, abs=0.15)


def test_leapfrog_is_second_order():
    assert observed_order(leapfrog_error) == pytest.approx(2.0, abs=0.1)


def test_rk4_exact_for_linear_ode():
    # dy/dt = 1  ->  y = t, integrated exactly by any consistent method
    y = rk4_integrate(lambda t, y: [1.0], 0.0, [0.0], 0.1, 10)
    assert y[0] == pytest.approx(1.0, abs=1e-14)


def test_rk4_sees_time_argument():
    # dy/dt = t  ->  y = t^2 / 2, a polynomial RK4 integrates exactly
    y = rk4_integrate(lambda t, y: [t], 0.0, [0.0], 0.25, 8)
    assert y[0] == pytest.approx(2.0, abs=1e-14)


def test_leapfrog_is_time_reversible():
    accel = lambda x: [-x[0] ** 3 - 0.3 * x[0]]  # anharmonic, still conservative
    x, v = leapfrog_integrate(accel, [1.3], [0.2], 0.01, 500)
    xb, vb = leapfrog_integrate(accel, x, [-vi for vi in v], 0.01, 500)
    assert xb[0] == pytest.approx(1.3, abs=1e-11)
    assert -vb[0] == pytest.approx(0.2, abs=1e-11)


def test_leapfrog_reuses_acceleration():
    calls = []

    def accel(x):
        calls.append(1)
        return [-x[0]]

    x, v, a = leapfrog_step(accel, [1.0], [0.0], 0.1)
    assert len(calls) == 2
    leapfrog_step(accel, x, v, 0.1, a)
    assert len(calls) == 3
