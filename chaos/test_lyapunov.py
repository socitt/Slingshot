"""Milestone 2: Lyapunov exponents against known values."""

import math

import pytest

from chaos.logistic import logistic, logistic_deriv
from chaos.lorenz import divergence, lorenz, lorenz_jacobian
from chaos.lyapunov import gram_schmidt, lyapunov_map, lyapunov_spectrum, lyapunov_two_trajectory
from physics.integrators import rk4_step
from physics.orbits import binary_with_probe, two_body


def logistic_lambda(r, x0=0.123456789, n=200_000):
    return lyapunov_map(lambda x: logistic(r, x), lambda x: logistic_deriv(r, x), x0, n)


def test_logistic_r4_is_ln2():
    assert logistic_lambda(4.0) == pytest.approx(math.log(2), abs=2e-3)


@pytest.mark.parametrize("x0", [0.1, 0.3, 0.77])
def test_logistic_r4_independent_of_seed(x0):
    assert logistic_lambda(4.0, x0, 50_000) == pytest.approx(math.log(2), abs=1e-2)


def test_logistic_period2_exponent_is_analytic():
    # On the 2-cycle, lambda = 0.5 ln|f'(x1) f'(x2)| = 0.5 ln|4 + 2r - r^2|
    r = 3.2
    assert logistic_lambda(r, n=2000) == pytest.approx(0.5 * math.log(abs(4 + 2 * r - r * r)), abs=1e-9)


def test_superstable_orbit_gives_minus_infinity():
    assert lyapunov_map(lambda x: logistic(2.0, x), lambda x: logistic_deriv(2.0, x),
                        0.5, 10, transient=0) == -math.inf


def test_signs_across_regimes():
    assert logistic_lambda(2.9, n=5000) < 0       # fixed point
    assert logistic_lambda(3.83, n=5000) < 0      # period-3 window
    assert logistic_lambda(3.7, n=50_000) > 0.2   # chaos


def test_gram_schmidt():
    q, norms = gram_schmidt([[3.0, 0.0], [1.0, 2.0]])
    assert norms == pytest.approx([3.0, 2.0])
    assert q[0] == pytest.approx([1.0, 0.0])
    assert q[1] == pytest.approx([0.0, 1.0])
    with pytest.raises(ArithmeticError):
        gram_schmidt([[1.0, 1.0], [2.0, 2.0]])


@pytest.fixture(scope="module")
def lorenz_spectrum():
    return lyapunov_spectrum(lorenz(), lorenz_jacobian(), [1.0, 1.0, 1.0],
                             h=0.01, steps=50_000, renorm_every=10, transient_steps=1000)


def test_lorenz_spectrum_known_values(lorenz_spectrum):
    l1, l2, l3 = lorenz_spectrum
    assert l1 == pytest.approx(0.906, abs=0.04)
    assert l2 == pytest.approx(0.0, abs=0.02)
    assert l3 == pytest.approx(-14.572, abs=0.05)


def test_lorenz_exponents_sum_to_trace(lorenz_spectrum):
    # an invariant: exact for the flow, RK4 gets it to many digits
    assert sum(lorenz_spectrum) == pytest.approx(divergence(), abs=1e-3)


def test_two_trajectory_agrees_with_variational_on_lorenz(lorenz_spectrum):
    f = lorenz()
    lam = lyapunov_two_trajectory(lambda y: rk4_step(f, 0.0, y, 0.01), [1.0, 1.0, 1.0],
                                  0.01, 50_000, renorm_every=10)
    assert lam == pytest.approx(lorenz_spectrum[0], abs=0.05)


def _probe_lambda(system, pos, vel, T, h=2e-3, k=None):
    d = [0.0] * len(pos + vel)
    d[k if k is not None else 0] = 1.0
    f = system.derivative
    return lyapunov_two_trajectory(lambda y: rk4_step(f, 0.0, y, h), pos + vel, h,
                                   int(T / h), renorm_every=25, direction=d)


def test_kepler_orbit_is_regular():
    # Regular motion separates linearly, so lambda(T) ~ log(T)/T -> 0.
    system, pos, vel = two_body(1.0, 0.5, 1.0, 0.3)
    l40 = _probe_lambda(system, pos, vel, 40)
    l80 = _probe_lambda(system, pos, vel, 80)
    assert l80 < 0.65 * l40 < 0.15


def test_probe_near_binary_is_chaotic_far_probe_is_not():
    near = _probe_lambda(*binary_with_probe(1.3), T=80, k=4)
    far = _probe_lambda(*binary_with_probe(4.0), T=80, k=4)
    assert near > 0.3
    assert far < 0.1
    assert near > 5 * far


def test_uniform_perturbation_is_a_boost_not_chaos():
    # Regression guard for a mistake made while writing this: an all-ones
    # perturbation shifts every velocity equally -- a Galilean boost --
    # so the separation grows identically for *any* system, and the
    # "exponent" says nothing about the dynamics.
    def ones_lambda(system, pos, vel):
        f = system.derivative
        return lyapunov_two_trajectory(lambda y: rk4_step(f, 0.0, y, 2e-3), pos + vel,
                                       2e-3, 10_000, renorm_every=25)

    kepler = ones_lambda(*two_body(1.0, 0.5, 1.0, 0.3))
    chaotic = ones_lambda(*binary_with_probe(1.3))
    assert kepler == pytest.approx(chaotic, rel=0.05)
