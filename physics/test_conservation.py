"""Milestone 1: conservation laws over long integrations."""

import pytest

from physics.orbits import figure_eight, kepler_period, sun_planet_moon, two_body
from physics.simulate import simulate


def rel_energy_errors(system, frames):
    e0 = system.energy(frames[0].pos, frames[0].vel)
    return [abs((system.energy(f.pos, f.vel) - e0) / e0) for f in frames]


def test_leapfrog_energy_error_is_bounded_without_drift():
    system, pos, vel = two_body(1.0, 0.5, a=1.0, e=0.5)
    T = kepler_period(1.0, system.total_mass)
    steps_per_orbit = 1000
    orbits = 20
    frames = simulate(system, pos, vel, T / steps_per_orbit,
                      steps_per_orbit * orbits, "leapfrog", every=10)
    errs = rel_energy_errors(system, frames)
    per_orbit = steps_per_orbit // 10
    first = max(errs[: per_orbit + 1])
    last = max(errs[-per_orbit:])
    assert max(errs) < 1e-3
    # symplectic: the last orbit is no worse than the first (no secular drift)
    assert last < 1.2 * first


def test_rk4_energy_is_accurate_but_drifts():
    system, pos, vel = two_body(1.0, 0.5, a=1.0, e=0.5)
    T = kepler_period(1.0, system.total_mass)
    frames = simulate(system, pos, vel, T / 1000, 20_000, "rk4", every=100)
    errs = rel_energy_errors(system, frames)
    assert max(errs) < 1e-7
    assert errs[-1] > errs[len(errs) // 10]  # monotone-ish secular drift


@pytest.mark.parametrize("method", ["rk4", "leapfrog"])
def test_momenta_conserved_in_three_body(method):
    system, pos, vel = figure_eight()
    frames = simulate(system, pos, vel, 1e-3, 3000, method, every=500)
    L0 = system.angular_momentum(pos, vel)
    for f in frames:
        px, py = system.momentum(f.vel)
        assert abs(px) < 1e-12 and abs(py) < 1e-12
        assert system.angular_momentum(f.pos, f.vel) == pytest.approx(L0, abs=1e-10)


def test_figure_eight_energy_rk4():
    system, pos, vel = figure_eight()
    frames = simulate(system, pos, vel, 1e-3, 6326, "rk4", every=200)
    assert max(rel_energy_errors(system, frames)) < 1e-10


def test_star_planets_frame_and_energy():
    system, pos, vel = sun_planet_moon()
    assert system.center_of_mass(pos) == pytest.approx((0.0, 0.0), abs=1e-15)
    assert system.momentum(vel) == pytest.approx((0.0, 0.0), abs=1e-15)
    frames = simulate(system, pos, vel, 2e-3, 3142, "leapfrog", every=100)
    assert max(rel_energy_errors(system, frames)) < 1e-5
