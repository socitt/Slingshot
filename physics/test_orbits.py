"""Milestone 1: measured orbital periods against Kepler's third law."""

import math

import pytest

from physics.orbits import FIGURE_EIGHT_PERIOD, figure_eight, kepler_period, two_body
from physics.simulate import simulate


def measured_period(frames):
    """Time of the first upward crossing of the x-axis by the relative
    vector r = body1 - body0, after starting at periapsis on +x...
    Uses linear interpolation between frames."""
    def angle(f):
        return math.atan2(f.pos[3] - f.pos[1], f.pos[2] - f.pos[0])

    prev = frames[0]
    for f in frames[1:]:
        a0, a1 = angle(prev), angle(f)
        # crossing from just below the +x axis (angle < 0) to above it
        if a0 < 0 <= a1 and abs(a1 - a0) < 1.0:
            return prev.t + (f.t - prev.t) * (-a0) / (a1 - a0)
        prev = f
    raise AssertionError("no full orbit found")


@pytest.mark.parametrize("m1,m2,a,e", [
    (1.0, 1e-3, 1.0, 0.0),
    (1.0, 1.0, 2.0, 0.0),
    (1.0, 0.5, 1.0, 0.6),
])
def test_two_body_period_matches_kepler(m1, m2, a, e):
    system, pos, vel = two_body(m1, m2, a, e)
    T = kepler_period(a, system.total_mass)
    frames = simulate(system, pos, vel, T / 2000, 2200, "rk4")
    assert measured_period(frames) == pytest.approx(T, rel=1e-6)


def test_leapfrog_period_also_matches():
    system, pos, vel = two_body(1.0, 0.3, 1.5, 0.2)
    T = kepler_period(1.5, system.total_mass)
    frames = simulate(system, pos, vel, T / 4000, 4400, "leapfrog")
    assert measured_period(frames) == pytest.approx(T, rel=1e-5)


def test_circular_orbit_keeps_its_radius():
    system, pos, vel = two_body(1.0, 1e-3, 1.0, 0.0)
    frames = simulate(system, pos, vel, 1e-2, 1000, "leapfrog", every=50)
    for f in frames:
        r = math.hypot(f.pos[2] - f.pos[0], f.pos[3] - f.pos[1])
        assert r == pytest.approx(1.0, abs=1e-4)


def test_figure_eight_returns_after_one_period():
    system, pos, vel = figure_eight()
    h = FIGURE_EIGHT_PERIOD / 6000
    frames = simulate(system, pos, vel, h, 6000, "rk4", every=6000)
    end = frames[-1]
    assert end.t == pytest.approx(FIGURE_EIGHT_PERIOD)
    assert max(abs(a - b) for a, b in zip(end.pos, pos)) < 1e-6
    assert max(abs(a - b) for a, b in zip(end.vel, vel)) < 1e-6
