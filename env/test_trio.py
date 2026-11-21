"""Milestone 4: the three-body navigation field."""

import math

import pytest

from env.navigation import TRIO, TRIO_HELDOUT, Task, evaluate, field_track, run_episode, trio_field
from physics.simulate import simulate


def zero(obs):
    return [0.0, 0.0]


def test_field_is_in_the_centre_of_mass_frame():
    system, pos, vel = trio_field()
    assert system.n == 3
    assert system.center_of_mass(pos) == pytest.approx((0.0, 0.0), abs=1e-15)
    assert system.momentum(vel) == pytest.approx((0.0, 0.0), abs=1e-15)


def test_cached_track_matches_direct_simulation():
    system, track = field_track(TRIO)
    assert len(track) == TRIO.max_steps + 1
    s2, pos, vel = trio_field()
    frames = simulate(s2, pos, vel, TRIO.dt, TRIO.max_steps, "leapfrog")
    assert list(track[-1]) == frames[-1].pos
    assert field_track(TRIO) is field_track(TRIO)  # cached


def test_planets_stay_bound_on_perturbed_orbits():
    # The planets are heavy enough (0.05, 0.02) to perturb each other:
    # the outer orbit swings between ~1.8 and ~2.0. Bound, not circular.
    system, track = field_track(TRIO)
    for pos in track[::20]:
        r1 = math.hypot(pos[2] - pos[0], pos[3] - pos[1])
        r2 = math.hypot(pos[4] - pos[0], pos[5] - pos[1])
        assert r1 == pytest.approx(1.0, rel=0.08)
        assert 1.6 < r2 < 2.35


def test_work_bookkeeping_balances_kinetic_energy():
    """sum(work by each body) + thrust work == change in kinetic energy,
    up to the O(dt^2) KDK residual. Checked with thrust on, in the field."""
    def wobble(obs):
        return [0.3 * math.sin(7 * obs[0]), 0.4 * math.cos(5 * obs[1])]

    res = run_episode(TRIO, wobble, TRIO.targets[0])
    assert res.steps > 50 and res.thrust_work != 0.0
    v0, v1 = TRIO.start_vel, res.final_vel
    dke = 0.5 * (v1[0] ** 2 + v1[1] ** 2) - 0.5 * (v0[0] ** 2 + v0[1] ** 2)
    assert sum(res.work) + res.thrust_work == pytest.approx(dke, abs=1e-3)


def test_thrust_work_accounts_for_a_burn():
    task = Task(**{**TRIO.__dict__, "field": None, "max_steps": 40})
    res = run_episode(task, lambda o: [1.0, 0.0], task.targets[0], record=True)
    v0 = task.start_vel
    vx = v0[0] + task.thrust_max * task.dt * res.steps
    vy = v0[1]
    ke_change = 0.5 * (vx * vx + vy * vy) - 0.5 * (v0[0] ** 2 + v0[1] ** 2)
    assert res.work == []
    assert res.thrust_work == pytest.approx(ke_change, rel=1e-9)


def test_probe_motion_is_chaotic():
    """Two coasting probes 1e-9 apart end up macroscopically apart within
    40 time units: the field the agent flies through is chaotic."""
    a = Task(**{**TRIO.__dict__, "targets": ((99.0, 99.0),), "bounds": 50.0,
                    "max_steps": 1600, "body_radius": 0.0})  # no crash: softening keeps it smooth
    b = Task(**{**a.__dict__, "start": (0.6 + 1e-9, 0.0)})
    ra = run_episode(a, zero, (99.0, 99.0))
    rb = run_episode(b, zero, (99.0, 99.0))
    assert math.dist(ra.final_pos, rb.final_pos) > 1e-3   # ~10^8 amplification


def test_naive_controllers_fail_in_the_field():
    def pd(obs):
        dx, dy, vx, vy = obs[0] * 3.0, obs[1] * 3.0, obs[2] * 1.5, obs[3] * 1.5
        return [3 * dx - 3 * vx, 3 * dy - 3 * vy]
    assert evaluate(TRIO, pd) < 0.2
    assert sum(run_episode(TRIO, pd, t).reached for t in TRIO.targets) <= 1
    assert len(TRIO_HELDOUT) == 6
