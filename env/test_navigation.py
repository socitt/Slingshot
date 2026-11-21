import math

import pytest

from env.navigation import FREE, OBS_SIZE, Task, episode_score, evaluate, observe, ring, run_episode
from physics.orbits import two_body


def zero(obs):
    return [0.0, 0.0]


def pd(obs):
    """Hand-written proportional-derivative controller: proves the task
    is solvable and gives a baseline score to compare agents against."""
    dx, dy, vx, vy = obs[0] * 2.0, obs[1] * 2.0, obs[2], obs[3]
    return [3 * dx - 3 * vx, 3 * dy - 3 * vy]


def test_zero_policy_drifts_in_a_straight_line():
    task = Task("t", targets=((5.0, 5.0),), start_vel=(0.3, -0.1), max_steps=20)
    res = run_episode(task, zero, task.targets[0], record=True)
    end = res.frames[-1]["probe"]
    assert end == pytest.approx([0.3 * 20 * task.dt, -0.1 * 20 * task.dt])
    assert res.fuel_used == 0.0


def test_constant_thrust_matches_kinematics():
    # leapfrog is exact for constant acceleration: x = a t^2 / 2
    task = Task("t", targets=((9.0, 0.0),), max_steps=40, fuel=100.0, bounds=50.0)
    res = run_episode(task, lambda o: [1.0, 0.0], task.targets[0], record=True)
    t = 40 * task.dt
    assert res.frames[-1]["probe"][0] == pytest.approx(0.5 * task.thrust_max * t * t)
    assert res.fuel_used == pytest.approx(task.thrust_max * t)


def test_thrust_is_clipped_to_a_disc():
    task = Task("t", targets=((9.0, 9.0),), max_steps=10, fuel=100.0)
    res = run_episode(task, lambda o: [5.0, 5.0], task.targets[0])
    assert res.fuel_used == pytest.approx(task.thrust_max * 10 * task.dt)


def test_fuel_runs_out():
    task = Task("t", targets=((9.0, 0.0),), max_steps=100, fuel=0.3, bounds=50.0)
    res = run_episode(task, lambda o: [1.0, 0.0], task.targets[0], record=True)
    assert res.fuel_used == pytest.approx(0.3)
    # after the burn the probe coasts at the 0.3 delta-v it bought
    xs = [f["probe"][0] for f in res.frames[-3:]]
    assert xs[2] - xs[1] == pytest.approx(0.3 * task.dt)


def test_pd_controller_reaches_every_target():
    for target in FREE.targets:
        res = run_episode(FREE, pd, target)
        assert res.reached and not res.crashed
    assert evaluate(FREE, pd) > 1.0


def test_scores_order_outcomes():
    t = FREE
    reached_fast = episode_score(t, True, False, 10, 0.5, 0.0, 1.0)
    reached_slow = episode_score(t, True, False, 90, 0.5, 0.0, 1.0)
    near_miss = episode_score(t, False, False, 100, 0.5, 0.2, 1.0)
    far_miss = episode_score(t, False, False, 100, 0.5, 0.9, 1.0)
    crashed = episode_score(t, False, True, 50, 0.5, 0.2, 1.0)
    assert reached_fast > reached_slow > near_miss > far_miss
    assert crashed < near_miss
    assert evaluate(FREE, zero) == 0.0


def test_out_of_bounds_ends_episode():
    task = Task("t", targets=((0.0, 3.0),), start_vel=(4.0, 0.0), bounds=1.0)
    res = run_episode(task, zero, task.targets[0])
    assert res.out_of_bounds and res.steps < task.max_steps


def test_field_bodies_move_and_pull_the_probe():
    field = lambda: two_body(1.0, 1e-3, 1.0, 0.0)
    task = Task("g", targets=((3.0, 3.0),), start=(0.0, 2.0), max_steps=10, field=field)
    res = run_episode(task, zero, task.targets[0], record=True)
    first, last = res.frames[0], res.frames[-1]
    assert first["bodies"] != last["bodies"]
    assert last["probe"][1] < 2.0       # fell toward the star at the origin


def test_crash_into_a_body():
    field = lambda: two_body(1.0, 1e-3, 1.0, 0.0)
    task = Task("g", targets=((3.0, 3.0),), start=(0.0, 0.3), max_steps=200, field=field)
    res = run_episode(task, zero, task.targets[0])
    assert res.crashed
    assert res.score < 0


def test_observation_layout():
    obs = observe(FREE, [0.0, 0.0], [0.5, 0.0], (1.0, -1.0), (0.5, 0.0), (0.0, 0.0), 1.0)
    assert len(obs) == OBS_SIZE
    assert obs[:4] == [0.5, -0.5, 0.5, 0.0]
    assert obs[4] == pytest.approx(0.5)     # soft-squashed gravity
    assert obs[8] == pytest.approx(0.5)     # half the fuel left


def test_ring():
    pts = ring(4, 2.0)
    assert pts[1] == pytest.approx((0.0, 2.0), abs=1e-12)
    assert all(math.hypot(*p) == pytest.approx(2.0) for p in pts)
