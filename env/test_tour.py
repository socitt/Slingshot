"""Milestone 8: the grand tour (catch several planets in order)."""

import dataclasses
import math

import pytest

from env.navigation import (FREE, TOUR, TOUR_HELDOUT, Tour, episode_score, evaluate, field_track,
                            grand_field, run_episode, target_at)


def zero(obs):
    return [0.0, 0.0]


def test_grand_field_has_a_star_and_four_planets():
    system, pos, vel = grand_field()
    assert system.n == 5 and system.masses[0] == max(system.masses)
    radii = [math.hypot(pos[2 * j] - pos[0], pos[2 * j + 1] - pos[1]) for j in range(1, 5)]
    assert radii == pytest.approx([0.9, 1.4, 2.0, 2.6], abs=0.01)


def test_tour_is_not_a_point_and_needs_stops():
    with pytest.raises(TypeError):
        x, y = Tour((2, 3), 0)
    with pytest.raises(ValueError):
        Tour(())


def test_target_follows_the_current_stop():
    t = Tour((2, 3, 4), 36)
    _, track = field_track(TOUR)
    for stop, body in enumerate(t.stops):
        assert target_at(TOUR, t, 5, stop) == track[41][2 * body: 2 * body + 2]


def test_catching_a_stop_moves_the_target_to_the_next_one():
    """Teleport-start the probe next to stop 1's planet: it catches stop 1
    at step 1, then the observation and frames point at stop 2."""
    t = Tour((2, 3), 0)
    x, y = target_at(TOUR, t, 1, 0)
    _, track = field_track(TOUR)
    # move with planet B (finite-difference velocity) so the probe doesn't crash into it
    vx = (track[2][4] - track[0][4]) / (2 * TOUR.dt)
    vy = (track[2][5] - track[0][5]) / (2 * TOUR.dt)
    # 0.08 away: inside the 0.12 capture radius, outside the 0.05 crash radius
    task = dataclasses.replace(TOUR, start=(x + 0.08, y), start_vel=(vx, vy), max_steps=5)
    seen = []

    def spy(obs):
        seen.append((obs[0], obs[1]))
        return [0.0, 0.0]

    res = run_episode(task, spy, t, record=True)
    assert res.stops_done == 1 and res.stop_steps == [1] and not res.reached
    assert res.steps >= 3   # (planet B then pulls the parked probe in: a crash at step 4 is physics)
    assert [f["stop"] for f in res.frames[:3]] == [0, 1, 1]
    # after the catch the target offset points at planet C, not B
    cx, cy = track[2][6:8]
    p = res.frames[2]["probe"]
    assert seen[2] == pytest.approx(((cx - p[0]) / task.pos_scale, (cy - p[1]) / task.pos_scale))


def test_completing_the_tour():
    t = Tour((2,), 0)                       # a one-stop tour is a rendezvous
    x, y = target_at(TOUR, t, 1, 0)
    task = dataclasses.replace(TOUR, start=(x + 0.01, y), start_vel=(0.0, 0.0))
    res = run_episode(task, zero, t)
    assert res.reached and res.stops_done == 1 and res.steps == 1


def test_score_counts_stops_and_matches_the_old_formula_for_one_target():
    t = FREE
    assert episode_score(t, True, False, 10, 0.5, 0.0, 1.0) == episode_score(
        t, True, False, 10, 0.5, 0.0, 1.0, stops_done=1)
    assert episode_score(t, False, False, 100, 0.5, 0.2, 1.0) == episode_score(
        t, False, False, 100, 0.5, 0.2, 1.0, stops_done=0)
    two_of_three = episode_score(TOUR, False, False, 800, 0.0, 0.5, 1.0, stops_done=2)
    one_of_three = episode_score(TOUR, False, False, 800, 0.0, 0.5, 1.0, stops_done=1)
    assert two_of_three == pytest.approx(one_of_three + 1.0)


def test_doing_nothing_rarely_catches_anything():
    # one launch of 36 drifts through B and C by luck; the rest catch nothing
    lucky = [t for t in TOUR.targets if run_episode(TOUR, zero, t).stops_done > 0]
    assert len(lucky) <= 1
    assert evaluate(TOUR, zero) < 0.2


def test_five_body_field_is_more_chaotic_than_the_three_body_one():
    """A 1e-15 nudge to the start changes whole outcomes for a random
    network within 800 steps (seen: 5 of 36 launches for this seed)."""
    import random
    from env.agent import SIZES, policy_from_genome
    from nn.mlp import init_params
    net = policy_from_genome(init_params(SIZES, random.Random(5)))
    nudged = dataclasses.replace(TOUR, start=(TOUR.start[0] + 1e-15, TOUR.start[1]))
    diffs = [abs(run_episode(TOUR, net, t).score - run_episode(nudged, net, t).score)
             for t in TOUR.targets]
    assert max(diffs) > 0.01      # an O(1)-sized change from a 1e-15 cause


def test_held_out_delays_are_new():
    assert not {t.delay for t in TOUR_HELDOUT} & {t.delay for t in TOUR.targets}
    assert len(TOUR_HELDOUT) == 12
