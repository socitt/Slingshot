"""Milestone 7: rendezvous with a moving planet."""

import dataclasses

import pytest

from env.navigation import (RENDEZVOUS, RENDEZVOUS_HELDOUT, Rendezvous, evaluate, field_track,
                            run_episode, target_at)


def zero(obs):
    return [0.0, 0.0]


def test_rendezvous_is_not_a_point():
    # a fixed target unpacks as (x, y); a Rendezvous must never do so silently
    with pytest.raises(TypeError):
        x, y = Rendezvous(2, 10)


def test_target_follows_the_body_with_delay():
    _, track = field_track(RENDEZVOUS)
    assert len(track) == RENDEZVOUS.max_steps + RENDEZVOUS.track_extra + 1
    t = Rendezvous(2, 55)
    assert target_at(RENDEZVOUS, t, 0) == track[55][4:6]
    assert target_at(RENDEZVOUS, t, 10) == track[65][4:6]
    assert target_at(RENDEZVOUS, (1.0, 2.0), 99) == (1.0, 2.0)


def test_delay_shifts_the_whole_field():
    res = run_episode(RENDEZVOUS, zero, Rendezvous(2, 110), record=True)
    _, track = field_track(RENDEZVOUS)
    assert res.frames[0]["bodies"] == list(track[110])
    assert res.frames[5]["target"] == track[115][4:6]


def test_delays_cover_one_orbit_and_held_out_ones_interleave():
    delays = [t.delay for t in RENDEZVOUS.targets]
    held = [t.delay for t in RENDEZVOUS_HELDOUT]
    assert max(held) <= RENDEZVOUS.track_extra
    assert not set(delays) & set(held)
    # the planet really is somewhere different at each launch
    starts = {tuple(round(c, 2) for c in target_at(RENDEZVOUS, t, 0)) for t in RENDEZVOUS.targets}
    assert len(starts) == len(delays)


def test_catching_the_planet_counts_as_reaching():
    t = Rendezvous(2, 0)
    x, y = target_at(RENDEZVOUS, t, 1)
    _, track = field_track(RENDEZVOUS)
    # start the probe right where the planet will be after one step
    task = dataclasses.replace(RENDEZVOUS, start=(x + 0.01, y), start_vel=(0.0, 0.0))
    res = run_episode(task, zero, t)
    assert res.reached and res.steps == 1 and not res.crashed


def test_delay_beyond_the_cached_history_is_rejected():
    with pytest.raises(ValueError, match="track_extra"):
        run_episode(RENDEZVOUS, zero, Rendezvous(2, RENDEZVOUS.track_extra + 1))


def test_doing_nothing_never_catches_it():
    assert not any(run_episode(RENDEZVOUS, zero, t).reached for t in RENDEZVOUS.targets)
    assert evaluate(RENDEZVOUS, zero) < 0.2
