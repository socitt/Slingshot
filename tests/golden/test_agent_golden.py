"""Golden trained agents: stored weights must still score what they
scored when trained, and must actually be good at the task."""

import pytest

from env.agent import policy_from_genome
from env.navigation import FREE, FREE_HELDOUT, evaluate, run_episode
from nn.mlp import MLP
from tests.golden.agent_cases import CASES, load_golden

PD_BASELINE = 1.04  # hand-written PD controller on FREE (see env tests)


@pytest.fixture(scope="module")
def free():
    g = load_golden("free")
    return g, MLP.from_dict(g["mlp"])


def test_spec_matches_case(free):
    g, _ = free
    assert {k: g[k] for k in CASES["free"]} == CASES["free"]


def test_stored_agent_rescores_exactly(free):
    g, net = free
    assert evaluate(FREE, net) == pytest.approx(g["score"], abs=1e-9)
    assert evaluate(FREE, net, FREE_HELDOUT) == pytest.approx(g["heldout_score"], abs=1e-9)


def test_free_agent_beats_the_pd_baseline_and_generalises(free):
    g, net = free
    assert g["score"] > PD_BASELINE
    assert g["heldout_score"] > 1.0
    assert all(run_episode(FREE, net, t).reached for t in FREE.targets + FREE_HELDOUT)


def test_training_curve_went_up(free):
    g, _ = free
    hist = g["history_best"]
    assert len(hist) == g["generations"]
    assert max(hist[-10:]) > hist[0] + 0.5


def test_weights_load_through_the_agent_path(free):
    g, net = free
    assert policy_from_genome(g["mlp"]["params"]).params == net.params


# --- milestone 4: agents in the three-body field ---------------------
#
# Field episodes are mildly chaotic (lambda ~ 1 over T = 8), so a
# platform ulp difference can grow ~1e3x: compare scores to 1e-6, not
# 1e-9. Reached flags could only flip for a trajectory grazing the
# capture radius; test_no_episode_grazes_the_capture_radius pins the
# margin (~2e-4 when generated) far above any amplified rounding error.

from env.navigation import ASSIST, HELDOUT, TRIO  # noqa: E402

ZERO_THRUST = {"trio": 0.191, "assist": 0.191}


@pytest.mark.parametrize("name,task", [("trio", TRIO), ("assist", ASSIST)])
def test_field_agent_rescores(name, task):
    g = load_golden(name)
    net = MLP.from_dict(g["mlp"])
    assert {k: g[k] for k in CASES[name]} == CASES[name]
    assert evaluate(task, net) == pytest.approx(g["score"], abs=1e-6)
    assert evaluate(task, net, HELDOUT[name]) == pytest.approx(g["heldout_score"], abs=1e-6)
    assert [run_episode(task, net, t).reached for t in task.targets] == g["reached"]


def test_trio_agent_navigates_the_field():
    g = load_golden("trio")
    n = len(g["reached"])
    assert n == 24
    assert sum(g["reached"]) >= 18           # of 24 training targets
    assert g["score"] > ZERO_THRUST["trio"] + 0.5
    # Generalisation: 12 seeded random held-out targets (random radius
    # too). The 6-target predecessor reached 1/6 of its held-out set.
    assert sum(g["heldout_reached"]) >= 6
    assert g["heldout_score"] > 0.6


def test_assist_agent_beats_doing_nothing_and_uses_a_planet():
    g = load_golden("assist")
    assert g["score"] > ZERO_THRUST["assist"] + 0.1
    assert sum(g["reached"]) >= 3
    # at least one successful episode got real energy from a planet
    # (bodies 1 and 2; body 0 is the star)
    assists = [max(w[1:]) for w, ok in zip(g["work"], g["reached"]) if ok]
    assert max(assists) > 0.3


@pytest.mark.parametrize("name,task", [("trio", TRIO), ("assist", ASSIST)])
def test_no_episode_grazes_the_capture_radius(name, task):
    import math
    net = MLP.from_dict(load_golden(name)["mlp"])
    margin = math.inf
    for t in task.targets + HELDOUT[name]:
        res = run_episode(task, net, t, record=True)
        margin = min([margin] + [abs(math.dist(f["probe"], t) - task.target_radius)
                                 for f in res.frames])
    assert margin > 1e-7


@pytest.mark.parametrize("name,task", [("trio", TRIO), ("assist", ASSIST)])
def test_golden_episodes_are_not_hypersensitive(name, task):
    """Platform-independent version of the rescoring tolerance: nudge
    the probe's start by 1e-12 (far above any ulp-level platform
    difference) and every stored episode's score must move < 1e-6.
    Random genomes can fail this (chaotic loitering); the goldens don't."""
    import dataclasses
    net = MLP.from_dict(load_golden(name)["mlp"])
    nudged = dataclasses.replace(task, start=(task.start[0] + 1e-12, task.start[1]))
    for t in task.targets + HELDOUT[name]:
        a, b = run_episode(task, net, t), run_episode(nudged, net, t)
        assert abs(a.score - b.score) < 1e-6, t
