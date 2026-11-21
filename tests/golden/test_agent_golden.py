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
