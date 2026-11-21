import json

import pytest

from env.navigation import FREE, RENDEZVOUS, Rendezvous
from render.vizdata import AGENTS, agent_json, chaos_json, episode_json, learning_json, lirk_log


def test_learning_replay_is_the_real_training_run():
    """The first generations of the replay must score exactly what the
    golden agent's own training recorded: same spec, same seed, same RNG."""
    golden = json.loads((AGENTS / "agent_free.json").read_text())
    rep = learning_json("free", sample_gens=[0, 4], swarm=True, generations=5)
    assert [b for b, _ in rep["curve"]] == [round(x, 3) for x in golden["history_best"][:5]]
    assert [s["gen"] for s in rep["samples"]] == [0, 4]
    assert len(rep["samples"][0]["flights"]) == golden["pop"]
    best = max(f["fitness"] for f in rep["samples"][1]["flights"])
    assert best == rep["samples"][1]["best"]


def test_best_only_learning_in_a_field_carries_the_field():
    rep = learning_json("assist", sample_gens=[0, 1], swarm=False, generations=2, pop=4)
    assert all(len(s["flights"]) == 1 for s in rep["samples"])
    assert len(rep["field"]) > 10 and len(rep["field"][0]) == 6
    assert rep["masses"][0] == max(rep["masses"])


def test_episode_json_shapes():
    ep = episode_json(FREE, lambda o: [1.0, 0.0], FREE.targets[0], every=3)
    assert len(ep["p"]) == len(ep["th"]) and "b" not in ep
    rv = episode_json(RENDEZVOUS, lambda o: [0.0, 0.0], Rendezvous(2, 55), every=5)
    assert rv["target"] == {"body": 2, "delay": 55}
    assert len(rv["tp"]) == len(rv["p"]) == len(rv["b"])


def test_agent_and_chaos_sections():
    a = agent_json("free")
    assert len(a["train"]) == 12 and a["score"] > 1.0
    c = chaos_json()
    assert c["divergence"][0][1] == pytest.approx(-9, abs=0.1)
    assert c["divergence"][-1][1] > -3          # the twins have separated
    assert len(c["lyapunov"]["lorenz"]) == 3


def test_lirk_log_parsing(tmp_path):
    p = tmp_path / "log"
    p.write_text("2026-11-21T08:10:00Z rc=1 secs=2 lirk test //...\njunk line\n")
    assert lirk_log(p) == [["08:10", 1, 2, "test"]]
    assert lirk_log(tmp_path / "missing") == []
