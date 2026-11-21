"""Training smoke test: a short seeded run must learn something, and be
exactly reproducible. Runs under a lirk per-target timeout."""

import pytest

from env.agent import policy_from_genome
from env.navigation import FREE, Task, evaluate
from env.training import load_agent, save_agent, train

SMALL = Task(name="small", targets=FREE.targets[:4])


@pytest.fixture(scope="module")
def run_a():
    return train(SMALL, "ga", generations=8, pop_size=12, seed=42)


def test_training_improves_over_the_initial_policy(run_a):
    first, last = run_a.history[0], run_a.history[-1]
    assert run_a.score > first.best
    assert last.mean > first.mean
    assert run_a.score > 0.2


def test_training_is_reproducible(run_a):
    again = train(SMALL, "ga", generations=8, pop_size=12, seed=42)
    assert again.genome == run_a.genome
    assert again.score == run_a.score


def test_es_trains_too():
    res = train(SMALL, "es", generations=6, pop_size=12, seed=42)
    assert res.score > res.history[0].best


def test_best_genome_rescores_to_reported_score(run_a):
    assert evaluate(SMALL, policy_from_genome(run_a.genome)) == run_a.score


def test_save_load_round_trip(run_a, tmp_path):
    path = tmp_path / "agent.json"
    save_agent(path, run_a.genome, task="small", seed=42)
    policy, meta = load_agent(path)
    assert meta == {"task": "small", "seed": 42}
    assert evaluate(SMALL, policy) == run_a.score


def test_rejects_bad_arguments():
    with pytest.raises(ValueError):
        train(SMALL, "sgd")
    with pytest.raises(ValueError):
        train(SMALL, init=[0.0])
