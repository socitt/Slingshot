"""numpy backend parity with the pure-Python backend.

Deliberately NOT a module-level ``pytest.importorskip``: on a machine
without numpy (iSH) that skips the whole module, pytest exits 5 ("no
tests ran") and lirk reports the target as FAILED -- see
docs/lirk-notes.md #13. Instead each numpy test skips itself, and the
first test always runs.
"""

import json
import random
from pathlib import Path

import pytest

from env.agent import SIZES, policy_from_genome
from env.backend import numpy_available
from env.navigation import ASSIST, FREE, FREE_HELDOUT, TRIO, evaluate
from env.training import train
from nn.mlp import init_params

needs_numpy = pytest.mark.skipif(not numpy_available(), reason="numpy not installed")
GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "data"


def test_backend_module_is_optional():
    """Always runs: the core must import and train without numpy."""
    import env.navigation
    import env.training
    assert "numpy" not in env.navigation.__dict__
    assert "numpy" not in env.training.__dict__


def genomes(n, seed=5):
    rng = random.Random(seed)
    return [init_params(SIZES, rng) for _ in range(n)]


@needs_numpy
@pytest.mark.parametrize("task,tol", [(FREE, 1e-12), (TRIO, 1e-7), (ASSIST, 1e-7)])
def test_scores_match_python_backend(task, tol):
    from npbackend.batch import evaluate_population
    gs = genomes(12)
    py = [evaluate(task, policy_from_genome(g)) for g in gs]
    npy = evaluate_population(task, gs, SIZES)
    assert npy == pytest.approx(py, abs=tol)


@needs_numpy
@pytest.mark.parametrize("name,task", [("free", FREE), ("trio", TRIO), ("assist", ASSIST)])
def test_golden_agents_score_the_same(name, task):
    from npbackend.batch import evaluate_population
    g = json.loads((GOLDEN / f"agent_{name}.json").read_text())
    assert evaluate_population(task, [g["mlp"]["params"]], SIZES)[0] == pytest.approx(g["score"], abs=1e-7)


@needs_numpy
def test_custom_targets():
    from npbackend.batch import evaluate_population
    gs = genomes(3)
    py = [evaluate(FREE, policy_from_genome(g), FREE_HELDOUT) for g in gs]
    assert evaluate_population(FREE, gs, SIZES, FREE_HELDOUT) == pytest.approx(py, abs=1e-12)


@needs_numpy
def test_training_with_numpy_matches_python():
    from env.navigation import Task
    small = Task(name="small", targets=FREE.targets[:3])
    a = train(small, "ga", generations=4, pop_size=8, seed=9, backend="python")
    b = train(small, "ga", generations=4, pop_size=8, seed=9, backend="numpy")
    assert b.score == pytest.approx(a.score, abs=1e-12)
    assert b.genome == a.genome
