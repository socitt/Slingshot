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
GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "data" / "agents"


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
def test_scores_match_python_backend_in_free_space():
    from npbackend.batch import evaluate_population
    gs = genomes(12)
    py = [evaluate(FREE, policy_from_genome(g)) for g in gs]
    assert evaluate_population(FREE, gs, SIZES) == pytest.approx(py, abs=1e-12)


@needs_numpy
@pytest.mark.parametrize("task", [TRIO, ASSIST])
def test_scores_match_python_backend_in_the_field(task):
    """Only summation order differs between backends (~1e-16), but the
    field is chaotic: a random genome that loiters through a close pass
    for 320 steps can amplify that to ~1e-3 in one episode (seen: genome
    9 of this seed, target 6 of TRIO, 6.6e-4). So: typical agreement is
    tight, worst case is bounded, and nothing is a threshold flip."""
    from npbackend.batch import evaluate_population
    gs = genomes(12)
    py = [evaluate(task, policy_from_genome(g)) for g in gs]
    diffs = sorted(abs(a - b) for a, b in zip(py, evaluate_population(task, gs, SIZES)))
    assert diffs[len(diffs) // 2] < 1e-9
    assert diffs[-1] < 1e-3


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
