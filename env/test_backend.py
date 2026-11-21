import sys

import pytest

from env.agent import policy_from_genome
from env.backend import BackendUnavailable, numpy_available, population_evaluator
from env.navigation import FREE, evaluate


def test_python_backend_matches_evaluate():
    from env.agent import GENOME_SIZE
    gs = [[0.0] * GENOME_SIZE, [0.01 * (i % 7 - 3) for i in range(GENOME_SIZE)]]
    f = population_evaluator(FREE, "python")
    assert f(gs) == [evaluate(FREE, policy_from_genome(g)) for g in gs]


def test_default_comes_from_environment(monkeypatch):
    monkeypatch.setenv("SLINGSHOT_BACKEND", "bogus")
    with pytest.raises(ValueError, match="unknown backend"):
        population_evaluator(FREE)


def test_numpy_missing_gives_a_clear_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "numpy", None)   # import numpy -> ImportError
    assert not numpy_available()
    with pytest.raises(BackendUnavailable, match="pip install numpy"):
        population_evaluator(FREE, "numpy")
