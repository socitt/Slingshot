"""Repo-wide pytest configuration.

Kept standard-library only on purpose: lirk folds this file into every
pytest target's fingerprint and holds its imports to the ``deps`` rule,
so importing a repo package here would force that dep onto every
target in the repo.
"""

import random

import pytest

#: The one seed tests should derive their randomness from.
SEED = 20261121


@pytest.fixture
def rng():
    """A freshly seeded ``random.Random`` per test -- never the global RNG."""
    return random.Random(SEED)


@pytest.fixture(autouse=True)
def _poison_global_random(monkeypatch):
    """Fail loudly if code under test draws from the unseeded global RNG."""

    def _refuse(*_a, **_k):
        raise AssertionError("use a seeded random.Random, not the global RNG")

    for name in ("random", "uniform", "gauss", "randint", "choice", "shuffle"):
        monkeypatch.setattr(random, name, _refuse)
