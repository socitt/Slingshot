"""Backend selection. ``python`` (default) needs only the standard
library; ``numpy`` evaluates whole populations at once and is optional.

    SLINGSHOT_BACKEND=numpy python3 -m cli.train ...
    python3 -m cli.train --backend numpy ...
"""

from __future__ import annotations

import os

from env.agent import SIZES, fitness_fn
from env.navigation import Task

BACKENDS = ("python", "numpy")


class BackendUnavailable(RuntimeError):
    pass


def default_backend() -> str:
    return os.environ.get("SLINGSHOT_BACKEND", "python")


def numpy_available() -> bool:
    try:
        import numpy  # noqa: F401
    except ImportError:
        return False
    return True


def population_evaluator(task: Task, backend: str | None = None):
    """Return ``f(genomes) -> list[score]`` for the chosen backend."""
    backend = backend or default_backend()
    if backend not in BACKENDS:
        raise ValueError(f"unknown backend {backend!r}; choose from {BACKENDS}")
    if backend == "numpy":
        if not numpy_available():
            raise BackendUnavailable(
                "the numpy backend needs numpy (pip install numpy); "
                "the default python backend needs nothing")
        from npbackend.batch import evaluate_population

        return lambda genomes: evaluate_population(task, genomes, SIZES)
    fitness = fitness_fn(task)
    return lambda genomes: [fitness(g) for g in genomes]
