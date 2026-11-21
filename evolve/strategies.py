"""Neuroevolution over flat parameter vectors, with an ask/tell API.

* ``GeneticAlgorithm`` -- truncation selection, elitism, uniform
  crossover, Gaussian mutation. Robust and simple.
* ``EvolutionStrategy`` -- OpenAI-style ES: antithetic Gaussian
  perturbations around a mean, rank-based fitness shaping, Adam-free
  plain gradient ascent. Smooth progress on smooth problems.

Both *maximise* fitness and draw every random number from their own
seeded ``random.Random``, so a run is reproducible from its seed.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Callable

Genome = list[float]


@dataclass
class Generation:
    index: int
    best: float
    mean: float
    best_genome: Genome = field(repr=False)


class GeneticAlgorithm:
    def __init__(self, x0: Genome, seed: int, pop_size: int = 32, elite: int | None = None,
                 parents: int | None = None, sigma: float = 0.1, init_sigma: float = 0.5,
                 crossover: float = 0.5) -> None:
        # defaults scale with the population: top quarter breed, top eighth survive
        if parents is None:
            parents = max(1, pop_size // 4)
        if elite is None:
            elite = max(1, min(parents, pop_size // 8))
        if not 0 < elite <= parents <= pop_size:
            raise ValueError("need 0 < elite <= parents <= pop_size")
        self.rng = random.Random(seed)
        self.pop_size, self.elite, self.parents = pop_size, elite, parents
        self.sigma, self.crossover = sigma, crossover
        self.population = [list(x0)] + [
            [v + self.rng.gauss(0.0, init_sigma) for v in x0] for _ in range(pop_size - 1)
        ]

    def ask(self) -> list[Genome]:
        return self.population

    def tell(self, fitness: list[float]) -> None:
        order = sorted(range(len(fitness)), key=lambda i: fitness[i], reverse=True)
        ranked = [self.population[i] for i in order]
        parents = ranked[: self.parents]
        nxt = [list(g) for g in ranked[: self.elite]]
        rng = self.rng
        while len(nxt) < self.pop_size:
            a = parents[rng.randrange(len(parents))]
            if rng.random() < self.crossover:
                b = parents[rng.randrange(len(parents))]
                child = [x if rng.random() < 0.5 else y for x, y in zip(a, b)]
            else:
                child = list(a)
            nxt.append([v + rng.gauss(0.0, self.sigma) for v in child])
        self.population = nxt


class EvolutionStrategy:
    def __init__(self, x0: Genome, seed: int, pop_size: int = 32, sigma: float = 0.1,
                 lr: float = 0.05, weight_decay: float = 0.0) -> None:
        if pop_size < 2 or pop_size % 2:
            raise ValueError("pop_size must be even (antithetic pairs) and >= 2")
        self.rng = random.Random(seed)
        self.mean = list(x0)
        self.pop_size, self.sigma, self.lr, self.wd = pop_size, sigma, lr, weight_decay
        self._noise: list[Genome] = []

    def ask(self) -> list[Genome]:
        half = [[self.rng.gauss(0.0, 1.0) for _ in self.mean] for _ in range(self.pop_size // 2)]
        self._noise = half + [[-e for e in eps] for eps in half]
        return [[m + self.sigma * e for m, e in zip(self.mean, eps)] for eps in self._noise]

    def tell(self, fitness: list[float]) -> None:
        n = len(fitness)
        order = sorted(range(n), key=lambda i: fitness[i])
        shaped = [0.0] * n
        for rank, i in enumerate(order):  # centred ranks in [-0.5, 0.5]
            shaped[i] = rank / (n - 1) - 0.5
        scale = self.lr / (n * self.sigma)
        grad = [0.0] * len(self.mean)
        for w, eps in zip(shaped, self._noise):
            for j, e in enumerate(eps):
                grad[j] += w * e
        self.mean = [m * (1.0 - self.wd) + scale * g for m, g in zip(self.mean, grad)]


STRATEGIES = {"ga": GeneticAlgorithm, "es": EvolutionStrategy}


def run(strategy, fitness_fn: Callable[[Genome], float], generations: int,
        on_generation: Callable[[Generation], None] | None = None) -> tuple[Genome, float, list[Generation]]:
    """Drive ``strategy`` for ``generations``; return the best genome
    ever evaluated, its fitness, and per-generation stats."""
    best_g: Genome = []
    best_f = -math.inf
    history = []
    for gen in range(generations):
        pop = strategy.ask()
        fit = [fitness_fn(g) for g in pop]
        i = max(range(len(fit)), key=fit.__getitem__)
        if fit[i] > best_f:
            best_f, best_g = fit[i], list(pop[i])
        stats = Generation(gen, fit[i], math.fsum(fit) / len(fit), list(pop[i]))
        history.append(stats)
        if on_generation:
            on_generation(stats)
        strategy.tell(fit)
    return best_g, best_f, history
