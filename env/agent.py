"""Glue between nn/ and env/: genomes become policies."""

from __future__ import annotations

from env.navigation import ACT_SIZE, OBS_SIZE, Task, evaluate
from nn.mlp import MLP, param_count

HIDDEN = (12,)
SIZES = [OBS_SIZE, *HIDDEN, ACT_SIZE]
GENOME_SIZE = param_count(SIZES)


def policy_from_genome(genome: list[float]) -> MLP:
    return MLP(SIZES, genome)


def fitness_fn(task: Task):
    def fitness(genome: list[float]) -> float:
        return evaluate(task, policy_from_genome(genome))
    return fitness
