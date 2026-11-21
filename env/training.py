"""Train navigation agents by neuroevolution. Deterministic per seed."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from env.agent import GENOME_SIZE, SIZES, fitness_fn, policy_from_genome
from env.navigation import TASKS, Task, evaluate
from evolve.strategies import STRATEGIES, Generation, run

DEFAULTS = {
    "ga": dict(sigma=0.15, init_sigma=0.5),
    "es": dict(sigma=0.1, lr=0.1),
}


@dataclass
class TrainResult:
    genome: list[float]
    score: float
    history: list[Generation]


def train(task: Task, strategy: str = "ga", generations: int = 30, pop_size: int = 24,
          seed: int = 1, init: list[float] | None = None,
          on_generation: Callable[[Generation], None] | None = None) -> TrainResult:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown strategy {strategy!r}; choose from {sorted(STRATEGIES)}")
    x0 = list(init) if init is not None else [0.0] * GENOME_SIZE
    if len(x0) != GENOME_SIZE:
        raise ValueError(f"init genome has {len(x0)} params, expected {GENOME_SIZE}")
    strat = STRATEGIES[strategy](x0, seed=seed, pop_size=pop_size, **DEFAULTS[strategy])
    genome, score, history = run(strat, fitness_fn(task), generations, on_generation)
    return TrainResult(genome, score, history)


def save_agent(path: str | Path, genome: list[float], **meta) -> None:
    net = policy_from_genome(genome)
    net.save(path, **meta)


def load_agent(path: str | Path):
    """Return ``(policy, metadata)``."""
    data = json.loads(Path(path).read_text())
    meta = {k: v for k, v in data.items() if k != "mlp"}
    if data["mlp"]["sizes"] != SIZES:
        raise ValueError(f"{path}: network sizes {data['mlp']['sizes']} != current {SIZES}")
    return policy_from_genome(data["mlp"]["params"]), meta


def task_by_name(name: str) -> Task:
    try:
        return TASKS[name]
    except KeyError:
        raise ValueError(f"unknown task {name!r}; choose from {sorted(TASKS)}") from None


__all__ = ["train", "save_agent", "load_agent", "task_by_name", "evaluate", "TrainResult"]
