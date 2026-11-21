"""Assemble the data behind the slingshot visual explainer.

Everything comes from the repository: the golden agents and their
training specs, the golden chaos numbers, and lirk's invocation log.
Training is deterministic, so "watch it learn" re-runs a golden agent's
own training from its recorded spec and samples flights along the way.
"""

from __future__ import annotations

import dataclasses
import json
import math
import re
from pathlib import Path

from env.agent import policy_from_genome
from env.navigation import HELDOUT, TASKS, TRIO, Rendezvous, Tour, field_track, run_episode
from env.training import load_agent, train

ROOT = Path(__file__).resolve().parent.parent
AGENTS = ROOT / "tests" / "golden" / "data" / "agents"
CHAOS = ROOT / "tests" / "golden" / "data" / "chaos"
LIRK_LOG = ROOT / "docs" / ".lirk-invocations.log"


def _r(x: float) -> float:
    return round(x, 3)


def _target_json(t):
    if isinstance(t, Tour):
        return {"stops": list(t.stops), "delay": t.delay}
    return {"body": t.body, "delay": t.delay} if isinstance(t, Rendezvous) else [_r(t[0]), _r(t[1])]


def episode_json(task, policy, target, every: int = 2, bodies: bool = True) -> dict:
    """One flight, downsampled to every ``every``-th frame (last kept)."""
    r = run_episode(task, policy, target, record=True)
    fr = r.frames[::every]
    if (len(r.frames) - 1) % every:
        fr.append(r.frames[-1])
    out = {"target": _target_json(target), "reached": r.reached, "crashed": r.crashed,
           "score": _r(r.score), "fuel": _r(r.fuel_used), "steps": r.steps,
           "work": [_r(w) for w in r.work], "thrust_work": _r(r.thrust_work),
           "stops_done": r.stops_done, "stop_steps": r.stop_steps,
           "p": [[_r(f["probe"][0]), _r(f["probe"][1])] for f in fr],
           "th": [[_r(f["thrust"][0]), _r(f["thrust"][1])] for f in fr]}
    if isinstance(target, (Rendezvous, Tour)):
        out["tp"] = [[_r(f["target"][0]), _r(f["target"][1])] for f in fr]
    if isinstance(target, Tour):
        out["stop"] = [f["stop"] for f in fr]
    if bodies and task.field is not None:
        out["b"] = [[_r(v) for v in f["bodies"]] for f in fr]
    return out


def agent_json(name: str, every: int = 2) -> dict:
    path = AGENTS / f"agent_{name}.json"
    g = json.loads(path.read_text())
    policy, _ = load_agent(path)
    task = TASKS[g["task"]]
    return {
        "task": g["task"], "score": _r(g["score"]), "heldout": _r(g["heldout_score"]),
        "history": [_r(x) for x in g["history_best"]],
        "strategy": g["strategy"], "generations": g["generations"], "pop": g["pop"],
        "dt": task.dt * every, "radius": task.target_radius, "fuel": task.fuel,
        "masses": list(task.field()[0].masses) if task.field else [],
        "train": [episode_json(task, policy, t, every) for t in task.targets],
        "held": [episode_json(task, policy, t, every) for t in HELDOUT[task.name]],
    }


def learning_json(name: str, sample_gens: list[int], swarm: bool, target_index: int = 0,
                  every: int = 4, generations: int | None = None, pop: int | None = None,
                  stages_every: int = 0) -> dict:
    """Re-run golden agent ``name``'s training and record, at each sampled
    generation, the flight of every genome (``swarm``) or of the best one,
    all aimed at one showcase target. ``generations``/``pop`` override the
    spec (for quick tests); with the spec's own values the final best
    genome is the golden agent. With ``stages_every`` > 0 (needs numpy),
    every that-many generations also record how many genomes caught how
    many stops, over all of the task's targets: skills emerging in stages."""
    from tests.golden.agent_cases import CASES
    spec = CASES[name]          # the canonical training spec (the golden file may not exist yet)
    task = TASKS[spec["task"]]
    target = task.targets[target_index]
    gens = generations if generations is not None else spec["generations"]
    sample = sorted({g for g in sample_gens if g < gens})
    frames: dict[int, dict] = {}

    stages = []

    def on_population(gen, population, fitness):
        if stages_every and gen % stages_every == 0:
            from npbackend.batch import evaluate_population
            from env.agent import SIZES
            _, caught = evaluate_population(task, population, SIZES, details=True)
            n_stops = len(task.targets[0].stops) if isinstance(task.targets[0], Tour) else 1
            per_genome = caught.max(axis=1)          # best tour of each genome
            stages.append({"gen": gen,
                           "genomes": [int((per_genome == k).sum()) for k in range(n_stops + 1)],
                           "episodes": [int((caught == k).sum()) for k in range(n_stops + 1)]})
        if gen not in sample:
            return
        best = max(range(len(fitness)), key=fitness.__getitem__)
        chosen = range(len(population)) if swarm else [best]
        frames[gen] = {
            "best": _r(fitness[best]), "mean": _r(math.fsum(fitness) / len(fitness)),
            "flights": [episode_json(task, policy_from_genome(population[i]), target, every,
                                     bodies=False) | {"fitness": _r(fitness[i])} for i in chosen],
        }

    curve = []
    train(task, spec["strategy"], gens, pop or spec["pop"], spec["seed"],
                on_generation=lambda g: curve.append([_r(g.best), _r(g.mean)]),
                backend=spec.get("backend", "python"), on_population=on_population)
    out = {"task": task.name, "target": _target_json(target), "swarm": swarm, "dt": task.dt * every,
           "generations": gens, "curve": curve,
           "samples": [dict(gen=g, **frames[g]) for g in sample]}
    if stages:
        out["stages"] = stages
    if task.field is not None:
        # the field is the same for every flight of the showcase target
        _, track = field_track(task)
        off = target.delay if isinstance(target, (Rendezvous, Tour)) else 0
        out["field"] = [[_r(v) for v in track[k]] for k in range(off, off + task.max_steps + 1, every)]
        out["masses"] = list(task.field()[0].masses)
    return out


def chaos_json() -> dict:
    a = dataclasses.replace(TRIO, targets=((99.0, 99.0),), bounds=50.0, max_steps=1600, body_radius=0.0)
    b = dataclasses.replace(a, start=(0.6 + 1e-9, 0.0))
    zero = lambda o: [0.0, 0.0]  # noqa: E731
    ra = run_episode(a, zero, (99.0, 99.0), record=True)
    rb = run_episode(b, zero, (99.0, 99.0), record=True)
    sep = [[_r(fa["t"]), round(math.log10(max(1e-16, math.dist(fa["probe"], fb["probe"]))), 3)]
           for fa, fb in zip(ra.frames, rb.frames)][::8]
    ly = json.loads((CHAOS / "chaos_lyapunov_scores.json").read_text())
    return {
        "divergence": sep,
        "coast": {"a": [[_r(f["probe"][0]), _r(f["probe"][1])] for f in ra.frames[::4]],
                  "b": [[_r(f["probe"][0]), _r(f["probe"][1])] for f in rb.frames[::4]]},
        "lyapunov": {"logistic_r4": ly["logistic_r4"], "lorenz": ly["lorenz_spectrum_T100"],
                     "feigenbaum": ly["feigenbaum_estimates"]},
    }


def lirk_log(path: Path = LIRK_LOG) -> list:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            m = re.match(r"(\S+) rc=(\d+) secs=(\d+) lirk (\S+)", line)
            if m:
                out.append([m.group(1)[11:16], int(m.group(2)), int(m.group(3)), m.group(4)])
    return out


def build(agent_names: list[str], learning: dict | None = None) -> dict:
    """``learning`` maps an agent name to kwargs for ``learning_json``."""
    data = {"agents": {n: agent_json(n) for n in agent_names}, **chaos_json(), "lirk_log": lirk_log()}
    if learning:
        data["learning"] = {n: learning_json(n, **kw) for n, kw in learning.items()}
    return data
