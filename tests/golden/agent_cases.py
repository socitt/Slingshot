"""Golden trained agents. Each case records exactly how it was trained
(including the backend: python and numpy agree to ~1e-9 in the field,
which is enough to reorder an ES ranking now and then, so a retrain is
only bit-reproducible on the backend it was made with);
``regen`` retrains it (slow: tens of seconds per agent on a desktop,
much longer on iSH), while the golden test only re-*scores* the stored
weights, which is fast and deterministic: free-space flight over 100
steps is not chaotic, so scores reproduce to ~1e-12."""

from __future__ import annotations

import json
from pathlib import Path

from env.agent import SIZES, policy_from_genome
from env.navigation import HELDOUT, evaluate, run_episode
from env.training import task_by_name, train

DATA_DIR = Path(__file__).parent / "data" / "agents"

CASES = {
    "free": dict(task="free", strategy="es", generations=100, pop=32, seed=1),
    # numpy backend for training speed (~3 min vs ~40); rescoring in the
    # golden test is pure Python. Regenerating this case needs numpy.
    "trio": dict(task="trio", strategy="es", generations=500, pop=64, seed=1, backend="numpy"),
    "assist": dict(task="assist", strategy="es", generations=100, pop=32, seed=1),
}


def run_case(name: str) -> dict:
    spec = CASES[name]
    task = task_by_name(spec["task"])
    res = train(task, spec["strategy"], spec["generations"], spec["pop"], spec["seed"],
                backend=spec.get("backend", "python"))
    net = policy_from_genome(res.genome)
    out = {"case": name, **spec, "score": res.score,
           "history_best": [g.best for g in res.history], "mlp": net.to_dict()}
    heldout = HELDOUT[task.name]
    out["heldout_score"] = evaluate(task, net, heldout)
    episodes = [run_episode(task, net, t) for t in task.targets]
    out["reached"] = [r.reached for r in episodes]
    out["heldout_reached"] = [run_episode(task, net, t).reached for t in heldout]
    out["work"] = [r.work for r in episodes]
    assert out["mlp"]["sizes"] == SIZES
    return out


def golden_path(name: str) -> Path:
    return DATA_DIR / f"agent_{name}.json"


def load_golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text())
