"""Golden trained agents. Each case records exactly how it was trained;
``regen`` retrains it (slow: tens of seconds per agent on a desktop,
much longer on iSH), while the golden test only re-*scores* the stored
weights, which is fast and deterministic: free-space flight over 100
steps is not chaotic, so scores reproduce to ~1e-12."""

from __future__ import annotations

import json
from pathlib import Path

from env.agent import SIZES, policy_from_genome
from env.navigation import FREE_HELDOUT, evaluate
from env.training import task_by_name, train

DATA_DIR = Path(__file__).parent / "data"

CASES = {
    "free": dict(task="free", strategy="es", generations=100, pop=32, seed=1),
}


def run_case(name: str) -> dict:
    spec = CASES[name]
    task = task_by_name(spec["task"])
    res = train(task, spec["strategy"], spec["generations"], spec["pop"], spec["seed"])
    net = policy_from_genome(res.genome)
    out = {"case": name, **spec, "score": res.score,
           "history_best": [g.best for g in res.history], "mlp": net.to_dict()}
    if spec["task"] == "free":
        out["heldout_score"] = evaluate(task, net, FREE_HELDOUT)
    assert out["mlp"]["sizes"] == SIZES
    return out


def golden_path(name: str) -> Path:
    return DATA_DIR / f"agent_{name}.json"


def load_golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text())
