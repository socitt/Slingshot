"""Golden chaos cases: a short Lorenz trajectory and seeded Lyapunov
scores. The trajectory horizon is 2 time units (about two e-foldings at
lambda ~ 0.9), so platform ulp differences stay far below tolerance."""

from __future__ import annotations

import json
import math
from pathlib import Path

from chaos.logistic import feigenbaum_estimates, logistic, logistic_deriv, superstable_sequence
from chaos.lorenz import lorenz, lorenz_jacobian, trajectory
from chaos.lyapunov import lyapunov_map, lyapunov_spectrum

DATA_DIR = Path(__file__).parent / "data"


def lorenz_short() -> dict:
    traj = trajectory([1.0, 1.0, 1.0], 0.01, 200, every=20)
    return {"h": 0.01, "steps": 200, "every": 20, "points": traj}


def lyapunov_scores() -> dict:
    logistic_r4 = lyapunov_map(lambda x: logistic(4.0, x), lambda x: logistic_deriv(4.0, x),
                               0.123456789, 20_000)
    spectrum = lyapunov_spectrum(lorenz(), lorenz_jacobian(), [1.0, 1.0, 1.0],
                                 0.01, 10_000, 10, 1000)
    return {
        "logistic_r4": logistic_r4,
        "ln2": math.log(2),
        "lorenz_spectrum_T100": spectrum,
        "superstable_r": superstable_sequence(8),
        "feigenbaum_estimates": feigenbaum_estimates(8),
    }


CASES = {"lorenz_short": lorenz_short, "lyapunov_scores": lyapunov_scores}


def run_case(name: str) -> dict:
    return {"case": name, **CASES[name]()}


def golden_path(name: str) -> Path:
    return DATA_DIR / f"chaos_{name}.json"


def load_golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text())
