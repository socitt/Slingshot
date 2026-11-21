"""Golden physics cases: one function per case, shared by the golden
test and by ``regen.py`` so the two can never disagree about
how a trajectory was produced.

Chaos makes long float trajectories fragile (an x87 build on iSH can
differ from x86-64 in the last ulp, and three-body dynamics amplify
that exponentially), so every case is a *short horizon*.
"""

from __future__ import annotations

import json
from pathlib import Path

from physics.orbits import figure_eight, pythagorean, two_body
from physics.simulate import simulate

DATA_DIR = Path(__file__).parent / "data" / "physics"

CASES = {
    # name: (builder, h, steps, method, every)
    "figure8_rk4": (figure_eight, 1e-3, 2000, "rk4", 250),
    "figure8_leapfrog": (figure_eight, 1e-3, 2000, "leapfrog", 250),
    "kepler_e06_rk4": (lambda: two_body(1.0, 0.5, 1.0, 0.6), 2e-3, 1500, "rk4", 150),
    "pythagorean_rk4": (pythagorean, 1e-4, 10000, "rk4", 1000),
}


def run_case(name: str) -> dict:
    builder, h, steps, method, every = CASES[name]
    system, pos, vel = builder()
    frames = simulate(system, pos, vel, h, steps, method, every)
    return {
        "case": name,
        "masses": list(system.masses),
        "h": h,
        "steps": steps,
        "method": method,
        "energy0": system.energy(pos, vel),
        "frames": [{"t": f.t, "pos": f.pos, "vel": f.vel} for f in frames],
    }


def golden_path(name: str) -> Path:
    return DATA_DIR / f"physics_{name}.json"


def load_golden(name: str) -> dict:
    return json.loads(golden_path(name).read_text())
