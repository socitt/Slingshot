"""Drive an N-body ``System`` forward with a chosen integrator."""

from __future__ import annotations

from dataclasses import dataclass

from physics.gravity import System
from physics.integrators import leapfrog_step, rk4_step

METHODS = ("rk4", "leapfrog")


@dataclass
class Frame:
    t: float
    pos: list[float]
    vel: list[float]


def simulate(
    system: System,
    pos: list[float],
    vel: list[float],
    h: float,
    steps: int,
    method: str = "leapfrog",
    every: int = 1,
) -> list[Frame]:
    """Integrate ``steps`` steps of size ``h``; record every ``every``-th
    frame, always including the initial and final ones."""
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    if h <= 0 or steps < 0 or every < 1:
        raise ValueError("need h > 0, steps >= 0, every >= 1")
    k = 2 * system.n
    pos, vel = list(pos), list(vel)
    frames = [Frame(0.0, list(pos), list(vel))]
    acc = None
    for step in range(1, steps + 1):
        if method == "rk4":
            y = rk4_step(system.derivative, (step - 1) * h, pos + vel, h)
            pos, vel = y[:k], y[k:]
        else:
            pos, vel, acc = leapfrog_step(system.accelerations, pos, vel, h, acc)
        if step % every == 0 or step == steps:
            frames.append(Frame(step * h, pos, vel))
    return frames
