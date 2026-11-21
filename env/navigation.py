"""The navigation task: fly a thrust-limited probe to a target.

The probe is a massless test particle. It feels the gravity of an
optional N-body field (the bodies move under their own gravity,
integrated alongside it) plus its own thrust, chosen each step by a
policy from a fixed-size observation vector. Without a field this is
plain free-space navigation (milestone 3); with one, the cheapest route
may be a gravity assist (milestone 4).

Integration is kick-drift-kick leapfrog for bodies and probe together,
with thrust held constant across each step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

from physics.gravity import System

Policy = Callable[[list[float]], list[float]]
FieldBuilder = Callable[[], tuple[System, list[float], list[float]]]

#: Size of the observation vector every policy receives.
OBS_SIZE = 9
#: Size of the action vector: thrust direction/magnitude in [-1, 1]^2.
ACT_SIZE = 2


@dataclass(frozen=True)
class Task:
    name: str
    targets: tuple[tuple[float, float], ...]
    start: tuple[float, float] = (0.0, 0.0)
    start_vel: tuple[float, float] = (0.0, 0.0)
    target_radius: float = 0.1
    dt: float = 0.05
    max_steps: int = 100
    thrust_max: float = 0.5      # acceleration magnitude at full throttle
    fuel: float = 2.0            # delta-v budget
    fuel_weight: float = 0.3     # score penalty at a fully spent budget
    bounds: float = 5.0          # leaving |x| or |y| > bounds ends the episode
    field: FieldBuilder | None = None
    body_radius: float = 0.05    # closer than this to a body is a crash
    pos_scale: float = 2.0       # observation normalisers
    vel_scale: float = 1.0
    accel_ref: float = 0.5


@dataclass
class Result:
    score: float
    reached: bool
    crashed: bool
    out_of_bounds: bool
    steps: int
    fuel_used: float
    min_dist: float
    frames: list = field(default_factory=list, repr=False)


def _soft(v: float, ref: float) -> float:
    """Squash an unbounded magnitude into (-1, 1), keeping its sign."""
    return v / (abs(v) + ref)


def observe(task: Task, p: list[float], v: list[float], target: tuple[float, float],
            g: tuple[float, float], nearest: tuple[float, float], fuel_left: float) -> list[float]:
    ps, vs = task.pos_scale, task.vel_scale
    return [
        (target[0] - p[0]) / ps, (target[1] - p[1]) / ps,
        v[0] / vs, v[1] / vs,
        _soft(g[0], task.accel_ref), _soft(g[1], task.accel_ref),
        nearest[0] / ps, nearest[1] / ps,
        fuel_left / task.fuel,
    ]


def run_episode(task: Task, policy: Policy, target: tuple[float, float],
                record: bool = False) -> Result:
    if task.field is not None:
        system, bpos, bvel = task.field()
        bacc = system.accelerations(bpos)
    else:
        system, bpos, bvel, bacc = None, [], [], []

    p, v = list(task.start), list(task.start_vel)
    tx, ty = target
    d0 = math.hypot(tx - p[0], ty - p[1])
    dmin, fuel_used = d0, 0.0
    dt, half = task.dt, 0.5 * task.dt
    frames = []

    def gravity_and_nearest(px, py, pos):
        if system is None:
            return (0.0, 0.0), (0.0, 0.0), math.inf
        gx, gy = system.field_at(px, py, pos)
        best, near = math.inf, (0.0, 0.0)
        for j in range(system.n):
            dx, dy = pos[2 * j] - px, pos[2 * j + 1] - py
            d = math.hypot(dx, dy)
            if d < best:
                best, near = d, (dx, dy)
        return (gx, gy), near, best

    g, near, dbody = gravity_and_nearest(p[0], p[1], bpos)
    reached = crashed = oob = False
    step = 0
    for step in range(1, task.max_steps + 1):
        fuel_left = task.fuel - fuel_used
        ax, ay = policy(observe(task, p, v, target, g, near, fuel_left))
        ax = max(-1.0, min(1.0, ax)) * task.thrust_max
        ay = max(-1.0, min(1.0, ay)) * task.thrust_max
        mag = math.hypot(ax, ay)
        if mag > task.thrust_max:              # clip to a disc, not a square
            ax, ay, mag = ax * task.thrust_max / mag, ay * task.thrust_max / mag, task.thrust_max
        if mag * dt > fuel_left:               # running dry: partial burn
            k = fuel_left / (mag * dt) if mag > 0 else 0.0
            ax, ay, mag = ax * k, ay * k, mag * k
        fuel_used += mag * dt
        if record:
            frames.append({"t": (step - 1) * dt, "probe": list(p), "bodies": list(bpos),
                           "thrust": (ax, ay)})

        # kick-drift-kick, probe and bodies together
        vx = v[0] + half * (g[0] + ax)
        vy = v[1] + half * (g[1] + ay)
        p = [p[0] + dt * vx, p[1] + dt * vy]
        if system is not None:
            bvel = [vi + half * ai for vi, ai in zip(bvel, bacc)]
            bpos = [xi + dt * vi for xi, vi in zip(bpos, bvel)]
            bacc = system.accelerations(bpos)
            bvel = [vi + half * ai for vi, ai in zip(bvel, bacc)]
        g, near, dbody = gravity_and_nearest(p[0], p[1], bpos)
        v = [vx + half * (g[0] + ax), vy + half * (g[1] + ay)]

        d = math.hypot(tx - p[0], ty - p[1])
        dmin = min(dmin, d)
        if d < task.target_radius:
            reached = True
            break
        if dbody < task.body_radius:
            crashed = True
            break
        if abs(p[0]) > task.bounds or abs(p[1]) > task.bounds:
            oob = True
            break

    if record:
        frames.append({"t": step * dt, "probe": list(p), "bodies": list(bpos), "thrust": (0.0, 0.0)})
    score = episode_score(task, reached, crashed, step, fuel_used, dmin, d0)
    return Result(score, reached, crashed, oob, step, fuel_used, dmin, frames)


def episode_score(task: Task, reached: bool, crashed: bool, steps: int,
                  fuel_used: float, dmin: float, d0: float) -> float:
    """Reaching the target scores 1 plus up to 0.5 for arriving early.
    Missing scores up to 0.8 for how close the probe got. Spending the
    whole fuel budget costs ``fuel_weight``, crashing costs 0.5."""
    if reached:
        score = 1.0 + 0.5 * (1.0 - steps / task.max_steps)
    else:
        score = 0.8 * max(0.0, 1.0 - dmin / d0) if d0 > 0 else 0.0
    score -= task.fuel_weight * fuel_used / task.fuel
    if crashed:
        score -= 0.5
    return score


def evaluate(task: Task, policy: Policy, targets=None) -> float:
    """Mean episode score over the task's (or the given) targets."""
    targets = task.targets if targets is None else targets
    return math.fsum(run_episode(task, policy, t).score for t in targets) / len(targets)


def ring(n: int, radius: float, phase: float = 0.0) -> tuple[tuple[float, float], ...]:
    return tuple((radius * math.cos(phase + 2 * math.pi * k / n),
                  radius * math.sin(phase + 2 * math.pi * k / n)) for k in range(n))


#: Milestone 3: free space. Twelve targets at two radii with staggered
#: angles -- a single ring lets the net memorise eight directions
#: instead of learning to steer (held-out score ~0.5 vs ~1.0 on train).
FREE = Task(name="free", targets=ring(6, 1.0) + ring(6, 2.0, phase=math.pi / 6))
#: Held-out targets for FREE: the same ring rotated by half a spacing.
FREE_HELDOUT = ring(8, 1.5, phase=math.pi / 8)

TASKS = {"free": FREE}
