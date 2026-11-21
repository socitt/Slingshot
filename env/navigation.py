"""The navigation task: fly a thrust-limited probe to a target.

The probe is a massless test particle. It feels the gravity of an
optional N-body field (the bodies move under their own gravity,
integrated alongside it) plus its own thrust, chosen each step by a
policy from a fixed-size observation vector. Without a field this is
plain free-space navigation (milestone 3); with one, the cheapest route
may be a gravity assist (milestone 4).

Integration is kick-drift-kick leapfrog, with thrust held constant
across each step. The probe is massless, so the bodies' motion does not
depend on it: their track is integrated once per task (same leapfrog,
same step) and cached, which makes episodes several times cheaper.
"""

from __future__ import annotations

import functools
import math
import random
from dataclasses import dataclass, field
from typing import Callable

from physics.gravity import System
from physics.integrators import leapfrog_step
from physics.orbits import star_planets

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
    track_extra: int = 0         # extra field history for Rendezvous delays
    field: FieldBuilder | None = None
    body_radius: float = 0.05    # closer than this to a body is a crash
    pos_scale: float = 2.0       # observation normalisers
    vel_scale: float = 1.0
    accel_ref: float = 0.5


@dataclass(frozen=True)
class Rendezvous:
    """A moving target: catch body ``body`` of the field. ``delay`` starts
    the episode that many steps into the field's (cached) history, so the
    same task can meet the planets at different phases. Deliberately not
    a tuple: a fixed target is an ``(x, y)`` pair, and this must never
    unpack as one."""
    body: int
    delay: int = 0


Target = "tuple[float, float] | Rendezvous"


@dataclass
class Result:
    score: float
    reached: bool
    crashed: bool
    out_of_bounds: bool
    steps: int
    fuel_used: float
    min_dist: float
    #: work done on the probe by each body's gravity (positive = the
    #: body gave the probe energy: a gravity assist)
    work: list = field(default_factory=list)
    thrust_work: float = 0.0
    final_pos: tuple = (0.0, 0.0)
    final_vel: tuple = (0.0, 0.0)
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


@functools.lru_cache(maxsize=16)
def field_track(task: Task) -> tuple[System | None, tuple[tuple[float, ...], ...]]:
    """Body positions at every step boundary 0..max_steps + track_extra
    (cached)."""
    if task.field is None:
        return None, ()
    system, pos, vel = task.field()
    track, acc = [tuple(pos)], None
    for _ in range(task.max_steps + task.track_extra):
        pos, vel, acc = leapfrog_step(system.accelerations, pos, vel, task.dt, acc)
        track.append(tuple(pos))
    return system, tuple(track)


def _per_body_field(system: System | None, px: float, py: float, pos) -> tuple[list, tuple, float]:
    """Per-body accelerations on the probe, the offset to the nearest
    *planet* (any body but the heaviest -- the star is always felt via
    gravity; planets are what a flyby needs to be timed against), and
    the distance to the nearest body of any kind (for crashes)."""
    if system is None:
        return [], (0.0, 0.0), math.inf
    eps2 = system.softening ** 2
    heaviest = max(range(system.n), key=system.masses.__getitem__) if system.n > 1 else -1
    accs, best_any, best_planet, near = [], math.inf, math.inf, (0.0, 0.0)
    for j, mj in enumerate(system.masses):
        dx, dy = pos[2 * j] - px, pos[2 * j + 1] - py
        d2 = dx * dx + dy * dy
        r2 = d2 + eps2
        s = system.G * mj / (r2 * math.sqrt(r2))
        accs.append((dx * s, dy * s))
        best_any = min(best_any, d2)
        if j != heaviest and d2 < best_planet:
            best_planet, near = d2, (dx, dy)
    return accs, near, math.sqrt(best_any)


def target_at(task: Task, target, step: int) -> tuple[float, float]:
    """Where the target is at ``step`` of an episode."""
    if isinstance(target, Rendezvous):
        _, track = field_track(task)
        if not track:
            raise ValueError("a Rendezvous target needs a field")
        if target.delay > task.track_extra:
            raise ValueError(f"delay {target.delay} > track_extra {task.track_extra}")
        b = track[target.delay + step]
        return b[2 * target.body], b[2 * target.body + 1]
    return target


def run_episode(task: Task, policy: Policy, target,
                record: bool = False) -> Result:
    system, track = field_track(task)
    nb = system.n if system is not None else 0
    off = target.delay if isinstance(target, Rendezvous) else 0
    bpos = track[off] if track else ()

    p, v = list(task.start), list(task.start_vel)
    tx, ty = target_at(task, target, 0)
    d0 = math.hypot(tx - p[0], ty - p[1])
    dmin, fuel_used, thrust_work = d0, 0.0, 0.0
    work = [0.0] * nb
    dt, half = task.dt, 0.5 * task.dt
    frames = []

    accs, near, dbody = _per_body_field(system, p[0], p[1], bpos)
    reached = crashed = oob = False
    step = 0
    for step in range(1, task.max_steps + 1):
        g = (sum(a[0] for a in accs), sum(a[1] for a in accs))
        fuel_left = task.fuel - fuel_used
        ax, ay = policy(observe(task, p, v, (tx, ty), g, near, fuel_left))
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
                           "thrust": (ax, ay), "target": (tx, ty)})

        # kick-drift-kick
        vx = v[0] + half * (g[0] + ax)
        vy = v[1] + half * (g[1] + ay)
        p = [p[0] + dt * vx, p[1] + dt * vy]
        if track:
            bpos = track[off + step]
            if off or isinstance(target, Rendezvous):
                tx, ty = target_at(task, target, step)
        new_accs, near, dbody = _per_body_field(system, p[0], p[1], bpos)
        gx = sum(a[0] for a in new_accs)
        gy = sum(a[1] for a in new_accs)
        v_old = v
        v = [vx + half * (gx + ax), vy + half * (gy + ay)]
        # Energy bookkeeping. Over a KDK step v' - v = half*sum_j(a_j + a_j')
        # + dt*thrust, and dKE = (v' - v).(v' + v)/2 exactly, so weighting
        # each term by the step-average velocity attributes dKE to bodies
        # and thrust with no residual. (Weighting by the half-step velocity
        # leaves an O(dt^2 a^2) error that is huge on close passes.)
        ux, uy = 0.5 * (v_old[0] + v[0]), 0.5 * (v_old[1] + v[1])
        for j in range(nb):
            work[j] += half * ((accs[j][0] + new_accs[j][0]) * ux + (accs[j][1] + new_accs[j][1]) * uy)
        thrust_work += dt * (ax * ux + ay * uy)
        accs = new_accs

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
        frames.append({"t": step * dt, "probe": list(p), "bodies": list(bpos), "thrust": (0.0, 0.0),
                       "target": (tx, ty)})
    score = episode_score(task, reached, crashed, step, fuel_used, dmin, d0)
    return Result(score, reached, crashed, oob, step, fuel_used, dmin, work, thrust_work,
                  tuple(p), tuple(v), frames)


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


def trio_field():
    """Star plus two planets: heavy enough to bend a probe's path, and
    (with the probe) a chaotic restricted few-body problem."""
    return star_planets([(0.05, 1.0, 0.0), (0.02, 1.8, math.pi)], softening=0.03)


#: Milestone 4: start on a tight orbit round the star, reach targets
#: beyond the outer planet. The fuel budget covers a Hohmann-like climb
#: (~0.35 delta-v) with margin, but not brute-force straight lines.
def scatter(n: int, r_min: float, r_max: float, seed: int) -> tuple[tuple[float, float], ...]:
    """``n`` seeded points in an annulus (a private RNG, never the global one)."""
    rng = random.Random(seed)
    pts = []
    for _ in range(n):
        r, a = rng.uniform(r_min, r_max), rng.uniform(0.0, 2.0 * math.pi)
        pts.append((r * math.cos(a), r * math.sin(a)))
    return tuple(pts)


#: Trained on 24 targets: with 6, agents memorised routes (1/6 held-out
#: targets reached); with 24 the same budget reaches 7/12 held-out.
TRIO = Task(
    name="trio",
    targets=ring(24, 2.4, phase=math.pi / 24),
    start=(0.6, 0.0),
    start_vel=(0.0, math.sqrt(1.0 / 0.6)),
    dt=0.025,
    max_steps=320,
    thrust_max=0.5,
    fuel=1.0,
    bounds=4.0,
    field=trio_field,
    body_radius=0.06,
    pos_scale=3.0,
    vel_scale=1.5,
    accel_ref=1.0,
)
#: Held out: random radii as well as angles, so not just ring gaps.
TRIO_HELDOUT = scatter(12, 2.2, 2.6, seed=1234)

#: TRIO on a budget: 0.45 delta-v barely covers the climb, and arriving
#: early is worth less than saving fuel, so energy from a planet flyby
#: (positive ``Result.work`` for a planet) pays.
ASSIST = Task(**{**TRIO.__dict__, "name": "assist", "fuel": 0.45, "fuel_weight": 0.5,
                  "targets": ring(6, 2.4, phase=math.pi / 6)})
ASSIST_HELDOUT = ring(6, 2.4)

#: Milestone 7: catch the outer planet itself. It goes round once in
#: ~16.5 time units (~660 steps), so each launch delay meets it at a
#: different phase. Thirty-six phases spread over one orbit: with twelve,
#: agents memorised launch windows (11/12 trained, 1/12 held-out caught);
#: with thirty-six the same budget catches 10/12 unseen phases.
OUTER_PERIOD_STEPS = 660
RENDEZVOUS = Task(**{**TRIO.__dict__, "name": "rendezvous",
                     "targets": tuple(Rendezvous(2, round(OUTER_PERIOD_STEPS * k / 36))
                                      for k in range(36)),
                     "target_radius": 0.12,
                     "track_extra": OUTER_PERIOD_STEPS})


def _heldout_delays(n: int, seed: int) -> tuple[Rendezvous, ...]:
    trained = {t.delay for t in RENDEZVOUS.targets}
    pool = [d for d in range(OUTER_PERIOD_STEPS) if d not in trained]
    return tuple(Rendezvous(2, d) for d in sorted(random.Random(seed).sample(pool, n)))


RENDEZVOUS_HELDOUT = _heldout_delays(12, seed=4321)

TASKS = {"free": FREE, "trio": TRIO, "assist": ASSIST, "rendezvous": RENDEZVOUS}
HELDOUT = {"free": FREE_HELDOUT, "trio": TRIO_HELDOUT, "assist": ASSIST_HELDOUT,
           "rendezvous": RENDEZVOUS_HELDOUT}
