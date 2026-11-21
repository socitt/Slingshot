"""Optional numpy backend: evaluate a whole population at once.

Runs every (genome, target) episode in lockstep as rows of numpy
arrays, mirroring ``env.navigation.run_episode`` step for step (same
thrust clipping, fuel accounting, termination order and score). Only
summation order differs, so scores agree with the pure-Python backend
to ~1e-12 in free space and ~1e-9 in the chaotic field.

Nothing in the core imports this module; ``env.backend`` loads it on
request, so the project still runs where numpy can't be installed.
"""

from __future__ import annotations

import math

import numpy as np

from env.navigation import Rendezvous, Task, Tour, field_track


def _unpack(genomes: np.ndarray, sizes: list[int]):
    """(P, n_params) -> per-layer (W: (P, out, in), b: (P, out))."""
    layers, k = [], 0
    for a, b in zip(sizes, sizes[1:]):
        W = genomes[:, k:k + a * b].reshape(-1, b, a)
        k += a * b
        layers.append((W, genomes[:, k:k + b]))
        k += b
    return layers


def evaluate_population(task: Task, genomes: list[list[float]], sizes: list[int],
                        targets=None, details: bool = False):
    """Mean episode score per genome, like ``env.navigation.evaluate``.
    With ``details``, also return the (genome, target) array of stops
    caught -- how the grand-tour visual tracks skills emerging."""
    targets = task.targets if targets is None else targets
    P, T = len(genomes), len(targets)
    B = P * T
    G = np.asarray(genomes, dtype=np.float64)
    layers = [(W.repeat(T, axis=0), b.repeat(T, axis=0)) for W, b in _unpack(G, sizes)]
    # Moving targets (Rendezvous, Tour) start at a per-episode offset into
    # the field's history and follow a body -- for a tour, the body of the
    # current stop; fixed targets have offset 0 and one stop.
    moving = [isinstance(t, (Rendezvous, Tour)) for t in targets]
    if any(moving) and not all(moving):
        raise ValueError("mix of fixed and moving targets in one call")
    moving = all(moving) and len(targets) > 0
    stop_lists = [t.stops if isinstance(t, Tour) else (t.body,) if moving else (0,) for t in targets]
    smax = max(len(x) for x in stop_lists)
    stops = np.tile(np.asarray([list(x) + [x[-1]] * (smax - len(x)) for x in stop_lists],
                               dtype=np.int64), (P, 1))                  # (B, smax)
    nst = np.tile(np.asarray([len(x) for x in stop_lists], dtype=np.int64), P)
    kstop = np.zeros(B, dtype=np.int64)                                   # current stop
    if moving:
        off = np.tile(np.asarray([t.delay for t in targets], dtype=np.int64), P)
    else:
        off = np.zeros(B, dtype=np.int64)
        tgt = np.tile(np.asarray(targets, dtype=np.float64), (P, 1))      # (B, 2)

    system, track = field_track(task)
    if system is not None:
        tr = np.asarray(track, dtype=np.float64).reshape(len(track), system.n, 2)
        masses = np.asarray(system.masses)
        eps2 = system.softening ** 2
        heaviest = int(np.argmax(masses)) if system.n > 1 else -1
        planet = np.ones(system.n, dtype=bool)
        if heaviest >= 0:
            planet[heaviest] = False

    rows = np.arange(B)

    def bodies(s):
        return tr[off + s]                                                 # (B, n, 2)

    def target_pos(s):
        return bodies(s)[rows, stops[rows, np.minimum(kstop, nst - 1)]] if moving else tgt

    def field(p, s):
        if system is None:
            z = np.zeros((B, 2))
            return z, z, np.full(B, np.inf)
        d = bodies(s) - p[:, None, :]                                      # (B, n, 2)
        d2 = (d * d).sum(axis=2)
        r2 = d2 + eps2
        s_ = system.G * masses[None, :] / (r2 * np.sqrt(r2))
        g = (d * s_[:, :, None]).sum(axis=1)
        dp = np.where(planet[None, :], d2, np.inf)
        j = dp.argmin(axis=1)
        near = d[rows, j]
        return g, near, np.sqrt(d2.min(axis=1))

    p = np.tile(np.asarray(task.start, dtype=np.float64), (B, 1))
    v = np.tile(np.asarray(task.start_vel, dtype=np.float64), (B, 1))
    tgt = target_pos(0)
    d0 = np.hypot(*(tgt - p).T)
    dmin = d0.copy()
    fuel_used = np.zeros(B)
    steps = np.zeros(B, dtype=np.int64)
    reached = np.zeros(B, dtype=bool)
    crashed = np.zeros(B, dtype=bool)
    oob = np.zeros(B, dtype=bool)
    active = np.ones(B, dtype=bool)
    dt, half, tmax = task.dt, 0.5 * task.dt, task.thrust_max
    g, near, _ = field(p, 0)

    for step in range(1, task.max_steps + 1):
        if not active.any():
            break
        fuel_left = task.fuel - fuel_used
        obs = np.column_stack([
            (tgt - p) / task.pos_scale,
            v / task.vel_scale,
            g / (np.abs(g) + task.accel_ref),
            near / task.pos_scale,
            fuel_left / task.fuel,
        ])
        act = obs
        for li, (W, b) in enumerate(layers):
            act = np.tanh(np.einsum("boi,bi->bo", W, act) + b)
        a = np.clip(act, -1.0, 1.0) * tmax
        mag = np.hypot(a[:, 0], a[:, 1])
        over = mag > tmax
        a[over] *= (tmax / mag[over])[:, None]
        mag[over] = tmax
        dry = mag * dt > fuel_left
        k = np.where(mag > 0, fuel_left / np.where(mag > 0, mag * dt, 1.0), 0.0)
        a[dry] *= k[dry][:, None]
        mag[dry] *= k[dry]

        v_half = v + half * (g + a)
        p_new = p + dt * v_half
        g_new, near_new, dbody = field(p_new, step)
        if moving:
            tgt = target_pos(step)
        v_new = v_half + half * (g_new + a)

        m = active
        fuel_used[m] += mag[m] * dt
        p[m], v[m], g[m], near[m] = p_new[m], v_new[m], g_new[m], near_new[m]
        steps[m] = step
        d = np.hypot(*(tgt - p).T)
        dmin[m] = np.minimum(dmin[m], d[m])
        hit = m & (d < task.target_radius)
        final = hit & (kstop + 1 == nst)
        inter = hit & ~final
        kstop[hit] += 1
        if inter.any():                       # next stop: measure closeness from here
            tgt = target_pos(step)
            dn = np.hypot(*(tgt - p).T)
            d0[inter] = dn[inter]
            dmin[inter] = dn[inter]
        crash = m & ~hit & (dbody < task.body_radius)
        lost = m & ~hit & ~crash & ((np.abs(p[:, 0]) > task.bounds) | (np.abs(p[:, 1]) > task.bounds))
        reached |= final
        crashed |= crash
        oob |= lost
        active = m & ~(final | crash | lost)

    miss = 0.8 * np.maximum(0.0, 1.0 - dmin / np.where(d0 > 0, d0, 1.0))
    miss = np.where(d0 > 0, miss, 0.0)
    score = kstop + np.where(reached, 0.5 * (1.0 - steps / task.max_steps), miss)
    score = score - task.fuel_weight * fuel_used / task.fuel - 0.5 * crashed
    per = score.reshape(P, T)
    if details:
        return [math.fsum(row) / T for row in per.tolist()], kstop.reshape(P, T)
    return [math.fsum(row) / T for row in per.tolist()]
