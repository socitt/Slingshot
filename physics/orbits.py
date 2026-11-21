"""Initial conditions for classic gravitational systems (G = 1 unless given).

Each builder returns ``(system, pos, vel)`` in the centre-of-mass frame.
"""

from __future__ import annotations

import math

from physics.gravity import System

#: Period of the Chenciner-Montgomery figure-eight choreography
#: (Simo 2002, masses 1, G = 1).
FIGURE_EIGHT_PERIOD = 6.32591398


def kepler_period(a: float, total_mass: float, G: float = 1.0) -> float:
    """Kepler's third law: T = 2 pi sqrt(a^3 / (G M))."""
    return 2.0 * math.pi * math.sqrt(a ** 3 / (G * total_mass))


def two_body(
    m1: float, m2: float, a: float, e: float = 0.0, G: float = 1.0
) -> tuple[System, list[float], list[float]]:
    """Two bodies on a Kepler orbit of semi-major axis ``a`` and
    eccentricity ``e``, starting at periapsis on the x-axis, orbiting
    counter-clockwise."""
    if not 0.0 <= e < 1.0:
        raise ValueError("eccentricity must be in [0, 1)")
    M = m1 + m2
    rp = a * (1.0 - e)
    # vis-viva at periapsis: v^2 = G M (2/rp - 1/a) = G M (1+e)/(a(1-e))
    vp = math.sqrt(G * M * (2.0 / rp - 1.0 / a))
    f1, f2 = m2 / M, m1 / M
    pos = [-f1 * rp, 0.0, f2 * rp, 0.0]
    vel = [0.0, -f1 * vp, 0.0, f2 * vp]
    return System((m1, m2), G=G), pos, vel


def figure_eight() -> tuple[System, list[float], list[float]]:
    """Three equal masses chasing each other round a figure-eight."""
    x1, y1 = 0.97000436, -0.24308753
    vx3, vy3 = -0.93240737, -0.86473146
    pos = [x1, y1, -x1, -y1, 0.0, 0.0]
    vel = [-vx3 / 2, -vy3 / 2, -vx3 / 2, -vy3 / 2, vx3, vy3]
    return System((1.0, 1.0, 1.0)), pos, vel


def pythagorean() -> tuple[System, list[float], list[float]]:
    """Burrau's problem: masses 3, 4, 5 at rest on a 3-4-5 triangle.
    Chaotic, with very close encounters -- a stress test, not a demo."""
    pos = [1.0, 3.0, -2.0, -1.0, 1.0, -1.0]
    return System((3.0, 4.0, 5.0)), pos, [0.0] * 6


def sun_planet_moon() -> tuple[System, list[float], list[float]]:
    """A heavy star with two planets on circular orbits: a gentle,
    nearly-integrable field to fly a probe through."""
    ms, m1, m2 = 1.0, 1e-3, 3e-4
    r1, r2 = 1.0, 1.6
    v1, v2 = math.sqrt(ms / r1), math.sqrt(ms / r2)
    pos = [0.0, 0.0, r1, 0.0, -r2, 0.0]
    vel = [0.0, 0.0, 0.0, v1, 0.0, -v2]
    system = System((ms, m1, m2))
    # shift to the centre-of-mass frame
    cx, cy = system.center_of_mass(pos)
    px, py = system.momentum(vel)
    M = system.total_mass
    pos = [p - (cx if k % 2 == 0 else cy) for k, p in enumerate(pos)]
    vel = [v - (px / M if k % 2 == 0 else py / M) for k, v in enumerate(vel)]
    return system, pos, vel


def binary_with_probe(
    r_probe: float,
    q: float = 1.0,
    separation: float = 1.0,
    probe_mass: float = 1e-9,
    softening: float = 0.02,
) -> tuple[System, list[float], list[float]]:
    """A circular binary (mass ratio ``q`` = m2/m1, total mass 1) plus a
    near-massless probe started at distance ``r_probe`` on the +y axis
    with the circular speed it would have around a point of mass 1.

    Close to the binary (r ~ 1.3 separations) the probe's motion is
    chaotic; far away (r >~ 4) it is a regular, nearly Keplerian orbit.
    Softening keeps close probe encounters integrable.
    """
    m1 = 1.0 / (1.0 + q)
    m2 = q / (1.0 + q)
    r1, r2 = separation * m2, separation * m1  # distances from the COM
    w = math.sqrt(1.0 / separation ** 3)  # binary angular speed, M = 1
    vp = math.sqrt(1.0 / r_probe)
    pos = [r1, 0.0, -r2, 0.0, 0.0, r_probe]
    vel = [0.0, w * r1, 0.0, -w * r2, -vp, 0.0]
    return System((m1, m2, probe_mass), softening=softening), pos, vel
