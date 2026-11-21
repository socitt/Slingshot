"""Newtonian N-body gravity in 2-D, in nondimensional units (G = 1 by default).

Positions and velocities are flat lists ``[x0, y0, x1, y1, ...]``. A
full first-order state for ``rk4_step`` is ``pos + vel``.

Optional Plummer softening ``eps`` replaces ``r^2`` with ``r^2 + eps^2``
in both force and potential, so the two stay consistent and energy is
still exactly conserved by the continuous dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class System:
    masses: tuple[float, ...]
    G: float = 1.0
    softening: float = 0.0
    _eps2: float = field(init=False, repr=False)

    def __post_init__(self):
        object.__setattr__(self, "masses", tuple(float(m) for m in self.masses))
        if any(m <= 0 for m in self.masses):
            raise ValueError("masses must be positive")
        if self.softening < 0:
            raise ValueError("softening must be >= 0")
        object.__setattr__(self, "_eps2", self.softening * self.softening)

    @property
    def n(self) -> int:
        return len(self.masses)

    @property
    def total_mass(self) -> float:
        return math.fsum(self.masses)

    # --- forces -------------------------------------------------------
    def accelerations(self, pos: list[float]) -> list[float]:
        n, m, G, eps2 = self.n, self.masses, self.G, self._eps2
        acc = [0.0] * (2 * n)
        for i in range(n):
            xi, yi = pos[2 * i], pos[2 * i + 1]
            mi = m[i]
            for j in range(i + 1, n):
                dx = pos[2 * j] - xi
                dy = pos[2 * j + 1] - yi
                r2 = dx * dx + dy * dy + eps2
                s = G / (r2 * math.sqrt(r2))
                fx, fy = dx * s, dy * s
                mj = m[j]
                acc[2 * i] += mj * fx
                acc[2 * i + 1] += mj * fy
                acc[2 * j] -= mi * fx
                acc[2 * j + 1] -= mi * fy
        return acc

    def field_at(self, x: float, y: float, pos: list[float]) -> tuple[float, float]:
        """Acceleration a massless test particle (the probe) feels at (x, y)."""
        ax = ay = 0.0
        G, eps2 = self.G, self._eps2
        for j, mj in enumerate(self.masses):
            dx = pos[2 * j] - x
            dy = pos[2 * j + 1] - y
            r2 = dx * dx + dy * dy + eps2
            s = G * mj / (r2 * math.sqrt(r2))
            ax += dx * s
            ay += dy * s
        return ax, ay

    def derivative(self, t: float, state: list[float]) -> list[float]:
        """``f(t, y)`` for rk4: d(pos, vel)/dt = (vel, acc)."""
        k = 2 * self.n
        return state[k:] + self.accelerations(state[:k])

    # --- invariants ---------------------------------------------------
    def kinetic_energy(self, vel: list[float]) -> float:
        return 0.5 * math.fsum(
            m * (vel[2 * i] ** 2 + vel[2 * i + 1] ** 2)
            for i, m in enumerate(self.masses)
        )

    def potential_energy(self, pos: list[float]) -> float:
        n, m, eps2 = self.n, self.masses, self._eps2
        terms = []
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[2 * j] - pos[2 * i]
                dy = pos[2 * j + 1] - pos[2 * i + 1]
                terms.append(-self.G * m[i] * m[j] / math.sqrt(dx * dx + dy * dy + eps2))
        return math.fsum(terms)

    def energy(self, pos: list[float], vel: list[float]) -> float:
        return self.kinetic_energy(vel) + self.potential_energy(pos)

    def momentum(self, vel: list[float]) -> tuple[float, float]:
        m = self.masses
        return (
            math.fsum(m[i] * vel[2 * i] for i in range(self.n)),
            math.fsum(m[i] * vel[2 * i + 1] for i in range(self.n)),
        )

    def angular_momentum(self, pos: list[float], vel: list[float]) -> float:
        """z-component of total angular momentum about the origin."""
        return math.fsum(
            m * (pos[2 * i] * vel[2 * i + 1] - pos[2 * i + 1] * vel[2 * i])
            for i, m in enumerate(self.masses)
        )

    def center_of_mass(self, pos: list[float]) -> tuple[float, float]:
        M = self.total_mass
        return (
            math.fsum(m * pos[2 * i] for i, m in enumerate(self.masses)) / M,
            math.fsum(m * pos[2 * i + 1] for i, m in enumerate(self.masses)) / M,
        )
