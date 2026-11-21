"""Lyapunov exponents three ways.

* ``lyapunov_map`` -- 1-D maps, averaging log|f'(x)| along an orbit.
* ``lyapunov_spectrum`` -- full spectrum of a flow from its Jacobian,
  integrating the variational equations with RK4 and re-orthonormalising
  (Gram-Schmidt) every few steps (Benettin et al. 1980).
* ``lyapunov_two_trajectory`` -- largest exponent of anything with a
  step function, from the growth of a tiny separation between two
  copies, renormalised periodically. Needs no Jacobian, so it works on
  the N-body problem directly.
"""

from __future__ import annotations

import math
from typing import Callable

from physics.integrators import rk4_step


def lyapunov_map(f: Callable[[float], float], df: Callable[[float], float],
                 x0: float, n: int, transient: int = 1000) -> float:
    x = x0
    for _ in range(transient):
        x = f(x)
    total = 0.0
    for _ in range(n):
        d = abs(df(x))
        if d == 0.0:
            return -math.inf  # superstable: the orbit hit the critical point
        total += math.log(d)
        x = f(x)
    return total / n


def gram_schmidt(vectors: list[list[float]]) -> tuple[list[list[float]], list[float]]:
    """Modified Gram-Schmidt. Returns (orthonormal vectors, norms after
    projection) -- the norms are the per-direction stretch factors."""
    basis, norms = [], []
    for v in vectors:
        w = list(v)
        scale = math.sqrt(sum(a * a for a in v))
        for q in basis:
            c = sum(a * b for a, b in zip(w, q))
            w = [a - c * b for a, b in zip(w, q)]
        nrm = math.sqrt(sum(a * a for a in w))
        if nrm <= 1e-12 * scale:
            raise ArithmeticError("tangent vectors became linearly dependent")
        basis.append([a / nrm for a in w])
        norms.append(nrm)
    return basis, norms


def lyapunov_spectrum(f, jac, y0: list[float], h: float, steps: int,
                      renorm_every: int = 10, transient_steps: int = 0,
                      k: int | None = None) -> list[float]:
    """The ``k`` largest Lyapunov exponents (default: all) of
    ``dy/dt = f(t, y)`` with Jacobian ``jac(y)``."""
    n = len(y0)
    k = n if k is None else k
    y, t = list(y0), 0.0
    for _ in range(transient_steps):
        y = rk4_step(f, t, y, h)
        t += h

    def extended(t, Y):
        x = Y[:n]
        J = jac(x)
        out = f(t, x)
        for j in range(k):
            v = Y[n + j * n: n + (j + 1) * n]
            out.extend(sum(Ji[c] * v[c] for c in range(n)) for Ji in J)
        return out

    Q = [[1.0 if i == j else 0.0 for i in range(n)] for j in range(k)]
    sums = [0.0] * k
    elapsed = 0.0
    for step in range(1, steps + 1):
        Y = rk4_step(extended, t, y + [c for v in Q for c in v], h)
        t += h
        y = Y[:n]
        Q = [Y[n + j * n: n + (j + 1) * n] for j in range(k)]
        if step % renorm_every == 0:
            Q, norms = gram_schmidt(Q)
            for j in range(k):
                sums[j] += math.log(norms[j])
            elapsed = step * h
    if elapsed == 0.0:
        raise ValueError("steps must be at least renorm_every")
    return [s / elapsed for s in sums]


def lyapunov_two_trajectory(step: Callable[[list[float]], list[float]],
                            y0: list[float], dt: float, steps: int,
                            d0: float = 1e-9, renorm_every: int = 10,
                            direction: list[float] | None = None) -> float:
    """Largest Lyapunov exponent from two nearby trajectories. ``step``
    advances a state by ``dt``. The separation is measured in the full
    state space and rescaled to ``d0`` every ``renorm_every`` steps."""
    n = len(y0)
    if direction is None:
        direction = [1.0] * n
    nd = math.sqrt(sum(c * c for c in direction))
    a = list(y0)
    b = [ai + d0 * c / nd for ai, c in zip(a, direction)]
    total, elapsed = 0.0, 0.0
    for i in range(1, steps + 1):
        a, b = step(a), step(b)
        if i % renorm_every == 0:
            d = math.sqrt(sum((bi - ai) ** 2 for ai, bi in zip(a, b)))
            if d == 0.0:
                raise ArithmeticError("trajectories coincided")
            total += math.log(d / d0)
            b = [ai + (bi - ai) * (d0 / d) for ai, bi in zip(a, b)]
            elapsed = i * dt
    if elapsed == 0.0:
        raise ValueError("steps must be at least renorm_every")
    return total / elapsed
