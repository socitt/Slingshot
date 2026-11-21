"""The Lorenz system, integrated with the shared RK4 stepper."""

from __future__ import annotations

from physics.integrators import rk4_step

SIGMA, RHO, BETA = 10.0, 28.0, 8.0 / 3.0


def lorenz(sigma: float = SIGMA, rho: float = RHO, beta: float = BETA):
    """Return ``f(t, y)`` for the Lorenz equations."""
    def f(t, y):
        x, yy, z = y
        return [sigma * (yy - x), x * (rho - z) - yy, x * yy - beta * z]
    return f


def lorenz_jacobian(sigma: float = SIGMA, rho: float = RHO, beta: float = BETA):
    def jac(y):
        x, yy, z = y
        return [[-sigma, sigma, 0.0],
                [rho - z, -1.0, -x],
                [yy, x, -beta]]
    return jac


def trajectory(y0: list[float], h: float, steps: int, every: int = 1,
               sigma: float = SIGMA, rho: float = RHO, beta: float = BETA) -> list[list[float]]:
    f = lorenz(sigma, rho, beta)
    y, out = list(y0), [list(y0)]
    for i in range(1, steps + 1):
        y = rk4_step(f, (i - 1) * h, y, h)
        if i % every == 0:
            out.append(y)
    return out


def divergence(sigma: float = SIGMA, beta: float = BETA) -> float:
    """Phase-space volume contraction rate (trace of the Jacobian), a
    constant: the Lyapunov exponents must sum to it."""
    return -(sigma + 1.0 + beta)
