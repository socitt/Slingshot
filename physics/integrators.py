"""Time integrators over flat lists of floats.

* ``rk4_step`` -- classic 4th-order Runge-Kutta for any first-order ODE
  ``dy/dt = f(t, y)``. General purpose (also used for Lorenz in chaos/),
  but not symplectic: energy drifts secularly over long runs.
* ``leapfrog_step`` -- kick-drift-kick velocity Verlet for separable
  Hamiltonians with velocity-independent forces. 2nd order, symplectic
  and time-reversible: energy error stays bounded and oscillates rather
  than drifting.
"""

from __future__ import annotations

from typing import Callable

from vecmath.vectors import axpy

Derivative = Callable[[float, list[float]], list[float]]
Accel = Callable[[list[float]], list[float]]


def rk4_step(f: Derivative, t: float, y: list[float], h: float) -> list[float]:
    k1 = f(t, y)
    k2 = f(t + 0.5 * h, axpy(0.5 * h, k1, y))
    k3 = f(t + 0.5 * h, axpy(0.5 * h, k2, y))
    k4 = f(t + h, axpy(h, k3, y))
    h6 = h / 6.0
    return [
        yi + h6 * (a + 2.0 * b + 2.0 * c + d)
        for yi, a, b, c, d in zip(y, k1, k2, k3, k4)
    ]


def rk4_integrate(
    f: Derivative, t0: float, y0: list[float], h: float, steps: int
) -> list[float]:
    """Advance ``steps`` RK4 steps and return the final state only."""
    t, y = t0, list(y0)
    for _ in range(steps):
        y = rk4_step(f, t, y, h)
        t += h
    return y


def leapfrog_step(
    accel: Accel,
    x: list[float],
    v: list[float],
    h: float,
    a: list[float] | None = None,
) -> tuple[list[float], list[float], list[float]]:
    """One kick-drift-kick step.

    Returns ``(x', v', a')``. Pass the returned ``a'`` back in as ``a``
    on the next call to reuse it: one force evaluation per step instead
    of two.
    """
    if a is None:
        a = accel(x)
    half = 0.5 * h
    v_half = axpy(half, a, v)
    x_new = axpy(h, v_half, x)
    a_new = accel(x_new)
    v_new = axpy(half, a_new, v_half)
    return x_new, v_new, a_new


def leapfrog_integrate(
    accel: Accel, x0: list[float], v0: list[float], h: float, steps: int
) -> tuple[list[float], list[float]]:
    x, v, a = list(x0), list(v0), None
    for _ in range(steps):
        x, v, a = leapfrog_step(accel, x, v, h, a)
    return x, v
