"""The logistic map x -> r x (1 - x): period doubling, windows, chaos."""

from __future__ import annotations

import math

#: Feigenbaum's first constant.
FEIGENBAUM_DELTA = 4.669201609102990


def logistic(r: float, x: float) -> float:
    return r * x * (1.0 - x)


def logistic_deriv(r: float, x: float) -> float:
    """d/dx of the map."""
    return r * (1.0 - 2.0 * x)


def orbit(r: float, x0: float, n: int, transient: int = 0) -> list[float]:
    x = x0
    for _ in range(transient):
        x = r * x * (1.0 - x)
    out = []
    for _ in range(n):
        x = r * x * (1.0 - x)
        out.append(x)
    return out


def attractor(r: float, x0: float = 0.2, transient: int = 4000,
              keep: int = 512, digits: int = 5) -> list[float]:
    """Distinct points the orbit visits after the transient, rounded to
    ``digits``. A period-p cycle gives p points; chaos gives ~``keep``."""
    return sorted(set(round(x, digits) for x in orbit(r, x0, keep, transient)))


def period(r: float, max_period: int = 64, **kw) -> int | None:
    """Period of the attracting cycle at ``r``, or None if it looks
    chaotic (more than ``max_period`` distinct points)."""
    p = len(attractor(r, **kw))
    return p if p <= max_period else None


def bifurcation(r_min: float, r_max: float, columns: int, x0: float = 0.2,
                transient: int = 1000, keep: int = 120) -> list[tuple[float, list[float]]]:
    """Sample the bifurcation diagram: for each of ``columns`` evenly
    spaced r values, the post-transient orbit points."""
    if columns < 2:
        raise ValueError("need at least 2 columns")
    step = (r_max - r_min) / (columns - 1)
    return [(r_min + i * step, orbit(r_min + i * step, x0, keep, transient))
            for i in range(columns)]


def superstable_r(n: int, guess: float, tol: float = 1e-13, max_iter: int = 60) -> float:
    """Parameter r at which x = 1/2 lies on a cycle of period 2**n
    (Newton's method on f_r^(2^n)(1/2) - 1/2, differentiating in r)."""
    r = guess
    m = 2 ** n
    for _ in range(max_iter):
        x, dx = 0.5, 0.0
        for _ in range(m):
            dx = x * (1.0 - x) + r * (1.0 - 2.0 * x) * dx
            x = r * x * (1.0 - x)
        step = (x - 0.5) / dx
        r -= step
        if abs(step) < tol:
            return r
    raise ArithmeticError(f"Newton did not converge for n={n} from {guess}")


def superstable_sequence(count: int) -> list[float]:
    """R_0 .. R_{count-1}: superstable parameters of the period-doubling
    cascade (R_0 = 2, R_1 = 1 + sqrt 5, ...). Each guess extrapolates
    with Feigenbaum's delta, so Newton lands on the right root."""
    rs = [2.0, 1.0 + math.sqrt(5.0)]
    while len(rs) < count:
        guess = rs[-1] + (rs[-1] - rs[-2]) / FEIGENBAUM_DELTA
        rs.append(superstable_r(len(rs), guess))
    return rs[:count]


def feigenbaum_estimates(count: int) -> list[float]:
    """Successive ratios (R_{n} - R_{n-1}) / (R_{n+1} - R_n) -> delta."""
    rs = superstable_sequence(count)
    return [(rs[i] - rs[i - 1]) / (rs[i + 1] - rs[i]) for i in range(1, count - 1)]
