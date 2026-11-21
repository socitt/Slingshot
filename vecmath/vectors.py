"""Small vector toolkit, standard library only.

Two flavours live here:

* ``Vec2`` -- an immutable 2-D vector with operators, for readable
  geometry code (rendering, scoring, initial conditions).
* flat-list helpers (``axpy``, ``lincomb`` ...) -- operate on plain
  lists of floats and are what the integrators use in their hot loops,
  because allocating a Vec2 per operation is slow on iSH.
"""

from __future__ import annotations

import math

__all__ = [
    "Vec2",
    "add",
    "sub",
    "scale",
    "axpy",
    "lincomb",
    "dot",
    "norm",
    "max_abs_diff",
]


class Vec2:
    __slots__ = ("x", "y")

    def __init__(self, x: float = 0.0, y: float = 0.0) -> None:
        object.__setattr__(self, "x", float(x))
        object.__setattr__(self, "y", float(y))

    def __setattr__(self, name, value):  # immutable
        raise AttributeError("Vec2 is immutable")

    # --- construction -------------------------------------------------
    @classmethod
    def polar(cls, r: float, theta: float) -> "Vec2":
        return cls(r * math.cos(theta), r * math.sin(theta))

    # --- arithmetic ---------------------------------------------------
    def __add__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x + o.x, self.y + o.y)

    def __sub__(self, o: "Vec2") -> "Vec2":
        return Vec2(self.x - o.x, self.y - o.y)

    def __mul__(self, k: float) -> "Vec2":
        return Vec2(self.x * k, self.y * k)

    __rmul__ = __mul__

    def __truediv__(self, k: float) -> "Vec2":
        return Vec2(self.x / k, self.y / k)

    def __neg__(self) -> "Vec2":
        return Vec2(-self.x, -self.y)

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Vec2) and self.x == o.x and self.y == o.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __iter__(self):
        yield self.x
        yield self.y

    def __repr__(self) -> str:
        return f"Vec2({self.x!r}, {self.y!r})"

    # --- geometry -----------------------------------------------------
    def dot(self, o: "Vec2") -> float:
        return self.x * o.x + self.y * o.y

    def cross(self, o: "Vec2") -> float:
        """z-component of the 3-D cross product."""
        return self.x * o.y - self.y * o.x

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def norm2(self) -> float:
        return self.x * self.x + self.y * self.y

    def normalized(self) -> "Vec2":
        n = self.norm()
        if n == 0.0:
            raise ZeroDivisionError("cannot normalize the zero vector")
        return Vec2(self.x / n, self.y / n)

    def angle(self) -> float:
        return math.atan2(self.y, self.x)

    def rotated(self, theta: float) -> "Vec2":
        c, s = math.cos(theta), math.sin(theta)
        return Vec2(c * self.x - s * self.y, s * self.x + c * self.y)

    def perp(self) -> "Vec2":
        """Rotate +90 degrees."""
        return Vec2(-self.y, self.x)

    def is_close(self, o: "Vec2", tol: float = 1e-12) -> bool:
        return abs(self.x - o.x) <= tol and abs(self.y - o.y) <= tol


# --- flat-list helpers -----------------------------------------------

def add(a: list[float], b: list[float]) -> list[float]:
    return [x + y for x, y in zip(a, b)]


def sub(a: list[float], b: list[float]) -> list[float]:
    return [x - y for x, y in zip(a, b)]


def scale(a: list[float], k: float) -> list[float]:
    return [x * k for x in a]


def axpy(alpha: float, x: list[float], y: list[float]) -> list[float]:
    """Return ``alpha * x + y`` as a new list."""
    return [alpha * xi + yi for xi, yi in zip(x, y)]


def lincomb(base: list[float], terms: list[tuple[float, list[float]]]) -> list[float]:
    """Return ``base + sum(c_k * v_k)`` for ``terms = [(c_k, v_k), ...]``."""
    out = list(base)
    for c, v in terms:
        for i, vi in enumerate(v):
            out[i] += c * vi
    return out


def dot(a: list[float], b: list[float]) -> float:
    return math.fsum(x * y for x, y in zip(a, b))


def norm(a: list[float]) -> float:
    return math.sqrt(math.fsum(x * x for x in a))


def max_abs_diff(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"length mismatch: {len(a)} != {len(b)}")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)
