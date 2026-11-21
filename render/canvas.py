"""Plain-text plotting for narrow terminals (phones): no curses needed.

Points are binned into a character grid and each cell is drawn with a
density ramp, so dense regions (a bifurcation diagram's chaotic bands,
the Lorenz wings) read as darker.
"""

from __future__ import annotations

import os
import shutil

RAMP = " .:-=+*#%@"


def terminal_width(default: int = 48, cap: int = 60) -> int:
    """Width to draw at: the terminal's, capped so phones and wide
    terminals both get something readable."""
    cols = shutil.get_terminal_size((default, 24)).columns
    return max(20, min(cols, cap))


def use_color() -> bool:
    return os.environ.get("NO_COLOR") is None and os.environ.get("TERM", "dumb") != "dumb"


def _limits(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


def density_plot(
    points: list[tuple[float, float]],
    width: int,
    height: int,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    ramp: str = RAMP,
) -> list[str]:
    """Return ``height`` strings of exactly ``width`` characters. y grows
    upward. Points outside the limits are dropped."""
    if width < 1 or height < 1:
        raise ValueError("width and height must be positive")
    if not points:
        return [" " * width for _ in range(height)]
    x0, x1 = xlim or _limits([p[0] for p in points])
    y0, y1 = ylim or _limits([p[1] for p in points])
    counts = [[0] * width for _ in range(height)]
    for x, y in points:
        if not (x0 <= x <= x1 and y0 <= y <= y1):
            continue
        c = min(width - 1, int((x - x0) / (x1 - x0) * width))
        r = min(height - 1, int((y - y0) / (y1 - y0) * height))
        counts[height - 1 - r][c] += 1
    peak = max(max(row) for row in counts) or 1
    top = len(ramp) - 1
    rows = []
    for row in counts:
        chars = []
        for n in row:
            if n == 0:
                chars.append(ramp[0])
            else:
                # sqrt scaling keeps sparse-but-present cells visible
                level = max(1, round(top * (n / peak) ** 0.5))
                chars.append(ramp[level])
        rows.append("".join(chars))
    return rows


def framed(rows: list[str], title: str = "", footer: str = "") -> list[str]:
    """Box the plot; the result is ``len(rows[0]) + 2`` wide."""
    w = len(rows[0]) if rows else len(title)
    head = f" {title} "[:w] if title else ""
    out = ["+" + head + "-" * (w - len(head)) + "+"]
    out += ["|" + r + "|" for r in rows]
    foot = f" {footer} "[:w] if footer else ""
    out.append("+" + foot + "-" * (w - len(foot)) + "+")
    return out
