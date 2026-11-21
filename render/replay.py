"""Terminal replay of a navigation episode: plain text plus optional
ANSI colour, no curses, sized for a phone.

Glyphs: ``*`` star (heaviest body), ``o`` planets, ``@`` probe,
``.`` probe trail, ``X`` target, ``+`` target zone edge, ``=`` thrust
direction marker next to the probe.
"""

from __future__ import annotations

import math

from render.canvas import framed

RESET = "\x1b[0m"
COLORS = {"*": "\x1b[33;1m", "o": "\x1b[36;1m", "@": "\x1b[37;1m", "X": "\x1b[31;1m",
          "+": "\x1b[31m", ".": "\x1b[90m", "=": "\x1b[35m"}
HOME, CLEAR = "\x1b[H", "\x1b[2J"


class View:
    """Maps world coordinates onto a character grid. Terminal cells are
    about twice as tall as wide, so rows cover twice the world height of
    a column's width and circles stay round."""

    def __init__(self, extent: float, width: int):
        if width < 10:
            raise ValueError("width must be >= 10")
        self.extent, self.width = extent, width
        self.height = max(5, width // 2)

    def cell(self, x: float, y: float) -> tuple[int, int] | None:
        e = self.extent
        c = int((x + e) / (2 * e) * self.width)
        r = int((e - y) / (2 * e) * self.height)
        if 0 <= c < self.width and 0 <= r < self.height:
            return r, c
        return None


def view_for(task, target, width: int) -> View:
    far = max([math.hypot(*target), math.hypot(*task.start)] +
              [math.hypot(*t) for t in task.targets])
    return View(extent=far * 1.15, width=width)


def draw(view: View, frame: dict, trail: list, target, target_radius: float,
         body_masses: tuple = (), color: bool = False) -> list[str]:
    grid = [[" "] * view.width for _ in range(view.height)]

    def put(x, y, ch, overwrite=True):
        rc = view.cell(x, y)
        if rc and (overwrite or grid[rc[0]][rc[1]] == " "):
            grid[rc[0]][rc[1]] = ch

    for k in range(16):  # target zone ring
        a = 2 * math.pi * k / 16
        put(target[0] + target_radius * math.cos(a), target[1] + target_radius * math.sin(a), "+", False)
    for x, y in trail:
        put(x, y, ".")
    put(*target, "X")
    bodies = frame.get("bodies") or []
    heaviest = max(range(len(body_masses)), key=body_masses.__getitem__) if body_masses else 0
    for j in range(len(bodies) // 2):
        put(bodies[2 * j], bodies[2 * j + 1], "*" if j == heaviest else "o")
    px, py = frame["probe"]
    tx, ty = frame.get("thrust", (0.0, 0.0))
    if tx or ty:
        n = math.hypot(tx, ty)
        step = 2 * view.extent / view.width
        put(px - 1.5 * step * tx / n, py - 1.5 * step * ty / n, "=")  # exhaust, behind the probe
    put(px, py, "@")

    lines = []
    for row in grid:
        if color:
            lines.append("".join(COLORS[c] + c + RESET if c in COLORS else c for c in row))
        else:
            lines.append("".join(row))
    return lines


def hud(result, frame: dict, task, step: int, total: int, width: int) -> list[str]:
    fuel_frac = max(0.0, 1.0 - frame.get("fuel_used", 0.0) / task.fuel)
    bar_w = max(4, width - 18)
    filled = round(fuel_frac * bar_w)
    lines = [
        f"t {frame['t']:5.2f}  step {step:3d}/{total:<3d}"[:width],
        f"fuel [{'#' * filled}{'-' * (bar_w - filled)}]"[:width],
    ]
    if step == total:
        outcome = ("REACHED" if result.reached else "CRASHED" if result.crashed
                   else "LOST" if result.out_of_bounds else "missed")
        lines.append(f"{outcome}  score {result.score:+.3f}"[:width])
        if result.work:
            assists = "  ".join(f"{w:+.2f}" for w in result.work)
            lines.append(f"work by body: {assists}"[:width])
            planet = sum(max(0.0, w) for w in result.work[1:])
            if planet > 0.05:
                lines.append(f"gravity assist: +{planet:.2f} energy"[:width])
    return lines


def annotate_fuel(frames: list, task) -> None:
    """Add cumulative ``fuel_used`` to recorded frames (from thrust)."""
    used = 0.0
    for f in frames:
        f["fuel_used"] = used
        used += math.hypot(*f["thrust"]) * task.dt


def render_episode(task, result, target, width: int, color: bool = False,
                   every: int = 1, body_masses: tuple = ()):
    """Yield (lines, is_last) per displayed frame."""
    frames = result.frames
    annotate_fuel(frames, task)
    view = view_for(task, target, width - 2)  # minus the frame
    trail = []
    total = len(frames) - 1
    for i, f in enumerate(frames):
        trail.append(tuple(f["probe"]))
        if i % every and i != total:
            continue
        lines = draw(view, f, trail[:-1], target, task.target_radius, body_masses, color)
        yield framed(lines, task.name, width=view.width) + hud(result, f, task, i, total, width), i == total
