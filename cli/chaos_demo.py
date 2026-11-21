"""chaos-demo: the chaos lab in a phone-width terminal.

    python3 -m cli.chaos_demo bifurcation [--rmin 2.8 --rmax 4.0]
    python3 -m cli.chaos_demo lorenz
    python3 -m cli.chaos_demo lyapunov
    python3 -m cli.chaos_demo --selftest
"""

from __future__ import annotations

import argparse
import math
import sys

from chaos.logistic import bifurcation, feigenbaum_estimates, logistic, logistic_deriv
from chaos.lorenz import divergence, lorenz, lorenz_jacobian, trajectory
from chaos.lyapunov import lyapunov_map, lyapunov_spectrum, lyapunov_two_trajectory
from physics.integrators import rk4_step
from physics.orbits import binary_with_probe
from render.canvas import density_plot, framed, terminal_width


def cmd_bifurcation(args) -> list[str]:
    w, h = args.width - 2, args.height
    cols = bifurcation(args.rmin, args.rmax, columns=w * 3, keep=args.height * 6)
    pts = [(r, x) for r, xs in cols for x in xs]
    rows = density_plot(pts, w, h, xlim=(args.rmin, args.rmax), ylim=(0.0, 1.0))
    lines = framed(rows, "logistic map x->rx(1-x)", f"r {args.rmin:g}..{args.rmax:g}")
    est = feigenbaum_estimates(8)[-1]
    lines.append(f"Feigenbaum delta ~ {est:.5f} (4.66920)")
    return lines


def cmd_lorenz(args) -> list[str]:
    traj = trajectory([1.0, 1.0, 1.0], 0.01, args.steps)[200:]
    rows = density_plot([(p[0], p[2]) for p in traj], args.width - 2, args.height)
    return framed(rows, "Lorenz x-z", f"{len(traj)} pts, rk4 h=0.01")


def cmd_lyapunov(args) -> list[str]:
    lines = ["Lyapunov exponents", "", "logistic map:"]
    for r in (2.9, 3.2, 3.5, 3.7, 3.83, 4.0):
        lam = lyapunov_map(lambda x: logistic(r, x), lambda x: logistic_deriv(r, x),
                           0.2, args.samples)
        tag = "chaos" if lam > 1e-3 else "order"
        lines.append(f"  r={r:<5g} {lam:+.4f}  {tag}")
    lines.append(f"  ln 2 = {math.log(2):.4f}")
    spec = lyapunov_spectrum(lorenz(), lorenz_jacobian(), [1.0, 1.0, 1.0],
                             0.01, args.samples // 2, 10, 500)
    lines += ["", "Lorenz spectrum:",
              "  " + "  ".join(f"{s:+.3f}" for s in spec),
              f"  sum {sum(spec):+.3f} (trace {divergence():+.3f})", "",
              "probe around a binary:"]
    for r in (1.3, 4.0):
        system, pos, vel = binary_with_probe(r)
        f, hh = system.derivative, 2e-3
        d = [0.0] * 12
        d[4] = 1.0
        lam = lyapunov_two_trajectory(lambda y: rk4_step(f, 0.0, y, hh), pos + vel,
                                      hh, args.samples // 5, renorm_every=25, direction=d)
        lines.append(f"  r={r:<4g} lambda ~ {lam:+.3f}")
    return lines


COMMANDS = {"bifurcation": cmd_bifurcation, "lorenz": cmd_lorenz, "lyapunov": cmd_lyapunov}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="chaos-demo", description=__doc__.splitlines()[0])
    p.add_argument("command", choices=sorted(COMMANDS))
    p.add_argument("--width", type=int, default=None, help="columns (default: terminal, max 60)")
    p.add_argument("--height", type=int, default=18)
    p.add_argument("--rmin", type=float, default=2.8)
    p.add_argument("--rmax", type=float, default=4.0)
    p.add_argument("--steps", type=int, default=6000, help="Lorenz steps")
    p.add_argument("--samples", type=int, default=40000, help="Lyapunov iterations")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        print("chaos-demo: ok")
        return 0
    args = build_parser().parse_args(argv)
    if args.width is None:
        args.width = terminal_width()
    if args.width < 12 or args.height < 4:
        print("chaos-demo: need --width >= 12 and --height >= 4", file=sys.stderr)
        return 2
    print("\n".join(COMMANDS[args.command](args)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
