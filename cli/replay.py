"""replay: watch a trained agent fly, in the terminal.

    python3 -m cli.replay                      # golden assist agent, target 0
    python3 -m cli.replay --task free --target 3
    python3 -m cli.replay --agent agent.json --fps 30
    python3 -m cli.replay --plain              # final frame only, no ANSI
    python3 -m cli.replay --selftest
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from env.navigation import HELDOUT, TASKS, run_episode
from env.training import load_agent
from render.canvas import terminal_width, use_color
from render.replay import CLEAR, HOME, render_episode

GOLDEN = Path(__file__).resolve().parent.parent / "tests" / "golden" / "data" / "agents"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="replay", description=__doc__.splitlines()[0])
    p.add_argument("--task", default="assist", choices=sorted(TASKS))
    p.add_argument("--agent", help="agent JSON (default: the golden agent for --task)")
    p.add_argument("--target", type=int, default=0, help="target index")
    p.add_argument("--heldout", action="store_true", help="index into held-out targets")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--every", type=int, default=2, help="draw every Nth step")
    p.add_argument("--width", type=int, default=None)
    p.add_argument("--plain", action="store_true", help="no animation, no colour: final frame")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        print("replay: ok")
        return 0
    args = build_parser().parse_args(argv)
    task = TASKS[args.task]
    agent = Path(args.agent) if args.agent else GOLDEN / f"agent_{args.task}.json"
    if not agent.exists():
        print(f"replay: no agent at {agent}; train one with python3 -m cli.train --task "
              f"{args.task} --out {agent.name}", file=sys.stderr)
        return 2
    policy, meta = load_agent(agent)
    targets = HELDOUT[task.name] if args.heldout else task.targets
    if not 0 <= args.target < len(targets):
        print(f"replay: --target must be 0..{len(targets) - 1}", file=sys.stderr)
        return 2
    target = targets[args.target]
    result = run_episode(task, policy, target, record=True)
    width = args.width or terminal_width()
    masses = ()
    if task.field is not None:
        masses = task.field()[0].masses
    color = use_color() and not args.plain
    frames = render_episode(task, result, target, width, color, args.every, masses)
    if args.plain:
        for lines, last in frames:
            if last:
                print("\n".join(lines))
        return 0
    out = sys.stdout
    out.write(CLEAR)
    for lines, last in frames:
        out.write(HOME + "\n".join(line + "\x1b[K" for line in lines) + "\n")
        out.flush()
        if not last:
            time.sleep(1.0 / args.fps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
