"""train: evolve a navigation agent and save its weights.

    python3 -m cli.train --task free --generations 60 --out agent.json
    python3 -m cli.train --task free --strategy es --pop 32 --seed 7
    python3 -m cli.train --selftest
"""

from __future__ import annotations

import argparse
import sys
import time

from env.navigation import FREE_HELDOUT, TASKS
from env.training import evaluate, load_agent, save_agent, task_by_name, train
from evolve.strategies import STRATEGIES

BAR = " ▁▂▃▄▅▆▇█"


def bar(score: float, lo: float = 0.0, hi: float = 1.5, width: int = 10) -> str:
    frac = max(0.0, min(1.0, (score - lo) / (hi - lo)))
    full, part = divmod(round(frac * width * 8), 8)
    return ("█" * full + (BAR[part] if part else "")).ljust(width)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="train", description=__doc__.splitlines()[0])
    p.add_argument("--task", default="free", choices=sorted(TASKS))
    p.add_argument("--strategy", default="ga", choices=sorted(STRATEGIES))
    p.add_argument("--generations", type=int, default=60)
    p.add_argument("--pop", type=int, default=24)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", help="start from this agent's weights (curriculum)")
    p.add_argument("--out", help="write the best agent here (JSON)")
    p.add_argument("--quiet", action="store_true", help="only print the summary")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        print("train: ok")
        return 0
    args = build_parser().parse_args(argv)
    task = task_by_name(args.task)
    init = None
    if args.init:
        policy, _ = load_agent(args.init)
        init = policy.params

    def report(g):
        if not args.quiet:
            print(f"gen {g.index:3d} {g.best:+.3f} {bar(g.best)} avg {g.mean:+.3f}", flush=True)

    t0 = time.time()
    res = train(task, args.strategy, args.generations, args.pop, args.seed, init, report)
    elapsed = time.time() - t0
    from env.agent import policy_from_genome
    net = policy_from_genome(res.genome)
    summary = {"task": task.name, "strategy": args.strategy, "generations": args.generations,
               "pop": args.pop, "seed": args.seed, "score": res.score}
    if task.name == "free":
        summary["heldout_score"] = evaluate(task, net, FREE_HELDOUT)
    print(f"best {res.score:+.4f} in {elapsed:.1f}s ({args.strategy}, seed {args.seed})")
    if "heldout_score" in summary:
        print(f"held-out targets {summary['heldout_score']:+.4f}")
    if args.out:
        save_agent(args.out, res.genome, **summary)
        print(f"saved {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
