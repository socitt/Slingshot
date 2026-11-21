"""export_viz: write the data behind the slingshot visual explainer.

    python3 -m cli.export_viz --out viz/data.json          # full (minutes: re-runs training)
    python3 -m cli.export_viz --out /tmp/d.json --quick    # seconds: tiny learning replay
    python3 -m cli.export_viz --selftest
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from render.vizdata import AGENTS, build

FULL_LEARNING = {
    # the classic: a whole population converging, generation by generation
    # 37 is where ES evaluated its best-ever genome (the golden agent, 1.080)
    "free": dict(sample_gens=[0, 1, 2, 3, 5, 8, 12, 20, 30, 37, 60, 99], swarm=True, target_index=3),
    # the field: just the best flight of a few generations (64 genomes is a lot of lines)
    "trio": dict(sample_gens=[0, 5, 15, 40, 80, 150, 250, 499], swarm=False, target_index=4),
}
QUICK_LEARNING = {"free": dict(sample_gens=[0, 1, 2], swarm=True, generations=3, pop=6)}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv == ["--selftest"]:
        print("export_viz: ok")
        return 0
    p = argparse.ArgumentParser(prog="export_viz", description=__doc__.splitlines()[0])
    p.add_argument("--out", required=True)
    p.add_argument("--quick", action="store_true", help="tiny learning replay (for tests)")
    args = p.parse_args(argv)
    agents = sorted(f.stem.removeprefix("agent_") for f in AGENTS.glob("agent_*.json"))
    t0 = time.time()
    data = build(agents, QUICK_LEARNING if args.quick else FULL_LEARNING)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, separators=(",", ":")))
    print(f"wrote {out} ({out.stat().st_size // 1024} KB, agents: {', '.join(agents)}) "
          f"in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
