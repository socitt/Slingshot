"""Regenerate golden reference data.

    python3 -m tests.golden.regen                 # every suite, every case
    python3 -m tests.golden.regen physics         # one suite
    python3 -m tests.golden.regen chaos lorenz_short
    python3 -m tests.golden.regen agents free      # retrains: slow

Only regenerate after an intentional numerical change, and say why in
the commit message.
"""

from __future__ import annotations

import json
import sys

from tests.golden import agent_cases, chaos_cases, physics_cases

SUITES = {"physics": physics_cases, "chaos": chaos_cases, "agents": agent_cases}


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    suites = [argv[0]] if argv else sorted(SUITES)
    for suite in suites:
        if suite not in SUITES:
            print(f"regen: unknown suite {suite!r}; choose from {sorted(SUITES)}", file=sys.stderr)
            return 2
        mod = SUITES[suite]
        mod.DATA_DIR.mkdir(exist_ok=True)
        for name in argv[1:] or sorted(mod.CASES):
            path = mod.golden_path(name)
            path.write_text(json.dumps(mod.run_case(name), indent=1) + "\n")
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
