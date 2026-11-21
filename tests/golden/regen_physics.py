"""Regenerate golden physics trajectories.

    python3 -m tests.golden.regen_physics          # all cases
    python3 -m tests.golden.regen_physics figure8_rk4

Only regenerate after an intentional numerical change, and say why in
the commit message.
"""

from __future__ import annotations

import json
import sys

from tests.golden.physics_cases import CASES, DATA_DIR, golden_path, run_case


def main(argv: list[str] | None = None) -> int:
    names = (argv if argv is not None else sys.argv[1:]) or sorted(CASES)
    DATA_DIR.mkdir(exist_ok=True)
    for name in names:
        golden_path(name).write_text(json.dumps(run_case(name), indent=1) + "\n")
        print(f"wrote {golden_path(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
