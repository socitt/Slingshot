# lirk notes (slingshot)

slingshot is lirk's 4th real repo, built and tested with lirk from the
first commit. This file is the feedback channel: invocation tally, every
`lirk verify` result, and every bit of friction, with exact commands and
output. Nothing is worked around silently; where a workaround exists in
this repo, the entry says so.

- lirk version: `lirk 1.0.0`, installed with `pip install -e ../lirk`
  (checkout at commit `584a733`).
- Python 3.13.18, pytest 9.1.1, Linux container (not yet on iSH).
- Raw log: [`docs/.lirk-invocations.log`](.lirk-invocations.log) — one
  line per `lirk` invocation (UTC time, exit code, seconds, args),
  written by a 6-line shell wrapper that runs `lirk "$@"` unchanged.

## Invocation tally

| Milestone | build | test | fix | verify | total | `signal: hangup` | cache disagreements |
|---|---|---|---|---|---|---|---|
| M1 physics core | 4 | 7 | 3 | 1 | 15 | 0 | 0 |

Running total: **15**. (Updated when committing; the log is authoritative.)

## `lirk verify` results

| Date | Command | Result |
|---|---|---|
| 2026-11-21 | `lirk verify --sample 5` (seed 1720564241) | 5 verified, 0 stale, 1 target(s) not cached — OK |

## Cache-correctness probes

- **Golden `data` directory invalidation — correct.** Perturbed one
  float in `tests/golden/data/physics_figure8_rk4.json` by 1e-6:
  `lirk test //tests/golden:physics_golden_test` re-ran and FAILED as it
  should. Restored the file: `lirk test //tests/golden/...` reported
  `cached //tests/golden:physics_golden_test`, i.e. it returned to the
  previous success, as documented ("reverting an edit returns you to the
  cached result").
- **Adding `pytest.ini` at the root re-ran exactly the pytest targets**
  and left the unittest target (`//physics:kepler_unittest`) cached.
  Correct.

## Friction, bugs and confusing messages

### 1. BUG (high): root `conftest.py` is fingerprinted but never loaded by pytest

*Status: open in lirk. Worked around in this repo with a root `pytest.ini` (explicit, commented).*

The docs say "Fixtures, `assert` rewriting and `conftest.py` all work as
usual" and that "every `conftest.py` ... between the package and the
repo root goes into the target's fingerprint". The second is true; the
first is not for a conftest above the package directory. lirk runs
`python3 -m pytest -p no:cacheprovider <src>` with `cwd` = package dir.
With no ini file anywhere, pytest picks that package dir as `rootdir`,
and `confcutdir` defaults to rootdir, so the repo-root `conftest.py` is
never collected.

Repro (root `conftest.py` defines an `rng` fixture; `physics/test_gravity.py` uses it):

```
$ lirk test //...
...
  FAIL   //physics:gravity_test: 1 of 1 src files failed: test_gravity
============================= test session starts ==============================
platform linux -- Python 3.13.18, pytest-9.1.1, pluggy-1.6.0
rootdir: /home/user/Slingshot/physics
...
E       fixture 'rng' not found
...
========================= 6 passed, 3 errors in 0.02s ==========================
```

Same file from the repo root passes (`python3 -m pytest physics/test_gravity.py` → `9 passed`).

**Why it matters more than a missing fixture:** an *autouse* fixture in
the root conftest fails silently. This repo's conftest has one that
poisons the global `random` functions to enforce seeding; in the first
`lirk test` run, `//vecmath:vectors_test` went green with that guard not
active at all. That is a false green of exactly the kind lirk exists to
prevent — and lirk is meanwhile re-running tests whenever that inert
conftest changes.

Workaround here: `pytest.ini` at the repo root (`[pytest]` section) pins
rootdir; output then shows `rootdir: /home/user/Slingshot`,
`configfile: pytest.ini`. Suggested fix in lirk: pass
`--rootdir=<repo root>` (or `--confcutdir=<repo root>`) in the pytest
argv in `lirk/actions.py`, or at minimum document that a root ini file
is required for a root conftest to load.

### 2. Docs: getting-started command reference is stale

`docs/getting-started.md`, "Command reference":
- "The entire CLI is two subcommands:" — the table below lists five
  (`build`, `test`, `verify`, `fix`, `clean`).
- The `lirk test` row says "Run each test src via `python3 -m
  unittest`", ignoring `runner = "pytest"`.
- The flags table's "Applies to" column says "both" / "all three"
  (`--root` "all three"), written before `verify`/`fix`/`clean` existed.
- "Two behaviours worth knowing up front:" is followed by three bullets.

### 3. Docs: contradictory statements about non-`.py` files in `srcs`

- build-format.md, `srcs`: non-`.py` files are "rejected when the BUILD
  file is parsed, with a message pointing at `data`".
- build-format.md, `data`: "putting a non-Python file in `srcs` instead
  produces a bogus syntax error."
- getting-started.md Troubleshooting still lists
  "`FAIL ...: fixture.txt: syntax error: invalid syntax`. A non-Python
  file is in `srcs`."

The parse-time rejection is presumably current; the other two are stale.

### 4. Docs: three different gitignore lists

README and build-format.md list `.lirk-cache.json`, `__pycache__/`,
`*.pyc`; getting-started.md adds `.lirk-cache.json.*.tmp`. This repo
uses all four.

### 5. `lirk fix --dry-run` labels every line `fixed`

```
$ lirk fix --dry-run
  fixed  //physics:conservation_test: would add :orbits, :simulate
...
lirk: would add 11 dep(s) to 8 target(s), 0 need a hand fix
lirk: dry run -- no files were changed
```

The trailer is clear, but the per-line status `fixed` contradicts
"would add". Something like `would-fix` / `WOULD` reads truer.

### 6. `lirk fix` inserts `deps` between `name` and `type` (cosmetic)

Starting from `name`, `type`, `srcs` tables, `lirk fix` wrote:

```toml
[[target]]
name = "integrators"
deps = ["//vecmath:vectors"]
type = "library"
srcs = ["integrators.py"]
```

Valid TOML, comments preserved, but it doesn't match the key order used
everywhere in lirk's own docs (`name`, `type`, `srcs`, `deps`). Appending
after the last key would keep hand-written files looking hand-written.
Left as lirk wrote it.

### 7. `lirk verify` labels libraries `(test)`, and "not cached" doesn't say which

```
$ lirk verify --sample 5
lirk: sampling 5 of 15 cached results (seed 1720564241)
  verified //physics:gravity (build)
  verified //physics:orbits (test)
  verified //vecmath:vectors (build)
  verified //physics:simulate (test)
  verified //tests/golden:physics_cases (test)
lirk: 5 verified, 0 stale, 1 target(s) not cached
```

`//physics:orbits` is a library. Reading `.lirk-cache.json` shows keys
are namespaced by mode (`build://...`, `test://...`), so "(test)" means
"the entry recorded during a `lirk test` run", not "a test". Worth
saying so (e.g. `(cached by lirk test)`). The "15 cached results" vs 13
targets mismatch has the same cause. "1 target(s) not cached" doesn't
name the target (here it was `//tests/golden:regen_physics`, which no
`lirk test` scope ever builds).

### 8. `build` and `test` don't share library results

A consequence of the mode-namespaced cache: immediately after a green
`lirk test //...`, `lirk build //...` re-validated 12 of 14 targets
(`lirk: 12 built, 2 cached`), including libraries `lirk test` had just
built from identical inputs. Not a correctness problem, just repeated
work — noticeable on iSH. If intentional, a sentence in "how it works"
would prevent the "why is it rebuilding?" moment.
