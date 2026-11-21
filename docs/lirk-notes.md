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
| M2 chaos lab | 0 | 6 | 2 | 2 | 10 | 0 | 1 (see #9 — caught by `lirk verify`) |
| M3 MLP + neuroevolution | 0 | 4 | 1 | 1 | 6 | 0 | 0 |
| M4 three-body field + M5 replay | 5 | 3 | 3 | 1 | 12 | 0 | 0 |
| M6 numpy backend | 0 | 4 | 1 | 3 | 8 | 0 | 0 |

Running total: **51**, including one run under a numpy-free venv (logged by hand); 1 more in a throwaway scratch repo for #13 is not counted. 9 exited 1: genuine test/build failures, deliberate probes, and one `lirk verify` I killed with `| head -1` (SIGPIPE; it exited 1 after 5 s and the cache was intact — every target still `cached` afterwards). None were lirk's fault. (Updated when committing; the log is authoritative.)

## `lirk verify` results

| Date | Command | Result |
|---|---|---|
| 2026-11-21 | `lirk verify --sample 5` (seed 1720564241) | 5 verified, 0 stale, 1 target(s) not cached — OK |
| 2026-11-21 | `lirk verify //cli/...` (deliberate probe, #9) | 0 verified, **1 stale** — FAILED (expected; the probe's point) |
| 2026-11-21 | `lirk verify --sample 5` (seed 1905431998) | 5 verified, 0 stale, 0 not cached — OK |
| 2026-11-21 | `lirk verify --sample 5` (seed not captured — I piped to `tail`) | 5 verified, 0 stale, 0 not cached — OK |
| 2026-11-21 | `lirk verify --sample 5` (seed 745970699, 61 cached results) | 5 verified, 0 stale, 0 not cached — OK |
| 2026-11-21 | `lirk verify --sample 5` (seed 1218428339) | interrupted by my own `\| head -1` (SIGPIPE) — no result |
| 2026-11-21 | `lirk verify --sample 5` (seed 3760781111, 62 cached results) | 5 verified, 0 stale, 0 target(s) not cached — OK |

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

## Timings (container, not iSH)

| When | Command | Wall time |
|---|---|---|
| after M3 (20 test targets, 147 pytest tests) | `lirk test //... --force` | 13.3 s |
| same tree | `python3 -m pytest -q` from root | 7.9 s |

The gap is lirk's one-process-per-src model (interpreter + pytest
startup per target), which is by design.

Most of the M4 build invocations were the #12 probe (cross-package and
out-of-repo `data`).

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

### 9. BUG/DOCS (high): the recommended entry-point test pattern caches a stale PASS unless the dep is hand-declared

*Status: open in lirk. In this repo the dep is declared by hand, with a comment in `cli/BUILD.lirk`.*

getting-started.md "Covering your entry point" recommends a test that
runs the program with `subprocess.run([sys.executable, "-m", "cli.render", ...])`.
That test never *imports* the entry-point module, so lirk's import
check sees no edge, `lirk build` is happy without one, and `lirk fix`
adds nothing (it did add every other dep in the repo correctly):

```
$ lirk fix --dry-run
  ...                      # 12 targets listed; //cli:chaos_demo_test is not among them
lirk: would add 23 dep(s) to 12 target(s), 0 need a hand fix
```

With no dep declared, editing the entry point leaves the test cached:

```
$ sed -i 's/print("chaos-demo: ok")/print("chaos-demo: BROKEN"); return 1/' cli/chaos_demo.py
$ lirk test //cli:chaos_demo_test
  cached  //cli:chaos_demo_test
lirk: 1/1 tests passed
lirk: OK
$ lirk verify //cli/...
  STALE  //cli:chaos_demo_test (test): cached as passing, fails when re-run: 1 of 1 src files failed: test_chaos_demo
...
lirk: an input these targets depend on is not in their fingerprint -- often a file read at runtime but not declared in `data`
lirk: 0 verified, 1 stale, 1 target(s) not cached
lirk: FAILED
```

That's a stale cached PASS on exactly the target the docs say to write
first. `lirk verify` did its job. The docs' example does show
`deps = [":render"]`, but never says that this dep is the one the import
check *can't* enforce and that `lirk fix` won't add, so it reads as
boilerplate. Suggestions: say so in the docs; and/or have the import
check also scan test srcs for string literals passed after `"-m"` (a
cheap heuristic that would catch the documented pattern exactly).

Also, the `verify` hint "often a file read at runtime but not declared
in `data`" points the wrong way here: the missing input is a Python
module run in a subprocess. Mentioning that case in the hint would save
a detour.

After declaring `deps = [":chaos_demo"]`, the same edit makes
`lirk test //cli/...` FAIL as it should, and reverting returns `OK`.

### 10. `lirk fix` relies on transitive deps for a direct import (fine, worth knowing)

`cli/chaos_demo.py` imports `physics.integrators` directly; `lirk fix`
didn't add `//physics:integrators` because `//chaos:lorenz` already
brings it in transitively. That matches the documented rule (the
closure counts), but it means a later refactor of `chaos` that drops
the dependency turns `cli:chaos_demo` into a build failure in a package
nobody touched. Not a bug; a note for anyone expecting Bazel's
"strict deps" behaviour.

### 11. Confirmed working as documented in M3 (no friction)

- `lirk fix` added 27 deps across 15 targets in one go, including
  cross-package edges into four new packages; zero needed a hand fix.
- Editing `evolve/strategies.py` re-ran exactly the dependent tests
  (strategies, training smoke, train CLI, agent golden, regen) and left
  physics/chaos cached.
- A real bug (GA default elite/parents larger than a small population)
  was caught by the subprocess CLI test, not by unit tests — the
  entry-point pattern from getting-started paid for itself, once its
  dep was declared (#9).

### 12. `data` paths may leave the package — and the repo — silently

*Status: open question for lirk; this repo doesn't rely on it.*

build-format.md says `data` entries are "paths relative to the package
directory" and says nothing about `..`. In practice:

```
# cli/BUILD.lirk, //cli:replay_cli_test
data = ["../tests/golden/data"]       # another package's directory
$ lirk build //cli:replay_cli_test    ->  cached  //cli:replay_cli_test
$ echo '{}' > tests/golden/data/zz_probe.json
$ lirk build //cli:replay_cli_test    ->  built  //cli:replay_cli_test   (fingerprinted: good)

data = ["../../lirk/docs"]            # outside the repo entirely
$ lirk build //cli:replay_cli_test
  built  //cli:replay_cli_test
lirk: 1 built, 12 cached, 0 failed, 0 skipped
lirk: OK
```

Cross-package data works and is fingerprinted. But a path escaping the
repo root is accepted without a word, while `.lirk-root`'s `ignore`
explicitly rejects absolute and `..` paths "rather than quietly
clamped". One of the two behaviours is probably unintended; either way
it's worth documenting. (An out-of-repo input also makes the cache
depend on files a fresh clone won't have.)

What this repo does instead: `//cli:replay_cli_test` depends on
`//tests/golden:agent_cases`, the target that owns `tests/golden/data`
as its `data` — Bazel's filegroup idea. Worth a line in the docs as
*the* way to share a data directory between packages.

### 13. A pytest module that skips itself fails under lirk ("contained no tests")

*Status: open in lirk. Avoided in this repo by skipping per test, not per module (explicit, commented in `npbackend/test_batch.py`).*

The standard way to make a test module optional on a dependency is a
module-level `pytest.importorskip`. In a throwaway repo:

```python
# pkg/test_modskip.py
import pytest
np = pytest.importorskip("numpy_not_installed_xyz")
def test_x():
    assert np
```

```
$ lirk test //...
  FAIL   //pkg:modskip: 1 of 1 src files failed: test_modskip -- contained no tests: test_modskip
...
collected 0 items / 1 skipped
============================== 1 skipped in 0.01s ==============================
  PASS   //pkg:partskip
lirk: 1/2 tests passed
lirk: FAILED
```

pytest itself exits 5 here (`python3 -m pytest test_modskip.py` → rc 5),
and lirk maps exit 5 to "contained no tests". But the module *did* say
something: it was skipped on purpose, and pytest printed `1 skipped`.
For slingshot this matters exactly on iSH, where numpy may be
unavailable: the optional-backend tests would turn `lirk test //...`
red. A file with *no* tests and a file that *skipped itself* are
different situations; lirk could tell them apart from pytest's summary
line (or the docs could recommend per-test skips, as done here).

Two related things that work well:
- A target whose tests all skip at function level passes, and is
  cached. Installing numpy later changes lirk's Python-environment
  fingerprint, so the target re-runs and the skipped tests actually
  execute — no stale PASS from a skip. Confirmed: switching between a
  numpy-free venv and the system interpreter re-ran all 25 targets each
  way.
- In the numpy-free venv (`python3 -m venv`, `pip install -e ../lirk
  pytest`), `lirk test //...` passed 25/25 with the numpy tests
  reported as skipped. That is the closest this container gets to iSH.
