# lirk notes (slingshot)

slingshot is lirk's 4th real repo, built and tested with lirk from the
first commit. This file is the feedback channel: invocation tally, every
`lirk verify` result, and every bit of friction, with exact commands and
output. Nothing is worked around silently; where a workaround exists in
this repo, the entry says so.

- lirk version: `lirk 1.0.0`, installed with `pip install -e ../lirk`
  (checkout at commit `584a733`), through finding #14. Since
  2026-11-21: `lirk 1.0.1`, lirk `main` at `53fab38` (includes PR #3,
  "Fix the slingshot findings"). See "Upgrade to lirk 1.0.1" below.
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
| M6 numpy backend | 0 | 4 | 1 | 2 | 7 | 0 | 0 |
| Post-M6: generalisation, fixes, #14 probes | 1 | 10 | 1 | 1 | 13 | 0 | 0 |
| lirk 1.0.1 upgrade check | 1 | 4 | 1 | 1 | 7 | 0 | 0 |
| Experiments, timing check | 0 | 2 | 0 | 1 | 3 | 0 | 0 |
| lirk main `8b922ed` (docs rewrite) | 0 | 2 | 0 | 0 | 2 | 0 | 0 |

Running total: **75**, including one run under a numpy-free venv (logged by hand); 1 more in a throwaway scratch repo for #13 is not counted. 14 exited 1: genuine test/build failures, deliberate probes (4 of them the #14 label-error checks, 1 the 1.0.1 seeding-guard probe), and one `lirk verify` I killed with `| head -1` (SIGPIPE; it exited 1 after 5 s and the cache was intact — every target still `cached` afterwards). None were lirk's fault. (Updated when committing; the log is authoritative.)

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
| 2026-11-21 | `lirk verify --sample 5` (seed 2024422273, 64 cached results) | 5 verified, 0 stale, 0 not cached — OK |
| 2026-11-21 | `lirk verify --sample 5` on lirk 1.0.1 (seed 3991276520, 74 cached results) | 5 verified, 0 stale, 0 target(s) not cached — OK |
| 2026-11-21 | `lirk verify --sample 5` on 1.0.1 (seed 329176913, 74 cached results) | 5 verified, 0 stale, 0 target(s) not cached — OK |

## Cache-correctness probes

- **Golden `data` directory invalidation — correct.** Perturbed one
  float in `tests/golden/data/physics_figure8_rk4.json` by 1e-6:
  `lirk test //tests/golden:physics_golden_test` re-ran and FAILED as it
  should. Restored the file: `lirk test //tests/golden/...` reported
  `cached //tests/golden:physics_golden_test`, i.e. it returned to the
  previous success, as documented ("reverting an edit returns you to the
  cached result").
- **Per-suite golden data directories invalidate per suite — correct.**
  After splitting `tests/golden/data/` into `physics/`, `chaos/` and
  `agents/` (each declared as its own `data` directory), adding
  `tests/golden/data/agents/zz_probe.json` re-ran exactly the four
  targets reading agent goldens (`//cli:replay_cli_test` via its dep on
  `//tests/golden:agent_cases`, `//npbackend:batch_test`,
  `//tests/golden:agent_golden_test`, `//tests/golden:regen_test`);
  physics and chaos goldens stayed cached. Removing the file returned
  everything to `cached`. Before the split, regenerating one agent
  re-ran every golden test: correct but coarse, and the fix was in this
  repo's layout, not lirk.
- **Adding `pytest.ini` at the root re-ran exactly the pytest targets**
  and left the unittest target (`//physics:kepler_unittest`) cached.
  Correct.

## Timings (container, not iSH)

| When | Command | Wall time |
|---|---|---|
| after M3 (20 test targets, 147 pytest tests) | `lirk test //... --force` | 13.3 s |
| same tree | `python3 -m pytest -q` from root | 7.9 s |

| lirk 1.0.1, 25 test targets, 187 tests | `lirk test //... --force` | 21.9 s in tests |

The gap is lirk's one-process-per-src model (interpreter + pytest
startup per target), which is by design.

Most of the M4 build invocations were the #12 probe (cross-package and
out-of-repo `data`).

### Timeout headroom for iSH (from 1.0.1's per-target durations)

lirk 1.0.1 prints each target's duration and a "slowest" summary, which
answered "are my `timeout`s safe on iSH?" directly. Heaviest targets vs
their `timeout`: `//chaos:lyapunov_test` 4.9 s / 300 s (61×),
`//npbackend:batch_test` 3.3 s / 120 s (36×, and mostly skipped where
numpy is absent), `//evolve:training_smoke_test` 2.2 s / 180 s (82×),
`//cli:train_test` 2.2 s / 300 s, `//tests/golden:agent_golden_test`
1.3 s / 120 s. iSH emulates x86 in software, so expect a 10–30×
slowdown: every target keeps at least ~36× headroom here, and the full
suite should take roughly 4–11 minutes there. Not yet measured on a
device.

## Friction, bugs and confusing messages

### 1. BUG (high): root `conftest.py` is fingerprinted but never loaded by pytest

*Status: **fixed in lirk 1.0.1** (PR #3), verified 2026-11-21 — see "Upgrade to lirk 1.0.1". The root `pytest.ini` stays (comment updated) so plain pytest behaves the same from any directory.*

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

*Status: fixed in 1.0.1 — now `would fix //env:backend_test: would add :backend`.*

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

*Status: fixed in 1.0.1 per PR #3. Seen: entries labelled `(build)`.
Not yet seen: a `(tests)` entry or a named not-cached target, since the
one 1.0.1 sample had neither.*

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

*Status: fixed in 1.0.1 for libraries. Right after a green `lirk test
//...`, `lirk build //...` printed `25 built, 24 cached`: the 24
libraries were reused and the 25 test targets were re-validated. PR #3
says a built test target never counts as tested; this run shows the
reverse holds too.*

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

### 14. Label errors are clear; a "did you mean" would make them great (minor)

Run from `physics/` (root found via `.lirk-root` every time):

```
$ lirk test //physics
lirk: malformed label '//physics': expected //package:name, //package:all, //package/... or //...
lirk: repo root is /home/user/Slingshot
$ lirk test //physics:orbit
lirk: unknown target: //physics:orbit
lirk: repo root is /home/user/Slingshot
$ lirk test //physics:orbits
lirk: //physics:orbits is type 'library', not 'test'
lirk: repo root is /home/user/Slingshot
```

All exit 1 and all accurate. Two cheap improvements: suggest the
nearest label for an unknown target (`did you mean //physics:orbits?`),
and for a library passed to `lirk test`, name the test targets that
depend on it (`//physics:orbits_test`, `//physics:conservation_test`,
...) — that is almost always what was meant. `//physics:all` and
`//tests/...` worked as documented.

## Upgrade to lirk 1.0.1 (2026-11-21)

`git checkout -B main origin/main` in `../lirk` (at `53fab38`: PR #3
`be4a296` "Fix the slingshot findings", plus PR #4 metadata, 1.0.1),
then `pip install -e ../lirk` → `lirk 1.0.1`.

1. **Full re-run, as announced.** First `lirk test //...` on 1.0.1, tree
   unchanged: `25 PASS, 24 built`, nothing cached, 18.8 s, `lirk: OK`.
   (ACTION_VERSION 11 → 12 invalidates old results on purpose.) Output
   now shows per-target durations, e.g. `PASS //physics:gravity_test (0.3s)`.
2. **The root conftest now loads under lirk, proven without the
   workaround.** I moved `pytest.ini` out of the repo temporarily and
   appended a probe to `vecmath/test_vectors.py`:

   ```python
   def test_probe_unseeded_random():  # TEMPORARY probe
       import random
       random.random()
   ```

   ```
   $ lirk test //vecmath:vectors_test //physics:gravity_test
     PASS   //physics:gravity_test (0.3s)
     FAIL   //vecmath:vectors_test (0.3s): 1 of 1 src files failed: test_vectors
   rootdir: /home/user/Slingshot
   E       AssertionError: use a seeded random.Random, not the global RNG
   FAILED test_vectors.py::test_probe_unseeded_random - AssertionError: use a se...
   ========================= 1 failed, 7 passed in 0.04s ==========================
   lirk: 1/2 tests passed
   lirk: FAILED
   ```

   rootdir is the repo root with no ini file involved. The global-RNG
   guard fired, and `gravity_test`'s `rng` fixture (also from the root
   conftest) resolved. On 1.0.0 both targets would have failed with
   `fixture 'rng' not found`, and the guard would have been silently
   inactive. The probe was reverted (`git diff` clean) and `pytest.ini`
   restored with a comment saying the bug is fixed.
3. **Restoring the edited `pytest.ini`** re-ran exactly the 23 pytest
   targets. The 2 unittest targets and every library stayed cached.
   The next run was fully cached.
4. `lirk verify --sample 5`: 5 verified, 0 stale.

Still open from this repo's list: #9 (subprocess entry-point tests
need a hand-declared dep), #12 (`data` may escape the repo), #13
(module-level skip reported as "contained no tests"), #14 ("did you
mean"). PR #3 says its docs changes cover #2–#4. Re-read on `53fab38`: the
subcommand count, the non-`.py` srcs message and the gitignore lists
(now the same four lines in README, build-format and getting-started)
are fixed. Two leftovers in getting-started.md:
- the `lirk test` row of the command table still says "Run each test
  src via `python3 -m unittest`", with no mention of `runner = "pytest"`;
- "Two behaviours worth knowing up front:" is followed by three bullets.

### 15. `lirk init` trial on a copy of slingshot: silently skips `env/`, and reproduces #9

*Status: open in lirk (1.0.1, `53fab38`). Run on a throwaway copy
(`git archive HEAD`, every `BUILD.lirk` and `.lirk-root` deleted), not on
this repo, and not counted in the tally.*

```
$ lirk init --dry-run
lirk: no .lirk-root found in any parent directory; using the current directory as the repo root: /tmp/claude-0/initcopy
  would create chaos/BUILD.lirk: library :chaos (3 srcs); test :chaos_test (3 srcs, pytest)
  ...                                  # 9 packages listed; env/ is not among them
lirk: dry run -- would draft 9 BUILD.lirk file(s); deps are derived from imports once the files exist
$ lirk init
  ...
  MANUAL //tests/golden:golden: agent_cases.py imports 'env.navigation' -- no target declares env/navigation.py -- declare it in some target's srcs, then run fix again
lirk: drafted 9 BUILD.lirk file(s), 18 target(s); 20 need a hand fix
lirk: next: review the files, declare runtime-read files as `data`, then `lirk test //...`
```

**a. `env/` is skipped by name, with no message (bug).** slingshot's
`env/` holds 4 modules and 3 test files. `lirk/scaffold.py` has
`SKIP_DIRS = {"__pycache__", "node_modules", "venv", "env", "build",
"dist", "site-packages"}`, and the same function already detects real
virtualenvs by their `pyvenv.cfg` (`_is_venv`). So the name list adds
nothing for real venvs and drops real packages called `env`, `build` or
`dist`. All 20 MANUAL lines are a consequence of the missing package.
Suggestions: rely on `pyvenv.cfg` (plus `site-packages` /
`__pycache__`), and print a `skipped env/ (looks like a virtualenv)`
line for anything skipped by name. `env/` is a common package name
(RL environments, settings), which is how slingshot ended up with one.

**b. #9 again, from the other side.** The drafted `//cli:cli_test` has
no `deps`: its tests run `python3 -m cli.<tool>` in a subprocess and
import nothing from `cli`. In the drafted tree `//cli:cli` itself fails
to build (it imports the undeclared `env`), yet:

```
$ lirk test //...
  PASS   //cli:cli_test (4.1s)
$ lirk build //cli:cli
lirk: 0 built, 5 cached, 1 failed, 0 skipped
lirk: FAILED
```

A green test sitting on top of a red library: the entry-point tests go
green with nothing they exercise in their fingerprint. `init` can't
infer the edge from imports, but it could make the same-package
library a dep of the same-package test target by default (the draft
puts every test of a package in one target anyway), which would cover
the documented entry-point pattern.

**c. Golden `data/` directories are not drafted (expected, but
detectable).** `tests/golden/data/{physics,chaos,agents}/` hold only
JSON, are read at runtime, and got no `data` entry. The closing
`next: ... declare runtime-read files as data` line does say so. A
directory containing no `.py` files, next to srcs in the same package,
is a strong hint: listing such directories as "possible `data`" would
turn the generic reminder into a specific one.

**d. Minor.** "(1 srcs)" for singular counts. The draft puts each
package into one library plus one test target, which is a sensible
start but drops per-file invalidation and per-target `timeout`; the
header comment could say "split further as needed".

What worked: the MANUAL messages are specific and say what to do; the
drafted deps for the packages that didn't touch `env/` matched the
hand-written ones (modulo the coarser target split); after `init`,
`lirk test //...` passed 5 of the 9 drafted test targets as-is.

## lirk main `8b922ed`: docs rewrite (PR #5), 2026-11-21

`git merge --ff-only origin/main` (`53fab38..8b922ed`), `pip install -e
../lirk` → still `lirk 1.0.1`. The only code changes are three
comment/docstring paths (`docs/` → `docs/dev/`). slingshot's cache
survived the reinstall: `lirk test //...` → 49 cached, `lirk: OK`.

- **#2 leftovers fixed.** `docs/commands.md`'s `lirk test` section no
  longer names a runner ("Each test file runs in its own process,
  with output passed through exactly as your test runner printed it"),
  and the "Two behaviours" list with three bullets is gone.
- **#1 is documented** in `docs/adopting.md`: "lirk runs pytest with
  your repo root as its root directory, so a repo-level `conftest.py`
  is always loaded."
- **#9 got worse in the docs (regression).** The new "Testing your
  entry point" section in `docs/adopting.md` shows only the test file
  (`subprocess.run([sys.executable, "-m", "cli.main", "--help"], ...)`).
  The old version at least showed a BUILD block with
  `deps = [":render"]`. The new one shows no BUILD block and never says
  the test needs a dep on the entry point's target. Followed exactly,
  it produces #9's stale cached PASS: the test imports nothing from
  `cli`, so neither the import check nor `lirk fix` / `lirk init` adds
  the edge, and editing `cli/main.py` leaves the test `cached`.
  Suggested text: "This test imports nothing from your app, so lirk
  can't see the dependency: add the entry point's target to `deps`
  yourself (`deps = [":main"]`). `lirk fix` won't add it."
- **#13 not covered.** `docs/troubleshooting.md` explains
  `contained no tests` only as "That test file defines no tests",
  which is wrong for a module-level `pytest.importorskip` (the file has
  tests; it skipped itself). Until lirk tells the two apart, a line
  recommending per-test skips for optional dependencies would help.
- #12, #14 and #15 are code issues, unchanged by a docs PR.
