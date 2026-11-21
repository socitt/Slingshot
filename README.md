# slingshot

A tiny neural network, trained by neuroevolution, learns to fly a probe
through a chaotic gravity field (a 2–3 body system) to a target —
discovering gravity assists on its own.

Pure Python 3.11+, standard library only, terminal output only. It is
meant to run on iSH (Linux on iOS), so the core never needs numpy.

Built and tested with [lirk](https://github.com/socitt/lirk) from the
first commit. lirk feedback lives in [docs/lirk-notes.md](docs/lirk-notes.md).

## Layout

| Package | What |
|---|---|
| `vecmath/` | `Vec2` plus flat float-list helpers |
| `physics/` | N-body gravity, RK4 + leapfrog, classic initial conditions |
| `chaos/` | logistic map (bifurcations, Feigenbaum delta), Lorenz, Lyapunov exponents |
| `render/` | phone-width plain-text plots |
| `cli/` | `chaos_demo` entry point (`python3 -m cli.chaos_demo lorenz`) |
| `tests/golden/` | golden trajectories and seeded scores (`data/`) and their generator |
| `nn/`, `evolve/`, `env/` | coming in later milestones |

## Building and testing

```sh
pip install -e ../lirk     # wherever lirk is checked out
pip install pytest
lirk test //...
lirk verify --sample 5
```

Plain `python3 -m pytest` from the repo root also works and is kept as
an independent cross-check.

## Milestones

1. ✅ Physics core with energy-conservation and orbit-period tests.
2. ✅ Chaos lab: bifurcation, Lorenz, Lyapunov exponents.
3. MLP + neuroevolution on navigation without gravity.
4. Navigation in a three-body field.
5. Terminal replay of the best agent.
6. Optional numpy backend.

## Conventions

- Units are nondimensional, `G = 1`.
- All randomness flows from a seeded `random.Random`; the root
  `conftest.py` makes the global RNG raise inside tests.
- Golden tests use short horizons with tolerances, or invariants —
  chaos amplifies ulp-level platform differences.
- Regenerate goldens with `python3 -m tests.golden.regen physics`, only
  after an intentional numerical change.
