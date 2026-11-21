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
| `render/` | phone-width plain-text plots and episode replay (ANSI, no curses) |
| `cli/` | `chaos_demo`, `train` and `replay` entry points |
| `tests/golden/` | golden trajectories and seeded scores (`data/`) and their generator |
| `nn/` | tiny MLP from scratch; a flat list of floats is the genome |
| `evolve/` | genetic algorithm and OpenAI-style evolution strategy (ask/tell, seeded) |
| `env/` | navigation task (thrust-limited probe, optional N-body field), scoring, training loop |

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
3. ✅ MLP + neuroevolution on navigation without gravity. The golden
   agent (ES, seed 1, 100 generations × 32) scores 1.080 on the 12
   training targets and 1.077 on 8 held-out ones, beating a hand-tuned
   PD controller (1.047). It is the best of four seeds tried; the
   others reached 0.90–0.99 (held-out 0.68–0.89).
4. ✅ Navigation in a three-body field (star + two planets, chaotic:
   coasting probes 1e-9 apart diverge ~1e8 in 40 time units).
   - `trio` (fuel 1.0): the golden agent reaches 5/6 training targets
     (score 0.810 vs 0.191 for zero thrust; PD controllers crash or
     run dry). It learned routes, not a general controller: held-out
     targets 1/6.
   - `assist` (fuel 0.45, fuel penalty 0.5): reaches 3/6 (0.306). In
     one of them the outer planet supplies +0.38 energy, as much as the
     engine — a gravity assist nobody programmed. GA runs collapse to
     "don't thrust" (exactly 0.191); ES escapes it. Seeded per-body
     work bookkeeping (`Result.work`) is how assists are detected.
5. ✅ Terminal replay: `python3 -m cli.replay` (golden assist agent),
   `--task free|trio|assist --target N`, `--plain` for a still.
6. Optional numpy backend.

## Training

```sh
python3 -m cli.train --task free --strategy es --generations 100 --pop 32 --seed 1 --out agent.json
```

## Watching

```sh
python3 -m cli.replay --task assist --target 5   # the gravity-assist episode
python3 -m cli.chaos_demo bifurcation
```

## Open problems

- Field agents overfit their six targets. More targets, a time/phase
  input, or a richer net would likely help; each costs training time
  that matters on iSH.
- A better escape from the do-nothing optimum on `assist` (novelty
  search, or shaping on energy gained) is the obvious next experiment.

## Conventions

- Units are nondimensional, `G = 1`.
- All randomness flows from a seeded `random.Random`; the root
  `conftest.py` makes the global RNG raise inside tests.
- Golden tests use short horizons with tolerances, or invariants —
  chaos amplifies ulp-level platform differences.
- Regenerate goldens with `python3 -m tests.golden.regen physics`, only
  after an intentional numerical change.
