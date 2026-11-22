# slingshot

A tiny neural network, trained by neuroevolution, learns to fly a probe
through a chaotic gravity field (a 2–3 body system) to a target —
discovering gravity assists on its own.

Pure Python 3.11+, standard library only, terminal output only. It is
meant to run on iSH (Linux on iOS), so the core never needs numpy.

Built and tested with [lirk](https://github.com/socitt/lirk) from the
first commit. lirk feedback lives in [docs/lirk-notes.md](docs/lirk-notes.md).

## Learn the ideas

[`docs/concepts/`](docs/concepts/README.md) explains everything this
project touches, with its own numbers: simulating gravity, chaos and
Lyapunov exponents, a neural net from scratch, learning by evolution,
gravity assists, and how to test a chaotic system.

## Layout

| Package | What |
|---|---|
| `vecmath/` | `Vec2` plus flat float-list helpers |
| `physics/` | N-body gravity, RK4 + leapfrog, classic initial conditions |
| `chaos/` | logistic map (bifurcations, Feigenbaum delta), Lorenz, Lyapunov exponents |
| `render/` | phone-width plain-text plots, episode replay (ANSI, no curses), and the data export for the visual explainer |
| `cli/` | `chaos_demo`, `train`, `replay` and `export_viz` entry points |
| `tests/golden/` | golden trajectories and seeded scores (`data/`) and their generator |
| `nn/` | tiny MLP from scratch; a flat list of floats is the genome |
| `evolve/` | genetic algorithm and OpenAI-style evolution strategy (ask/tell, seeded) |
| `npbackend/` | optional numpy batch evaluator (never imported by the core) |
| `env/` | navigation task (thrust-limited probe, optional N-body field; fixed targets, moving rendezvous, multi-stop tours), scoring, training loop |

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
   - `trio` (fuel 1.0, 24 targets on a ring beyond the outer planet):
     the golden agent reaches 19/24 (score 0.829 vs 0.191 for zero
     thrust; PD controllers crash or run dry) and 7/12 seeded random
     held-out targets (0.753). Trained on 6 targets, the same method
     memorised routes and reached 1/6 held-out; 12 targets gave ~3/6.
   - `assist` (fuel 0.45, fuel penalty 0.5): reaches 3/6 (0.306). In
     one of them the outer planet supplies +0.38 energy, as much as the
     engine — a gravity assist nobody programmed. GA runs collapse to
     "don't thrust" (exactly 0.191); ES escapes it. Seeded per-body
     work bookkeeping (`Result.work`) is how assists are detected.
5. ✅ Terminal replay: `python3 -m cli.replay` (golden assist agent),
   `--task free|trio|assist --target N`, `--plain` for a still.
6. ✅ Optional numpy backend: `--backend numpy` (or
   `SLINGSHOT_BACKEND=numpy`) scores a whole population × targets in
   lockstep — 12–21× faster here, matching the Python backend to 2e-16
   (free space) / 4e-9 (chaotic field). Nothing in the core imports
   numpy; without it, `--backend numpy` exits with a clear message and
   the numpy tests skip individually. N-body gravity itself stays pure
   Python: with three bodies numpy's per-call overhead loses.
7. ✅ Rendezvous: catch the moving outer planet. Launching at 36
   phases of its orbit, the golden agent intercepts it at most of them
   and at 10 of 12 never-seen phases (with 12 training phases an agent
   memorised launch windows: 1/12 held-out).
8. ✅ The grand tour: a star and four planets; slip past planet A, then
   catch B, C and D in order. A memoryless net has to handle every leg,
   because the target switches the moment a planet is caught. After
   1500 generations the golden agent catches B on 18 of 36 launches and
   B then C on 10 (score 0.764 vs 0.112 for doing nothing). A second
   seed flew the first complete tour, all three planets on one launch.
   Skills appear in stages; the visual explainer charts them.

## Training

```sh
python3 -m cli.train --task free --strategy es --generations 100 --pop 32 --seed 1 --out agent.json
```

## Watching

The visual explainer (replays, the swarm evolving, chaos plots) is
built from `python3 -m cli.export_viz --out data.json`, which re-runs
the golden agents' own training to sample it (a few minutes).

```sh
python3 -m cli.replay --task assist --target 5   # the gravity-assist episode
python3 -m cli.chaos_demo bifurcation
```

## Open problems

- Field agents still miss ~40% of held-out targets.
- `assist` scores stay around 0.3: agents reach 2–3 of 6 targets and
  always spend the whole budget.

What has been tried (numpy backend, ES unless noted), so the next
attempt starts from evidence:

| Idea | Result |
|---|---|
| 6 → 12 → 24 `trio` training targets | held-out reach 1/6 → ~3/6 → 7/12 (adopted) |
| GA instead of ES on `assist` | collapses to "don't thrust" (0.191) |
| bigger ES (64 × 500) on `assist` | 0.13–0.33, no better than 32 × 100 |
| larger ES sigma (0.25) | worse (≈0.05) |
| curriculum: trio agent, fuel annealed 1.0 → 0.45 | worse (≈0.02–0.06) |
| nearest planet's relative velocity as 2 more inputs | no gain (assist 0.27–0.30, trio 0.787 vs 0.829); reverted |
| training-only shaping: credit a miss for the highest orbit reached | worse (0.14–0.20, ≤1/6): it learned to climb in any direction |
| 12 → 24 → 36 `rendezvous` launch phases | held-out catches 1/12 → 6/12 → 10/12 (adopted) |

The gravity assist itself is robust, though: independent `assist` runs
keep rediscovering it. Outer-planet energy of +0.36 to +0.49 appeared in
a successful episode in 5 of the 9 `assist` runs above, plus the golden agent's
+0.38. Untried: novelty search, shaping on *direction-aware* energy (toward
the target), a recurrent or larger net.

## Conventions

- Units are nondimensional, `G = 1`.
- All randomness flows from a seeded `random.Random`; the root
  `conftest.py` makes the global RNG raise inside tests.
- Golden tests use short horizons with tolerances, or invariants —
  chaos amplifies ulp-level platform differences.
- Regenerate goldens with `python3 -m tests.golden.regen physics`, only
  after an intentional numerical change.
