# 7. The grand tour: multi-body traversal

*Code: [`env/navigation.py`](../../env/navigation.py) (`Tour`, `TOUR`,
`grand_field`). Watch: `python3 -m cli.replay --task tour --target 9`.*

Every earlier mission had one target. The grand tour asks for a
sequence: start deep inside a system of **a star and four planets**,
get past the innermost planet, then catch three more **in order**:

| Body | Orbit radius | Mass | Role |
|---|---|---|---|
| star | centre | 1 | felt everywhere |
| A | 0.9 | 0.03 | hazard: easy to hit, easy to catch by accident |
| B | 1.4 | 0.04 | stop 1 |
| C | 2.0 | 0.05 | stop 2 |
| D | 2.6 | 0.02 | stop 3 |

```mermaid
flowchart LR
    S["start<br/>r = 0.5"] -->|slip past A| B["catch B<br/>r = 1.4"]
    B --> C["catch C<br/>r = 2.0"]
    C --> D["catch D<br/>r = 2.6"]
```

## Why it's much harder than one target

**The target changes mid-flight.** The network has no memory, so it
can't "remember" that it has finished stop 1. Instead the observation
switches: the moment B is caught, the "target offset" inputs start
pointing at C. The same 146 weights must handle every leg.

**Arriving is not enough.** Catching B leaves the probe moving with
whatever velocity it had at B. If that's wrong for the climb to C, the
next leg starts badly. Good tours shape each leg for the next one,
which a reactive network has to discover without being told.

**Four planets, four rhythms.** Their periods are about 5.4, 10.4, 17.8
and 26.3 time units, so the configuration never repeats. Each of the 36
launch delays meets a different arrangement.

**More chaos.** In the three-body field, a coasting probe's outcome was
stable against a 10⁻⁹ nudge for about 20 time units. In the five-body
field over 800 steps (20 time units), a **10⁻¹⁵** nudge changes the
outcome of 5 of 36 launches for a random network: one ends in a crash,
the other leaves the system. That's why tour parity between the numpy
and Python backends is checked on stop counts and medians, not exact
scores ([`env/test_tour.py`](../../env/test_tour.py)).

## Scoring a tour

Each planet caught is worth 1. Finishing the tour adds up to 0.5 for
arriving early; otherwise the current leg adds up to 0.8 for how close
the probe got to its stop, measured from where that leg began. Fuel and
crashes cost as before. A single target is just a one-stop tour under
the same formula, which is why every earlier golden agent re-scores
identically after tours were added.

## Learning in stages

A one-target task is learned more or less all at once. A tour can only
be learned **in order**: nothing about stop 2 can be rewarded until some
genome catches stop 1. So evolution should show *stages*: a long phase
where the population learns the first catch, then a second phase where
a few genomes start reaching the second planet, and so on. The
visual explainer's skill-stages chart records exactly that: every 10
generations, how many of the 64 genomes catch 0, 1, 2 or 3 planets.

## What happened

The golden agent is ES, 64 genomes, 1500 generations (about 40 minutes
with the numpy backend). Doing nothing scores 0.112; it scores **0.764**.
Of its 36 launches, 18 catch nothing, 8 catch B, and **10 catch B and
then C**. Nothing it flies completes the tour.

The population around it tells the more interesting story. Every 10
generations the replay counted how many of all 2,304 flights (64 genomes
× 36 launches) caught at least 1, 2 or 3 planets:

| Generation | ≥ 1 planet | ≥ 2 planets | all 3 |
|---|---|---|---|
| 0 | 1.5% | 0.1% | 0% |
| 150 | 27.0% | 2.7% | 0% |
| 600 | 34.3% | 4.9% | 0% |
| 900 | 38.8% | 7.6% | 0.1% |
| 1200 | 41.6% | 10.1% | 0.6% |
| 1490 | 45.5% | 12.3% | 0.5% |

Three stages, in order:

1. **Catch B.** Fast: past 10% of flights by generation 20, then a long
   slow climb.
2. **Catch B, then C.** Slow and steady: it passes 5% only at generation
   530 and keeps rising to the end.
3. **The full tour.** Complete tours appear by luck from generation 80,
   but rarely: in 14 of the first 75 snapshots, then in 71 of the last
   75. By the end, 12 of the 64 genomes complete the tour on at least one
   launch.

The kept agent never completes a tour because it is chosen for its
*average* over all 36 launches, and the genomes that complete one are
worse on the rest. A second seed (`tour3` in the golden data) shows the
trade-off from the other side: a higher training score (0.821), one
complete grand tour (launch 8: watch it with `python3 -m cli.replay
--task tour --agent tests/golden/data/agents/agent_tour3.json --target 8`),
much weaker generalisation (held-out 0.20 vs 0.43), and less robust
flights (a 10⁻¹² nudge moves one of its scores by 7 × 10⁻⁵). In that
complete tour it catches B at step 151, C at step 177 (they happened to
be close together on that launch) and D at step 378, and planet C hands
it +0.37 energy on the way to D: a gravity assist inside a tour.

## Robust paths in a chaotic field

A random network's tour outcomes change under a 10⁻¹⁵ nudge. The trained
agent's don't: nudged by 10⁻¹², no score moves by more than 3 × 10⁻⁷, no
stop count changes, and no flight comes within 1.4 × 10⁻⁴ of the
capture radius. Evolution selected for flights that are reliable as
well as successful, which is what lets the golden test re-score them.
