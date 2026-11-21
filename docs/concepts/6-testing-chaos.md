# 6. Testing a chaotic system

*Code: [`tests/golden/`](../../tests/golden/), the root
[`conftest.py`](../../conftest.py). Build tool: [lirk](https://github.com/socitt/lirk).*

Page 2 showed that the probe's world amplifies a 10⁻⁹ difference
10⁸-fold in forty time units. Rounding differences between two
computers are about 10⁻¹⁶. So a naive test ("run 2,000 steps and
compare every float to the stored answer") would pass on the machine
that wrote it and fail on an iPhone running iSH, whose x86 emulation
may round differently. slingshot's tests are built around that.

## Five rules

1. **Short horizons for stored trajectories.** Golden trajectories
   (in `tests/golden/data/physics/`) cover only a few time units, before
   chaos has amplified anything, and compare with a tolerance (10⁻⁹,
   loosened to 10⁻⁶ for the near-collision Pythagorean problem).
2. **Invariants instead of trajectories.** Energy, momentum and angular
   momentum conservation; the Lorenz exponents summing to −13.667;
   Kepler's law. These hold on any machine.
3. **Averages instead of orbits.** The logistic map at r = 4 forgets its
   starting point within ~50 steps, so a stored orbit is meaningless.
   Its Lyapunov exponent, an average over 20,000 steps, is stable.
4. **Seed everything.** All randomness comes from a seeded
   `random.Random`. The root `conftest.py` makes the *global* random
   functions raise inside tests, so unseeded randomness fails loudly
   instead of producing a flaky test.
5. **Check that a stored result isn't sitting on a knife-edge.** A
   trained agent's score is re-checked to 10⁻⁶, and two extra tests
   make sure that's safe: every stored flight stays at least 10⁻⁷ away
   from the capture radius (so "reached" can't flip), and nudging the
   start by 10⁻¹² moves no score by more than 10⁻⁶. Random networks fail
   that second test; the trained ones pass.

## Proving the numpy backend agrees

The numpy and pure-Python backends add numbers in a different order,
which differs by ~10⁻¹⁶. In free space the scores agree to 10⁻¹². In the
chaotic field the *typical* genome agrees to 10⁻⁹, but one random genome
that loitered near a planet for 320 steps disagreed by 7 × 10⁻⁴. That's
the physics, not a bug, so the test checks the median tightly and the
worst case loosely, with the reason written down.

## The build tool watches the cache

Every test target runs through lirk, which caches results by content
hash and re-runs only what an edit affects. Caches can lie, so
`lirk verify --sample 5` regularly re-runs cached results to check.
It caught one real stale pass during this project: a test that ran the
`chaos_demo` program in a subprocess, edited without the test re-running
(the full story is in [`docs/lirk-notes.md`](../lirk-notes.md), #9).
