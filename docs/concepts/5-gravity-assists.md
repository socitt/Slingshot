# 5. Gravity assists

*Code: [`env/navigation.py`](../../env/navigation.py) (`Result.work`,
`ASSIST`, `RENDEZVOUS`). See it: `python3 -m cli.replay --task assist --target 5`.*

## Free energy from a planet?

Fly past a planet and, *in the planet's own frame*, nothing magic
happens: you fall in, swing round, and leave at exactly the speed you
arrived with. Gravity is conservative, so your speed relative to the
planet is unchanged. Only your **direction** has turned.

But the planet is moving round the star. Switch back to the star's
frame and your velocity is the planet's velocity plus your
planet-relative velocity:

$$\mathbf{v}_\text{out} = \mathbf{V}_\text{planet} + \mathbf{u}_\text{out}, \qquad |\mathbf{u}_\text{out}| = |\mathbf{u}_\text{in}|$$

Turning $\mathbf{u}$ to point more along the planet's motion makes
$|\mathbf{v}_\text{out}|$ bigger, by up to twice the planet's speed. Pass
*behind* the planet (relative to its motion) and you're flung forward;
pass in front and you're braked. The energy is real, and it comes out of
the planet's orbit, by an amount far too small to notice.

```
           planet moving →
                 ●───────▶
        probe in ╲     ╱ probe out, faster
                  ╲   ╱
                   ╲_╱   (passing behind the planet)
```

That's how Voyager toured the outer planets on a budget. Nobody told
slingshot's agent any of this.

## How the project detects one

Every episode keeps an energy ledger. Over each step, the change in the
probe's kinetic energy is split exactly between the work done by each
body's gravity and the work done by the engine:

$$\Delta KE = \sum_\text{bodies} W_j + W_\text{thrust}$$

The split uses the step-average velocity, which makes it exact rather
than approximate (an earlier version used the mid-step velocity and was
off by 7 units on one close pass by the star; the test that caught it
now checks the balance to 10⁻⁹).

Climbing away from the star always costs energy, so the star's entry
is negative. A **planet doing positive work** on a probe that then
reaches its target is a gravity assist.

## What the agents did

The `assist` task is the three-body field with only 0.45 units of fuel
(the climb alone needs about 0.35) and a heavier fuel penalty. In the
golden agent's flight to target 5:

| Source | Energy given to the probe |
|---|---|
| engine | +0.41 |
| outer planet | **+0.38** |
| inner planet | −0.23 |
| star | −1.09 |

The outer planet gave as much energy as the engine. And it isn't a
fluke of one run: across nine independent training runs on that task,
five produced a successful flight where a planet contributed between
+0.36 and +0.49. The assist keeps being rediscovered; the hard part is
everything around it (reaching more than 2–3 of the 6 targets).

## Rendezvous: catching a moving planet

The `rendezvous` mission makes the target the outer planet itself. It
goes round the star once every ~16.5 time units, so each episode
launches at a different point in its orbit (a different `delay` into
the field's history). A fixed point in space can be reached by aiming;
a planet has to be *intercepted*, which is where an agent has to use
the field rather than fight it.

Trained on 12 launch phases, agents caught the planet at 11 of them and
at only 1 of 12 new phases: they had memorised launch windows. Trained
on 36 phases, the golden agent catches it at 29 of 36, and at **10 of 12
phases it never saw** (score 0.804; doing nothing scores 0.087). Try
`python3 -m cli.replay --task rendezvous --target 7`.

**Next:** [6. Testing a chaotic system](6-testing-chaos.md).
