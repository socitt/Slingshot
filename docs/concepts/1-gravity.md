# 1. Simulating gravity

*Code: [`physics/gravity.py`](../../physics/gravity.py),
[`physics/integrators.py`](../../physics/integrators.py),
[`physics/orbits.py`](../../physics/orbits.py). Tests:
[`physics/test_conservation.py`](../../physics/test_conservation.py),
[`physics/test_orbits.py`](../../physics/test_orbits.py).*

## Newton, in one line

Every body pulls on every other with a force proportional to both
masses and falling off with the square of the distance. For body *i*:

$$\mathbf{a}_i = G \sum_{j \ne i} m_j \frac{\mathbf{r}_j - \mathbf{r}_i}{|\mathbf{r}_j - \mathbf{r}_i|^3}$$

slingshot works in units where $G = 1$, the star has mass 1, and the
inner planet orbits at distance 1. One "year" of the inner planet is
then $2\pi \approx 6.28$ time units, and nothing needs physical constants.

Two bodies that get very close would feel an almost infinite pull, so
the navigation field uses **softening**: $|r|^2$ becomes $|r|^2 +
\varepsilon^2$ with $\varepsilon = 0.03$. Force and potential energy
are softened the same way, so energy is still conserved exactly by the
true dynamics.

## The hard part: stepping forward in time

The equations say how velocity and position *change*. A computer has to
take finite steps of size $h$, and every step makes a small error. Two
integrators are implemented, and the difference between them is the
first lesson of the project.

**RK4 (Runge–Kutta, 4th order)** samples the slope four times per step
and blends them. Its error per unit time shrinks like $h^4$: halve the
step, and the error drops 16×. The test
[`test_rk4_is_fourth_order`](../../physics/test_integrators.py) measures
exactly that ratio (it gets 2⁴ within 0.15 in the exponent).

**Leapfrog (kick–drift–kick)** is cruder, only $h^2$, but it has a
property RK4 lacks: it is *symplectic*. It exactly preserves a slightly
perturbed version of the system's energy, so its energy error
**oscillates but never drifts**. RK4's error is tiny but always grows.

```
  kick:   v ← v + (h/2)·a(x)
  drift:  x ← x + h·v
  kick:   v ← v + (h/2)·a(x)      (a(x) is reused next step: one force call per step)
```

What the tests measured, over 20 orbits of an eccentric (e = 0.5) two-body system:

| Integrator | Max relative energy error | Over time |
|---|---|---|
| Leapfrog, 1000 steps/orbit | < 10⁻³ | last orbit no worse than the first |
| RK4, 1000 steps/orbit | < 10⁻⁷ | grows steadily |

So: RK4 when you need accuracy over a short horizon (the chaos lab, the
golden trajectories), leapfrog when you need stability over a long one
(the moving planets in every navigation episode). Leapfrog is also
**time-reversible**: run it forward 500 steps, flip the velocities, run
500 more, and you're back where you started to 10⁻¹¹.

## Checking it against reality

Kepler's third law predicts an orbit's period from its size alone:
$T = 2\pi\sqrt{a^3 / GM}$. The orbit tests launch two bodies, measure
when the line between them comes back round, and compare with Kepler:
agreement to one part in a million.

The **figure-eight** (Chenciner and Montgomery's three equal masses
chasing each other round a figure-of-eight) is a stricter check: after
its period of 6.3259 time units every body must be back where it
started. It is, to within 10⁻⁶.

## Conserved quantities as a test oracle

You rarely know the exact answer for three bodies, but you always know
what must *not* change: total energy, total momentum and angular
momentum. Those invariants are what the physics tests check, and they
are why the code can be trusted before anything chaotic happens.

**Next:** [2. Chaos](2-chaos.md), or why knowing the laws exactly still
isn't enough.
