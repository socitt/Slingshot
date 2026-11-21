# 2. Chaos

*Code: [`chaos/logistic.py`](../../chaos/logistic.py),
[`chaos/lorenz.py`](../../chaos/lorenz.py),
[`chaos/lyapunov.py`](../../chaos/lyapunov.py). Demo:
`python3 -m cli.chaos_demo bifurcation | lorenz | lyapunov`.*

A system is **chaotic** when it is perfectly deterministic and still
unpredictable: tiny differences in where you start grow exponentially.
Not randomness; amplification.

## The probe's own field is chaotic

Two probes coast from the same orbit, started one billionth of a unit
apart (`env/test_trio.py::test_probe_motion_is_chaotic`):

| Time | Distance between them |
|---|---|
| 0 | 10⁻⁹ |
| 10 | 1.4 × 10⁻⁸ |
| 20 | 2.7 × 10⁻⁸ |
| 30 | 1.2 × 10⁻⁶ |
| 40 | 0.27 |

For twenty time units they're indistinguishable. Then close passes by
the planets tear them apart, about 10⁸× in forty time units. Any
navigation policy is flying through this.

## The logistic map: chaos in one line

$$x_{n+1} = r\,x_n(1 - x_n)$$

A population model, and the simplest place to watch chaos appear.
Turn up $r$:

| r | Long-run behaviour |
|---|---|
| 2.8 | settles to one value |
| 3.2 | alternates between 2 values |
| 3.5 | cycles through 4 |
| 3.56 | 8 |
| 3.7 | never repeats: chaos |
| 3.83 | a window of order: period 3 |
| 4.0 | fully chaotic |

Each doubling arrives faster than the last, and the ratio of successive
gaps converges to **Feigenbaum's constant δ = 4.6692…**, which turns up
in every system that doubles its way into chaos, from dripping taps to
electronic circuits. slingshot finds it by solving for the *superstable*
parameters $R_n$ (where $x = \tfrac12$ is on the cycle) with Newton's
method, and its estimate is **4.66919** (`chaos/test_logistic.py`).

```
python3 -m cli.chaos_demo bifurcation --width 44
+ logistic map x->rx(1-x) -----------------+
|                                  ::+==-==|
|                          ---=--==== ---::|
|                   +****=====------- :::::|
|           =*******=   -====---:---: ::-::|
|@@@@@@@*                    :::---:: :::.:|
...
```

## Lyapunov exponents: putting a number on it

If two nearby states separate like $\delta(t) \approx \delta_0
e^{\lambda t}$, then $\lambda$ is the **Lyapunov exponent**. Positive
means chaos; the bigger it is, the sooner prediction fails. Roughly,
you lose one decimal digit of precision every $\ln 10 / \lambda$ time units.

slingshot measures it three ways (`chaos/lyapunov.py`):

1. **For a 1-D map**, average the log of the local stretching:
   $\lambda = \langle \ln |f'(x)| \rangle$. For the logistic map at
   $r = 4$ the exact answer is $\ln 2 = 0.69315$; the project gets
   0.69315 from 200,000 iterations (0.6932 from the 20,000 the golden
   test uses).
2. **For a flow with a known Jacobian** (Lorenz), evolve a set of tiny
   perturbation vectors alongside the trajectory and re-orthogonalise
   them every few steps (Benettin's method). This gives the whole
   *spectrum*: for Lorenz, **+0.906, 0, −14.57**. Their sum must equal
   the rate at which the system shrinks volume, $-(\sigma + 1 + \beta)
   = -13.667$, and it does to 10⁻³. That sum is an invariant the tests
   rely on.
3. **For anything you can step**, follow two copies, measure their
   separation, rescale it back to tiny every few steps, and average the
   log growth. No Jacobian needed, so it works on the N-body problem
   directly: a probe near the binary star gives $\lambda \approx 0.6$,
   a distant one $\approx 0$.

One trap worth knowing, because the project fell into it: perturbing
*every* velocity by the same amount is a uniform boost, not a
perturbation of the dynamics. The two copies just drift apart in a
straight line, and you measure the same "exponent" for any system.
There's a regression test for that mistake
(`test_uniform_perturbation_is_a_boost_not_chaos`).

## The Lorenz attractor

Three equations from a 1963 weather model:

$$\dot x = \sigma(y - x), \quad \dot y = x(\rho - z) - y, \quad \dot z = xy - \beta z$$

with $\sigma = 10, \rho = 28, \beta = 8/3$. Trajectories never repeat
and never escape, tracing the famous butterfly. It's the system that
gave chaos its "butterfly effect" name, and it's integrated here with
the same RK4 as the planets.

**Next:** [3. A neural network from scratch](3-neural-nets.md).
