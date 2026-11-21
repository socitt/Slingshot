import math

import pytest

from chaos.logistic import (FEIGENBAUM_DELTA, attractor, bifurcation, feigenbaum_estimates,
                            logistic, orbit, period, superstable_sequence)


@pytest.mark.parametrize("r,p", [(2.8, 1), (3.2, 2), (3.5, 4), (3.56, 8), (3.83, 3)])
def test_period_doubling_cascade_and_period3_window(r, p):
    assert period(r) == p


@pytest.mark.parametrize("r", [3.7, 3.9, 4.0])
def test_chaotic_parameters_have_no_short_period(r):
    assert period(r) is None


def test_fixed_point_below_first_bifurcation():
    r = 2.5
    (x,) = attractor(r)
    assert x == pytest.approx(1 - 1 / r, abs=1e-5)


def test_period_two_points_are_exact():
    r = 3.3
    disc = math.sqrt((r + 1) * (r - 3))
    expected = sorted([(r + 1 - disc) / (2 * r), (r + 1 + disc) / (2 * r)])
    assert attractor(r) == pytest.approx(expected, abs=1e-5)


def test_orbit_stays_in_unit_interval():
    xs = orbit(4.0, 0.123, 10_000)
    assert all(0.0 <= x <= 1.0 for x in xs)
    assert logistic(4.0, 0.5) == 1.0


def test_superstable_parameters_match_known_values():
    rs = superstable_sequence(5)
    known = [2.0, 3.2360679775, 3.4985616993, 3.5546408628, 3.5666673799]
    assert rs == pytest.approx(known, abs=1e-9)


def test_feigenbaum_delta():
    est = feigenbaum_estimates(10)
    assert est[-1] == pytest.approx(FEIGENBAUM_DELTA, abs=2e-4)
    # and the estimates converge
    assert abs(est[-1] - FEIGENBAUM_DELTA) < abs(est[0] - FEIGENBAUM_DELTA)


def test_bifurcation_sampling():
    cols = bifurcation(2.8, 4.0, columns=7, keep=5)
    assert [round(r, 10) for r, _ in cols] == [2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
    assert all(len(xs) == 5 for _, xs in cols)
    with pytest.raises(ValueError):
        bifurcation(3, 4, columns=1)
