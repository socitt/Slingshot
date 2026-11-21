import math

import pytest

from tests.golden.chaos_cases import load_golden, run_case


def test_lorenz_short_horizon_trajectory():
    golden, fresh = load_golden("lorenz_short"), run_case("lorenz_short")
    assert len(fresh["points"]) == len(golden["points"]) == 11
    for f, g in zip(fresh["points"], golden["points"]):
        assert f == pytest.approx(g, abs=1e-8)


@pytest.fixture(scope="module")
def scores():
    return load_golden("lyapunov_scores"), run_case("lyapunov_scores")


def test_seeded_logistic_score(scores):
    golden, fresh = scores
    # Seeded, but the r=4 orbit is chaotic: after ~50 iterations it is a
    # different (equally valid) orbit on another platform. The *average*
    # is what is robust, so compare to the golden loosely and to ln 2.
    assert fresh["logistic_r4"] == pytest.approx(golden["logistic_r4"], abs=5e-3)
    assert fresh["logistic_r4"] == pytest.approx(math.log(2), abs=1e-2)


def test_lorenz_spectrum_score(scores):
    golden, fresh = scores
    assert fresh["lorenz_spectrum_T100"] == pytest.approx(golden["lorenz_spectrum_T100"], abs=0.05)
    assert sum(fresh["lorenz_spectrum_T100"]) == pytest.approx(-(10 + 1 + 8 / 3), abs=1e-3)


def test_feigenbaum_numbers_are_platform_stable(scores):
    # Newton on a smooth function, not a chaotic orbit: tight tolerance.
    golden, fresh = scores
    assert fresh["superstable_r"] == pytest.approx(golden["superstable_r"], abs=1e-11)
    assert fresh["feigenbaum_estimates"] == pytest.approx(golden["feigenbaum_estimates"], abs=1e-6)
