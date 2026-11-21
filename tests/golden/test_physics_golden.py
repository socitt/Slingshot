"""Compare fresh short-horizon trajectories against checked-in goldens."""

import pytest

from tests.golden.physics_cases import CASES, load_golden, run_case

# Absolute tolerance on positions/velocities. Generous relative to
# double precision so platform ulp differences can't trip it, tight
# enough to catch any real change to the integrators or forces.
TOL = {"figure8_rk4": 1e-9, "figure8_leapfrog": 1e-9,
       "kepler_e06_rk4": 1e-9, "pythagorean_rk4": 1e-6}


@pytest.mark.parametrize("name", sorted(CASES))
def test_trajectory_matches_golden(name):
    golden, fresh = load_golden(name), run_case(name)
    assert fresh["masses"] == golden["masses"]
    assert (fresh["h"], fresh["steps"], fresh["method"]) == (
        golden["h"], golden["steps"], golden["method"])
    assert fresh["energy0"] == pytest.approx(golden["energy0"], rel=1e-12)
    assert len(fresh["frames"]) == len(golden["frames"])
    for f, g in zip(fresh["frames"], golden["frames"]):
        assert f["t"] == pytest.approx(g["t"], abs=1e-12)
        assert f["pos"] == pytest.approx(g["pos"], abs=TOL[name])
        assert f["vel"] == pytest.approx(g["vel"], abs=TOL[name])


def test_figure_eight_energy_invariant():
    # Published value for the Chenciner-Montgomery orbit: E ~ -1.2871
    assert load_golden("figure8_rk4")["energy0"] == pytest.approx(-1.2871, abs=1e-4)


def test_every_case_has_a_golden_file():
    for name in CASES:
        assert load_golden(name)["case"] == name
