"""Kepler-law sanity checks, deliberately plain unittest (runner default)."""

import math
import unittest

from physics.orbits import kepler_period, two_body


class KeplerPeriodTest(unittest.TestCase):
    def test_unit_orbit(self):
        self.assertAlmostEqual(kepler_period(1.0, 1.0), 2 * math.pi)

    def test_third_law_scaling(self):
        # T^2 / a^3 is constant for a fixed central mass
        for a in (0.5, 1.0, 3.0):
            self.assertAlmostEqual(kepler_period(a, 2.0) ** 2 / a ** 3,
                                   4 * math.pi ** 2 / 2.0)


class TwoBodySetupTest(unittest.TestCase):
    def test_centre_of_mass_frame(self):
        system, pos, vel = two_body(2.0, 0.5, 1.0, 0.3)
        cx, cy = system.center_of_mass(pos)
        px, py = system.momentum(vel)
        for value in (cx, cy, px, py):
            self.assertAlmostEqual(value, 0.0, places=14)

    def test_bound_orbit_has_negative_energy(self):
        system, pos, vel = two_body(1.0, 1.0, 1.0, 0.9)
        self.assertLess(system.energy(pos, vel), 0.0)

    def test_energy_matches_semi_major_axis(self):
        # E = -G m1 m2 / (2a) for any eccentricity
        for e in (0.0, 0.4, 0.8):
            system, pos, vel = two_body(1.0, 0.25, 2.0, e)
            self.assertAlmostEqual(system.energy(pos, vel), -0.25 / 4.0, places=12)

    def test_rejects_unbound_eccentricity(self):
        with self.assertRaises(ValueError):
            two_body(1.0, 1.0, 1.0, 1.0)
