"""Lorenz system basics, as plain unittest."""

import math
import unittest

from chaos.lorenz import BETA, RHO, divergence, lorenz, lorenz_jacobian, trajectory


class LorenzTest(unittest.TestCase):
    def test_nontrivial_fixed_points_are_stationary(self):
        c = math.sqrt(BETA * (RHO - 1))
        f = lorenz()
        for s in (1, -1):
            for v in f(0.0, [s * c, s * c, RHO - 1]):
                self.assertAlmostEqual(v, 0.0, places=12)

    def test_jacobian_matches_finite_differences(self):
        f, jac = lorenz(), lorenz_jacobian()
        y, h = [1.3, -2.1, 20.0], 1e-6
        J = jac(y)
        for k in range(3):
            up, dn = list(y), list(y)
            up[k] += h
            dn[k] -= h
            col = [(a - b) / (2 * h) for a, b in zip(f(0, up), f(0, dn))]
            for i in range(3):
                self.assertAlmostEqual(col[i], J[i][k], places=5)

    def test_trace_equals_divergence_everywhere(self):
        jac = lorenz_jacobian()
        for y in ([0, 0, 0], [5, -3, 30], [-10, 2, 1]):
            J = jac(y)
            self.assertAlmostEqual(J[0][0] + J[1][1] + J[2][2], divergence())

    def test_trajectory_stays_on_the_attractor(self):
        traj = trajectory([1.0, 1.0, 1.0], 0.01, 5000, every=50)
        self.assertEqual(len(traj), 101)
        for x, y, z in traj[10:]:
            self.assertLess(abs(x), 25)
            self.assertLess(abs(y), 30)
            self.assertTrue(0 < z < 55)
        # both wings get visited
        self.assertTrue(any(p[0] > 5 for p in traj) and any(p[0] < -5 for p in traj))
