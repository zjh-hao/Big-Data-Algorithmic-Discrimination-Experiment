"""Analytic inverse-price checks independent of credit threshold data."""
import unittest

import numpy as np

from research_extension_v3.theory_extension.allocation_policy import _Program
from research_extension_v3.theory_extension.opportunity_price import price_program


class OpportunityPriceTests(unittest.TestCase):
    def solve(self, etas):
        # A1=x, A2=1-x; reference=(.8,.2). The least-harm face is
        # x=.8, W0=.2. A1,A2 >= .2+eta requires minimum harm eta
        # for eta<=.3; eta>.3 is infeasible. This is a closed-form oracle.
        program = _Program([[0., 0.]], [1.], [[1., 1.]], [1.])
        return price_program(program, np.eye(2), [.8, .2], [1., .5],
                             etas=etas, numeric_tol=0.)

    def test_price_target_and_shadow_price_match_analytic_oracle(self):
        result = self.solve([.05, .10, .20])
        self.assertEqual(result["status"], "PASS")
        self.assertAlmostEqual(result["minimum_required_harm"], 0., places=8)
        self.assertAlmostEqual(result["W0"], .2, places=8)
        for row in result["rows"]:
            eta = row["eta"]
            self.assertTrue(row["feasible"])
            self.assertAlmostEqual(row["tau_star"], eta, places=8)
            self.assertAlmostEqual(row["local_tau_slope_wrt_eta"], 1., places=8)
            np.testing.assert_allclose(row["group_opportunity"], [.8-eta, .2+eta], atol=1e-8)
            self.assertTrue(row["target_met"])
            self.assertTrue(row["protection_price_met"])

    def test_prices_monotone_and_convex_on_equal_spaced_targets(self):
        values = np.array([r["tau_star"] for r in self.solve([.05, .10, .15, .20])["rows"]])
        self.assertTrue(np.all(np.diff(values) >= -1e-9))
        self.assertTrue(np.all(np.diff(values, n=2) >= -1e-9))

    def test_target_is_common_floor_not_each_groups_own_increment(self):
        row = self.solve([.1])["rows"][0]
        self.assertLess(row["group_opportunity"][0], .8)
        self.assertAlmostEqual(min(row["group_opportunity"]), .3, places=8)

    def test_extra_price_subtracts_unavoidable_positive_harm(self):
        # A1-A2<=.4 forces x<=.7: d*=.1, W0=.3. Requiring
        # minimum opportunity .35 costs total harm .15, extra harm .05.
        program = _Program([[1., -1.]], [.4], [[1., 1.]], [1.])
        result = price_program(program, np.eye(2), [.8, .2], [1., .5],
                               etas=[.05], numeric_tol=0.)
        self.assertAlmostEqual(result["minimum_required_harm"], .1, places=8)
        self.assertAlmostEqual(result["W0"], .3, places=8)
        row = result["rows"][0]
        self.assertAlmostEqual(row["minimum_harm_at_target"], .15, places=8)
        self.assertAlmostEqual(row["tau_star"], .05, places=8)

    def test_impossible_opportunity_goal_is_retained_as_infeasible(self):
        result = self.solve([.31])
        self.assertEqual(result["status"], "PASS")
        row = result["rows"][0]
        self.assertEqual(row["status"], "INFEASIBLE")
        self.assertFalse(row["feasible"])
        self.assertIsNone(row["tau_star"])
        self.assertIsNone(row["weights"])

    def test_no_arbitrary_negative_or_duplicate_targets(self):
        with self.assertRaises(ValueError):
            self.solve([-.01])
        with self.assertRaises(ValueError):
            self.solve([.01, .01])


if __name__ == "__main__":
    unittest.main()
