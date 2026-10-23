"""Explicit economic tau budget tests, separate from numerical tolerance."""
import unittest

import numpy as np

from research_extension_v3.protected_policy import apply_protected_policy
from research_extension_v3.theory_extension.allocation_policy import (
    FACE_TOL, _Program, _ranges, _allocate,
    fit_allocation_policies as fit_exact,
)
from research_extension_v3.theory_extension.budget_allocation import (
    _budget_face,_certify_stages,fit_allocation_policies,
)
from research_extension_v3.tests import test_allocation_policy as allocation_fixtures


class BudgetAllocationTests(unittest.TestCase):
    def test_normalized_certificate_rejects_hidden_stage_regression(self):
        stages=[{"level":.5,"remaining":[0,1],"frozen":[0],"coordinate_maxima_on_optimal_face":{"0":.5}},
                {"level":.49999,"remaining":[1],"frozen":[1],"coordinate_maxima_on_optimal_face":{"1":.49999}}]
        certificate=_certify_stages(stages,[.5,.49999])
        self.assertFalse(certificate["normalized_stage_certificate_pass"])
        self.assertGreater(certificate["stage_level_monotonicity_violation"],9e-6)
        stages[1].update(level=.6,coordinate_maxima_on_optimal_face={"1":.6})
        self.assertTrue(_certify_stages(stages,[.5,.6])["normalized_stage_certificate_pass"])

    def test_explicit_budget_opens_nontrivial_allocation_space(self):
        # Two opportunities (x,1-x), fairness forces x<=.6. The profit
        # baseline is (1,0), so smallest necessary worst harm is exactly .4.
        business=_Program([[1.,0.],[-1.,0.]], [.6,-.4], [[1.,1.]], [1.])
        rates=np.eye(2); reference=np.array([1.,0.])
        exact=_budget_face(business,rates,reference,.4,0.)
        lo0,hi0,err=_ranges(exact,rates)
        self.assertIsNone(err)
        self.assertLess(max(hi0-lo0),2e-8)
        budgeted=_budget_face(business,rates,reference,.4,.1)
        lo,hi,err=_ranges(budgeted,rates)
        self.assertIsNone(err)
        np.testing.assert_allclose(hi-lo,[.1+FACE_TOL,.1+FACE_TOL],atol=1e-10)
        sol,stages,_=_allocate(budgeted,rates,lo,hi,np.array([1.,0.]),mode="normalized_leximin")
        self.assertTrue(sol.success,sol.message)
        np.testing.assert_allclose(sol.x,[.55-.5*FACE_TOL,.45+.5*FACE_TOL],atol=1e-9)
        self.assertGreater(len(stages),0)
        self.assertLessEqual(max(reference-sol.x),.4+.1+FACE_TOL+1e-9)
        # The exact-face point is still feasible after increasing tau.
        self.assertLessEqual(np.max(budgeted.aub@np.array([.6,.4])-budgeted.bub),1e-12)

    def test_tau_zero_reproduces_frozen_exact_face_implementation(self):
        frame=allocation_fixtures.AllocationLendingTests().frame()
        cfg={"minimum_approvals":1,"minimum_subgroup_normal":1,"grid":20,
             "primary_risk_budget":.2,"primary_epsilon":.2,"betas":[.1]}
        exact=fit_exact(frame,cfg)
        budget=fit_allocation_policies(frame,{**cfg,"additional_harm_budgets":[0.],"primary_tau":0.})
        self.assertEqual(len(exact["policies"]),len(budget["policies"]))
        for before,after in zip(exact["policies"],budget["policies"]):
            self.assertEqual(before["status"],after["status"])
            np.testing.assert_allclose(apply_protected_policy(frame,before),apply_protected_policy(frame,after),atol=1e-12)
            if after["mode"]!="profit":
                self.assertEqual(after["tau"],0.)
                self.assertEqual(after["range_diagnostics"],before["range_diagnostics"])
                self.assertAlmostEqual(after["total_harm_cap"],after["minimum_required_delta"]+FACE_TOL,places=14)

    def test_harm_budget_logged_separately_and_respected(self):
        frame=allocation_fixtures.AllocationLendingTests().frame()
        cfg={"minimum_approvals":1,"minimum_subgroup_normal":1,"grid":20,
             "primary_risk_budget":.2,"primary_epsilon":.2,"betas":[.1],
             "additional_harm_budgets":[0.,.001,.003]}
        fitted=fit_allocation_policies(frame,cfg)
        self.assertEqual(len(fitted["policies"]),10)
        self.assertEqual(fitted["metadata"]["tau_values"],[0.,.001,.003])
        for p in fitted["policies"][1:]:
            self.assertTrue(p["feasible"],p.get("reason"))
            self.assertEqual(p["tau"],p["additional_harm_budget"])
            self.assertAlmostEqual(p["economic_total_harm_cap"],p["minimum_required_delta"]+p["tau"])
            self.assertAlmostEqual(p["total_harm_cap"]-p["economic_total_harm_cap"],FACE_TOL)
            self.assertLessEqual(p["validation_max_deterioration"],p["total_harm_cap"]+2e-8)
        with self.assertRaises(ValueError):
            fit_allocation_policies(frame,{**cfg,"additional_harm_budgets":[-.01]})


if __name__=="__main__":
    unittest.main()
