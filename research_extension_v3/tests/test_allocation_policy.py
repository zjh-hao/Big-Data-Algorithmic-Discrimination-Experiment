"""Geometry/semantic tests independent of the Home Credit experiment values."""
import unittest

import numpy as np
import pandas as pd

from research_extension_v3.protected_policy import apply_protected_policy
from research_extension_v3.theory_extension.allocation_policy import (
    _Program, _allocate, _ranges, fit_allocation_policies,
)


class AllocationGeometryTests(unittest.TestCase):
    def program(self):
        # x+y <= 1, while z has its own resource. Exact leximin is (.5,.5,1).
        return _Program([[1.,1.,0.]], [1.], [[0.,0.,0.]], [0.])

    def test_progressive_filling_beats_one_stage_on_second_resource(self):
        p=self.program(); rates=np.eye(3)
        lo,hi,err=_ranges(p,rates)
        self.assertIsNone(err)
        solution,stages,fixed=_allocate(p,rates,lo,hi,np.array([1.,0.,-1.]),mode="normalized_leximin")
        self.assertTrue(solution.success,solution.message)
        np.testing.assert_allclose(solution.x,[.5,.5,1.],atol=1e-8)
        self.assertEqual([set(s["frozen"]) for s in stages],[{0,1},{2}])
        self.assertEqual(len(fixed),3)
        p2=self.program()
        one,_,_=_allocate(p2,rates,lo,hi,np.array([1.,0.,-1.]),mode="normalized_maxmin")
        np.testing.assert_allclose(one.x,[.5,.5,.5],atol=1e-8)
        # The full solution Pareto dominates the one-stage profit tiebreak in
        # the three opportunity coordinates, verifying why stage 2 matters.
        self.assertTrue(np.all(solution.x>=one.x-1e-8))
        self.assertGreater(solution.x[2],one.x[2]+.4)

    def test_positive_affine_change_preserves_normalized_solution(self):
        # Add a fixed fourth variable to encode nonzero affine offsets.
        p=_Program([[1.,1.,0.,0.]], [1.], [[0.,0.,0.,1.]], [1.])
        original=np.c_[np.eye(3),np.zeros(3)]
        scales=np.array([.2,3.,5.]); shifts=np.array([.1,.2,-.4])
        changed=original*scales[:,None]
        changed[:,3]=shifts
        lo,hi,err=_ranges(p,changed)
        self.assertIsNone(err)
        solution,_,_=_allocate(p,changed,lo,hi,np.array([1.,0.,-1.,0.]),mode="normalized_leximin")
        self.assertTrue(solution.success,solution.message)
        np.testing.assert_allclose(solution.x,[.5,.5,1.,1.],atol=1e-8)

    def test_constant_coordinate_is_reported_but_not_bottleneck(self):
        p=_Program([[0.,0.]], [0.], [[1.,1.]], [1.])
        rates=np.array([[1.,0.],[0.,1.],[.2,.2]])
        lo,hi,err=_ranges(p,rates)
        self.assertIsNone(err)
        np.testing.assert_allclose(hi-lo,[1.,1.,0.],atol=1e-12)
        full,stages,fixed=_allocate(p,rates,lo,hi,np.array([1.,0.]),mode="normalized_leximin")
        self.assertTrue(full.success,full.message)
        np.testing.assert_allclose(full.x,[.5,.5],atol=1e-8)
        self.assertNotIn("2",fixed)
        absolute,_,_=_allocate(p,rates,lo,hi,np.array([1.,0.]),mode="absolute_maxmin")
        np.testing.assert_allclose(absolute.x,[.8,.2],atol=1e-8)

    def test_all_constant_uses_profit_tiebreak(self):
        p=_Program([[0.,0.]], [0.], [[1.,1.]], [1.])
        rates=np.array([[.3,.3]])
        sol,stages,_=_allocate(p,rates,np.array([.3]),np.array([.3]),np.array([0.,1.]),mode="normalized_leximin")
        self.assertTrue(sol.success,sol.message)
        self.assertEqual(stages,[])
        np.testing.assert_allclose(sol.x,[0.,1.],atol=1e-8)


class AllocationLendingTests(unittest.TestCase):
    def frame(self):
        rows=[]
        for g in (0,1):
            for band in range(3):
                for j in range(30):
                    # All cells have normal and difficult labels; heterogeneous
                    # scores provide an actual nontrivial threshold LP.
                    y=int(j%7==0)
                    rows.append({"SK_ID_CURR":len(rows)+100000,"TARGET":y,"thin":g,
                                 "income_band":band,"AMT_CREDIT":100.+j,
                                 "AMT_INCOME_TOTAL":1000.+band*100,
                                 "probability":min(.98,.015*j+.05*g+.02*band+.25*y)})
        return pd.DataFrame(rows)

    def test_frozen_mixtures_and_common_face(self):
        data=self.frame()
        cfg={"minimum_approvals":1,"minimum_subgroup_normal":1,"grid":20,
             "primary_risk_budget":.2,"primary_epsilon":.2,"betas":[.1]}
        frozen=fit_allocation_policies(data,cfg)
        self.assertEqual(len(frozen["policies"]),4)
        base=frozen["policies"][0]
        self.assertTrue(base["feasible"])
        new=frozen["policies"][1:]
        self.assertEqual(len(set(p["minimum_required_delta"] for p in new)),1)
        self.assertTrue(all(p["feasible"] for p in new),new)
        for policy in new:
            q=apply_protected_policy(data,policy)
            # Label / audit-column removal does not change application.
            np.testing.assert_array_equal(q,apply_protected_policy(data[["probability","thin"]],policy))
            for g in (0,1):
                order=np.argsort(data.loc[data.thin==g,"probability"].to_numpy())
                self.assertTrue(np.all(np.diff(q[data.thin.to_numpy()==g][order])<=1e-10))
            self.assertLessEqual(policy["validation_max_deterioration"],policy["minimum_required_delta"]+3e-8)
            self.assertGreaterEqual(policy["validation_metrics"]["utility"],.9*base["validation_metrics"]["utility"]-1e-6)
            # Independent normal-label cell denominators.
            for g in (0,1):
                for b in range(3):
                    mask=(data.thin==g)&(data.income_band==b)&(data.TARGET==0)
                    expected=1-q[mask].mean()
                    self.assertAlmostEqual(expected,policy["validation_metrics"][f"subgroup_{g}_{b}_frr"],places=10)


if __name__ == "__main__":
    unittest.main()
