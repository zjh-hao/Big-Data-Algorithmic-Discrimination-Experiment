"""Independent joint-threshold convex oracle and frozen-decision checks."""
import itertools
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from scipy.optimize import OptimizeResult, linprog

from research_extension_v3.protected_policy import (
    apply_protected_policy, evaluate_protected_policies, fit_protected_policies,
    protected_metrics,
)


def fixture():
    rows = []
    for g, band in itertools.product((0, 1), (0, 1, 2)):
        for y, p in ((0, .02+.025*band+.01*g), (0, .45+.04*band+.03*g),
                     (1, .10+.03*band+.02*g)):
            rows.append({"SK_ID_CURR": 100+len(rows), "TARGET": y, "thin": g,
                         "income_band": band, "probability": p,
                         "AMT_CREDIT": 100+13*len(rows)})
    return pd.DataFrame(rows)


def config():
    return {"grid": 50, "minimum_approvals": 2, "minimum_subgroup_normal": 1,
            "primary_loss_ratio": 3, "primary_risk_budget": .25,
            "primary_epsilon": .25, "primary_beta": .20, "primary_delta": .25,
            "policy_specs": [{"name": "gap", "mode": "profit_gap"},
                             {"name": "macro", "mode": "profit_gap_protected"},
                             {"name": "six", "mode": "budgeted_minimax"},
                             {"name": "macro_minimax", "mode": "budgeted_minimax", "protection": "macro"},
                             {"name": "robust", "mode": "budgeted_minimax", "robust": True}]}


def joint_oracle(frame, grid, cfg, epsilon=None):
    """One variable per hard threshold PAIR, rather than two marginal mixtures.

    Every joint distribution induces the two marginal mixtures used by the
    implementation, and independent marginals induce a joint distribution.
    Therefore the linear expected-count feasible sets and optima coincide.
    """
    choices = []
    g, p, y, amount = [frame[k].to_numpy() for k in ("thin", "probability", "TARGET", "AMT_CREDIT")]
    normal = [int(np.sum((g==h)&(y==0))) for h in (0,1)]
    for t0,t1 in itertools.product(grid, repeat=2):
        q = (p <= np.where(g==0,t0,t1)).astype(float)
        good = np.dot(q,1-y)
        bad = np.dot(q,y)
        rates = [np.sum(q[(g==h)&(y==0)])/normal[h] for h in (0,1)]
        choices.append((good-cfg["primary_loss_ratio"]*bad, q.sum(), bad,
                        np.dot(q,amount), rates[0]-rates[1]))
    choices = np.asarray(choices)
    objective = -choices[:,0]
    aub = [-choices[:,1], choices[:,2]-cfg["primary_risk_budget"]*choices[:,1], choices[:,3]/amount.sum()]
    bub = [-cfg["minimum_approvals"],0,cfg.get("capital_fraction",1)]
    if epsilon is not None:
        aub += [choices[:,4],-choices[:,4]]
        bub += [epsilon,epsilon]
    return linprog(objective,A_ub=aub,b_ub=bub,A_eq=[np.ones(len(choices))],b_eq=[1],bounds=(0,1),method="highs")


class ProtectedPolicyTests(unittest.TestCase):
    def test_numerical_retry_preserves_original_constraints(self):
        cfg=config();cfg["policy_specs"]=[]
        ordinary=fit_protected_policies(fixture(),cfg)
        calls=[]
        def fail_then_solve(*args,**kwargs):
            calls.append(kwargs)
            if len(calls)==1:
                return OptimizeResult(success=False,status=4,message="HiGHS 15 Unknown primalBasic")
            return linprog(*args,**kwargs)
        with patch("research_extension_v3.protected_policy.linprog",side_effect=fail_then_solve):
            retried=fit_protected_policies(fixture(),cfg)
        p=retried["policies"][0]
        self.assertEqual(p["status"],"PASS")
        self.assertEqual(len(p["solver_attempts"]),2)
        for name in ("A_ub","b_ub","A_eq","b_eq","bounds"):
            np.testing.assert_array_equal(calls[0][name],calls[1][name])
        self.assertAlmostEqual(p["validation_metrics"]["utility"],ordinary["policies"][0]["validation_metrics"]["utility"],places=8)

    def test_exhausted_numerical_failure_is_not_economic_infeasibility(self):
        failure=OptimizeResult(success=False,status=4,message="Numerical failure")
        with patch("research_extension_v3.protected_policy.linprog",return_value=failure):
            frozen=fit_protected_policies(fixture(),config())
        for p in frozen["policies"]:
            self.assertEqual(p["status"],"SOLVER_ERROR")
            self.assertFalse(p["feasible"])

    def test_actual_lgb731_tight_adaptive_lexicographic_regression(self):
        path=Path(__file__).resolve().parents[1]/"results/run_002/predictions/lightgbm_s731_validation.parquet"
        if not path.exists():
            self.skipTest("Frozen completed-run validation predictions are not present")
        cfg={"grid":256,"minimum_approvals":500,"minimum_subgroup_normal":100,
             "primary_loss_ratio":5,"primary_risk_budget":.05,"capital_fraction":1.,
             "primary_epsilon":.02,"primary_beta":.01,"primary_delta":.01,
             "policy_specs":[{"name":"adaptive_primary","mode":"adaptive_protected_minimax"}]}
        frozen=fit_protected_policies(pd.read_parquet(path),cfg)
        p=frozen["policies"][1]
        self.assertEqual(p["status"],"PASS",p.get("reason"))
        self.assertAlmostEqual(p["minimum_required_delta"],.015886206577999273,places=7)
        self.assertLessEqual(p["validation_max_constraint_violation"],2e-7)
        self.assertGreaterEqual(p["validation_metrics"]["utility"],.99*frozen["policies"][0]["validation_metrics"]["utility"]-2e-7)

    def test_adaptive_minimum_harm_matches_independent_joint_pair_oracle(self):
        frame,cfg=fixture(),config()
        frame["probability"]=[.01+.01*b if g==0 and y==0 else .8+.02*b if g==0 else
                              (.1+.02*b if i%3==0 else .7+.02*b) if y==0 else .4+.02*b
                              for i,(g,b,y) in enumerate(zip(frame.thin,frame.income_band,frame.TARGET))]
        cfg["policy_specs"]=[{"name":"adaptive","mode":"adaptive_protected_minimax","beta":.2,"delta":.01}]
        frozen=fit_protected_policies(frame,cfg)
        baseline,p=frozen["policies"]
        self.assertTrue(p["feasible"])
        y,g,band,score,amount=[frame[k].to_numpy() for k in ("TARGET","thin","income_band","probability","AMT_CREDIT")]
        hard=np.asarray([(score<=np.where(g==0,t0,t1)).astype(float)
                         for t0,t1 in itertools.product(frozen["metadata"]["threshold_grid"],repeat=2)])
        utility=hard@(1-(1+cfg["primary_loss_ratio"])*y)
        n,bad=hard.sum(axis=1),hard@y
        rows,rhs=[],[]
        def add(coeff,b,d=0):
            rows.append(np.r_[coeff,d]);rhs.append(b)
        add(-n,-cfg["minimum_approvals"])
        add(bad-cfg["primary_risk_budget"]*n,0)
        add(hard@amount/amount.sum(),1)
        bm=baseline["validation_metrics"]
        add(-utility,-(1-p["beta"])*bm["utility"])
        rates=[hard[:,(g==h)&(y==0)].mean(axis=1) for h in (0,1)]
        add(rates[0]-rates[1],p["epsilon"]);add(rates[1]-rates[0],p["epsilon"])
        for h,name in ((0,"thick"),(1,"thin")):
            add(-rates[h],bm[name+"_frr"]-1,d=-1)
            for b in (0,1,2):
                rate=hard[:,(g==h)&(band==b)&(y==0)].mean(axis=1)
                add(-rate,bm[f"subgroup_{h}_{b}_frr"]-1,d=-1)
        objective=np.zeros(len(hard)+1);objective[-1]=1
        oracle=linprog(objective,A_ub=rows,b_ub=rhs,A_eq=[np.r_[np.ones(len(hard)),0]],b_eq=[1],bounds=(0,1),method="highs")
        self.assertTrue(oracle.success)
        self.assertAlmostEqual(p["minimum_required_delta"],oracle.fun,places=7)
        self.assertGreater(p["minimum_required_delta"],p["requested_delta"])
        self.assertAlmostEqual(p["effective_delta"],p["minimum_required_delta"]+.001,places=10)
        self.assertEqual(p["delta"],p["effective_delta"])
        self.assertFalse(p["validation_requested_goal_met"])
        self.assertGreaterEqual(p["validation_metrics"]["utility"],(1-p["beta"])*bm["utility"]-2e-7)
        for h,b in itertools.product((0,1),(0,1,2)):
            key=f"subgroup_{h}_{b}_frr"
            self.assertLessEqual(p["validation_metrics"][key],bm[key]+p["effective_delta"]+2e-7)

    def test_adaptive_does_not_relax_risk_or_profit_when_infeasible(self):
        frame,cfg=fixture(),config()
        cfg["minimum_approvals"]=1000
        cfg["policy_specs"]=[{"name":"adaptive","mode":"adaptive_protected_minimax"}]
        frozen=fit_protected_policies(frame,cfg)
        p=frozen["policies"][1]
        self.assertFalse(p["feasible"])
        self.assertEqual(p["status"],"INFEASIBLE")
        self.assertIsNone(p["minimum_required_delta"])

    def test_lp_profit_and_gap_match_independent_joint_pair_oracle(self):
        frame,cfg = fixture(),config()
        frozen = fit_protected_policies(frame,cfg)
        for name,epsilon in [("lp_profit_baseline",None),("gap",cfg["primary_epsilon"])]:
            policy = next(p for p in frozen["policies"] if p["policy_id"]==name)
            oracle = joint_oracle(frame,frozen["metadata"]["threshold_grid"],cfg,epsilon)
            self.assertEqual(policy["feasible"],oracle.success)
            self.assertAlmostEqual(policy["validation_metrics"]["utility"],-oracle.fun,places=6)

    def test_budgeted_minimax_matches_independent_joint_pair_oracle(self):
        frame,cfg = fixture(),config()
        frozen = fit_protected_policies(frame,cfg)
        baseline = frozen["policies"][0]
        selected = next(p for p in frozen["policies"] if p["policy_id"]=="six")
        self.assertTrue(selected["feasible"])
        y,g,band,score,amount = [frame[k].to_numpy() for k in ("TARGET","thin","income_band","probability","AMT_CREDIT")]
        hard = np.asarray([(score <= np.where(g==0,t0,t1)).astype(float)
                           for t0,t1 in itertools.product(frozen["metadata"]["threshold_grid"],repeat=2)])
        utility = hard @ (1-(1+cfg["primary_loss_ratio"])*y)
        n,bad = hard.sum(axis=1),hard@y
        dim = len(hard)+1
        aub,bub = [],[]
        def add(values,b,z=0):
            aub.append(np.r_[values,z]);bub.append(b)
        add(-n,-cfg["minimum_approvals"])
        add(bad-cfg["primary_risk_budget"]*n,0)
        add(hard@amount/amount.sum(),1)
        add(-utility,-(1-selected["beta"])*baseline["validation_metrics"]["utility"])
        rates = [hard[:,(g==h)&(y==0)].mean(axis=1) for h in (0,1)]
        add(rates[0]-rates[1],selected["epsilon"])
        add(rates[1]-rates[0],selected["epsilon"])
        bm = baseline["validation_metrics"]
        for h,name in ((0,"thick"),(1,"thin")):
            add(-rates[h],bm[name+"_frr"]+selected["delta"]-1)
            for b in (0,1,2):
                rate = hard[:,(g==h)&(band==b)&(y==0)].mean(axis=1)
                add(-rate,bm[f"subgroup_{h}_{b}_frr"]+selected["delta"]-1)
                add(-rate,-1,z=-1)
        objective = np.zeros(dim);objective[-1]=1
        oracle = linprog(objective,A_ub=aub,b_ub=bub,A_eq=[np.r_[np.ones(len(hard)),0]],
                         b_eq=[1],bounds=(0,1),method="highs")
        self.assertTrue(oracle.success)
        self.assertAlmostEqual(oracle.fun,selected["validation_minimax_optimum"],places=7)

    def test_monotone_approval_only_uses_score_and_macro_group(self):
        frozen = fit_protected_policies(fixture(),config())
        frame = pd.DataFrame({"thin": np.repeat([0,1],101), "probability": np.tile(np.linspace(0,1,101),2)})
        for p in frozen["policies"]:
            if not p["feasible"]:
                continue
            q = apply_protected_policy(frame,p)
            self.assertTrue(((q>=0)&(q<=1)).all())
            for g in (0,1):
                self.assertTrue((np.diff(q[frame.thin==g]) <= 1e-12).all())
            other = frame.assign(TARGET=1,income_band=2,AMT_CREDIT=1)
            np.testing.assert_array_equal(q,apply_protected_policy(other,p))

    def test_feasible_validation_constraints_and_profit_floor(self):
        cfg,frame = config(),fixture()
        frozen = fit_protected_policies(frame,cfg)
        baseline = frozen["policies"][0]
        bm = baseline["validation_metrics"]
        for p in frozen["policies"]:
            if not p["feasible"]:
                continue
            m = p["validation_metrics"]
            self.assertGreaterEqual(m["approvals"],cfg["minimum_approvals"]-2e-7)
            self.assertLessEqual(m["risk"],p["risk_budget"]+2e-7)
            self.assertLessEqual(m["capital_fraction_used"],p["capital_fraction"]+2e-7)
            self.assertLessEqual(m["utility"],bm["utility"]+2e-7)
            if p["epsilon"] is not None:
                self.assertLessEqual(m["abs_frr_gap"],p["epsilon"]+2e-7)
            if p["beta"] is not None:
                self.assertGreaterEqual(m["utility"],(1-p["beta"])*bm["utility"]-2e-7)
            if p["noharm_delta"] is not None:
                keys = ["thin_frr","thick_frr"]
                if p["protection"]=="six":
                    keys += [f"subgroup_{g}_{b}_frr" for g,b in itertools.product((0,1),(0,1,2))]
                for k in keys:
                    self.assertLessEqual(m[k],bm[k]+p["noharm_delta"]+2e-7)
            if p["mode"]=="budgeted_minimax":
                self.assertLessEqual(m["worst_subgroup_frr"],p["validation_minimax_optimum"]+2e-7)

    def test_six_subgroup_worst_metric_uses_correct_normal_denominator(self):
        frame = fixture()
        q = np.ones(len(frame))
        chosen = (frame.thin==1)&(frame.income_band==2)&(frame.TARGET==0)
        q[chosen] = [.1,.5]
        m = protected_metrics(frame,q,3)
        self.assertAlmostEqual(m["subgroup_1_2_frr"],.7)
        self.assertAlmostEqual(m["worst_subgroup_frr"],.7)
        self.assertEqual(m["subgroup_1_2_normal_records"],2)
        self.assertAlmostEqual(m["thin_frr"],1.4/6)

    def test_evaluation_does_not_refit_on_test_labels_or_change_ledger(self):
        frame,cfg = fixture(),config()
        frozen = fit_protected_policies(frame,cfg)
        before = json.dumps(frozen,sort_keys=True,allow_nan=False)
        first = evaluate_protected_policies(frame,frozen,cfg)
        changed = frame.assign(TARGET=1-frame.TARGET)
        second = evaluate_protected_policies(changed,frozen,cfg)
        self.assertEqual(before,json.dumps(frozen,sort_keys=True,allow_nan=False))
        for a,b in zip(first,second):
            if a["feasible"]:
                self.assertEqual(a["approvals"],b["approvals"])
                self.assertNotEqual(a["utility"],b["utility"])

    def test_infeasible_is_explicit_and_unsupported_cells_rejected(self):
        cfg = config()
        cfg["minimum_approvals"] = 1000
        frozen = fit_protected_policies(fixture(),cfg)
        for p in frozen["policies"]:
            self.assertFalse(p["feasible"])
            self.assertEqual(p["status"],"INFEASIBLE")
            self.assertIsNone(p["mixture"])
            with self.assertRaises(ValueError):
                apply_protected_policy(fixture(),p)
        with self.assertRaises(ValueError):
            fit_protected_policies(fixture().loc[lambda f:f.income_band!=2],config())

    def test_robust_validation_halves_constraints_use_same_baseline(self):
        frame,cfg = fixture(),config()
        frozen = fit_protected_policies(frame,cfg)
        p = next(p for p in frozen["policies"] if p["policy_id"]=="robust")
        if not p["feasible"]:
            self.assertIn("infeasible",p["reason"].lower())
            return
        baseline = frozen["policies"][0]
        for half in (0,1):
            sub = frame.loc[frame.SK_ID_CURR%2==half]
            m = protected_metrics(sub,apply_protected_policy(sub,p),3)
            bm = protected_metrics(sub,apply_protected_policy(sub,baseline),3)
            self.assertLessEqual(m["risk"],p["risk_budget"]+2e-7)
            self.assertLessEqual(m["abs_frr_gap"],p["epsilon"]+2e-7)
            for g,b in itertools.product((0,1),(0,1,2)):
                k = f"subgroup_{g}_{b}_frr"
                if m[k] is not None:
                    self.assertLessEqual(m[k],bm[k]+p["noharm_delta"]+2e-7)


if __name__=="__main__":
    unittest.main()
