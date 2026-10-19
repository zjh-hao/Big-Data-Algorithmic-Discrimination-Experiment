import inspect
import unittest
import numpy as np
import pandas as pd
from research_extension_v2.common import CFG, validate
from research_extension_v2.policy_noharm import candidate_space, solve, approve, fit_policies
from research_extension_v2.subgroup_audit import fit_income_boundaries, income_groups, audit
from research_extension_v2.selective_label_ipw import exploration_plan, reveal, estimate, oracle_metrics
from research_extension_v2.exploration_risk import risk_audit
from research_extension_v2.statistics import auc_interval, joint_bootstrap
from extensions import fair_rule


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.p = np.array([.1, .1, .3, .4, .05, .15, .25, .35])
        self.y = np.array([0, 0, 1, 1, 0, 1, 0, 1]); self.g = np.repeat([0, 1], 4)
        self.s = candidate_space(self.p, self.y, self.g, .3, 2, 512)

    def test_all_constraints(self):
        b = solve(self.s); r = solve(self.s, epsilon=.5, baseline=b, delta=.1)
        self.assertEqual(r['status'], 'PASS'); a = approve(self.p, self.g, r)
        self.assertLessEqual(self.y[a].mean(), .3)
        for group, name in [(0, 'thick'), (1, 'thin')]:
            self.assertLessEqual((~a[(self.g==group)&(self.y==0)]).mean(), b['validation_'+name+'_frr']+.1+1e-12)
        self.assertLessEqual(r['validation_abs_frr_gap'], .5)

    def test_infeasible_no_fake_rejection(self):
        s = candidate_space([.1,.2,.1,.2], [1,0,1,0], [0,0,1,1], 0, 2, 512)
        r = solve(s, epsilon=0)
        self.assertEqual(r['status'], 'INFEASIBLE'); self.assertIsNone(r['thresholds'])
        with self.assertRaises(ValueError): approve(self.p, self.g, r)

    def test_space_identical(self):
        b = solve(self.s); f = solve(self.s, epsilon=.5); n = solve(self.s, epsilon=.5, baseline=b, delta=.1)
        self.assertEqual(len({r['candidate_fingerprint'] for r in [b,f,n]}), 1)
        legacy = fair_rule(self.p,self.y,self.g,.3,.5,2,512,.15)
        # Same space's optimum agrees with exhaustive threshold enumeration.
        counts = []
        for i,t0 in enumerate(self.s['t0']):
            for j,t1 in enumerate(self.s['t1']):
                a = self.p <= np.where(self.g==0,t0,t1)
                if a.sum()>=2 and self.y[a].mean()<=.3 and self.s['gap'][i,j]<=.5: counts.append(a.sum())
        self.assertEqual(f['validation_approvals'], max(counts))

    def test_ties_and_order_invariant(self):
        b = solve(self.s); a = approve(self.p, self.g, b)
        self.assertEqual(a[0], a[1])
        ix = np.array([7,6,5,4,3,2,1,0])
        r = solve(candidate_space(self.p[ix],self.y[ix],self.g[ix],.3,2,512))
        self.assertEqual(r['thresholds'], b['thresholds'])

    def test_only_validation_api(self):
        self.assertEqual(list(inspect.signature(fit_policies).parameters), ['validation', 'config'])
        self.assertNotIn('test', inspect.getsource(fit_policies))

    def test_noharm_can_remove_gap_solution(self):
        rng=np.random.default_rng(8); seen=False
        for _ in range(100):
            p=rng.random(40); y=np.tile([0,0,0,1],10); g=np.repeat([0,1],20)
            s=candidate_space(p,y,g,.2,2,512); b=solve(s)
            f=solve(s,epsilon=.01); n=solve(s,epsilon=.01,baseline=b,delta=0)
            if f['feasible'] and n.get('noharm_removed_candidates',0)>0: seen=True; break
        self.assertTrue(seen)


class SubgroupTests(unittest.TestCase):
    def test_training_only_boundaries(self):
        train=np.arange(100); cuts=fit_income_boundaries(train, CFG['income_quantiles'])
        np.testing.assert_allclose(cuts,[33,66])
        np.testing.assert_array_equal(income_groups([0,33,34,66,67,10000],cuts),[0,0,1,1,2,2])
        np.testing.assert_array_equal(cuts, fit_income_boundaries(train, CFG['income_quantiles']))

    def test_empty_and_small(self):
        rows=audit(np.array([0,1]),np.array([0,0]),[.1,.2],{'business':np.array([1,0],bool)},np.array([1,0],bool),[10,10],[20,30],100,100)
        self.assertEqual(len(rows),6)
        self.assertTrue(all(r['status']=='INSUFFICIENT_SUPPORT' for r in rows))
        self.assertIsNone(rows[-1]['frr']); self.assertIsNone(rows[-1]['approved_risk'])


class ReplayTests(unittest.TestCase):
    def make(self, rate=1):
        ids=np.arange(8); g=np.repeat([0,1],4); p=np.tile([1,0,0,1],2).astype(bool)
        y=np.tile([0,0,1,0],2); plan=exploration_plan(ids,g,p,rate,np.linspace(0,.99,8))
        m=plan.visible==1; ledger=reveal(plan,ids[m],y[m]); return ledger,plan,y

    def test_pi_one(self):
        ledger,_,_=self.make()
        for r in estimate(ledger,1):
            self.assertAlmostEqual(r['naive_frr'],r['ipw_frr']); self.assertEqual(r['ess_good'],3)

    def test_zero_not_identifiable(self):
        ledger,_,_=self.make(0)
        for r in estimate(ledger,1):
            self.assertEqual(r['status'],'NOT_IDENTIFIABLE'); self.assertIsNone(r['ipw_frr'])
            self.assertEqual(r['naive_frr'],0)

    def test_estimator_only_visible(self):
        ledger,_,_=self.make(.1); hidden=ledger.visible==0
        ledger.loc[hidden,'outcome']=1
        with self.assertRaises(ValueError): estimate(ledger,1)

    def test_oracle_separate_no_label_decisions(self):
        self.assertEqual(list(inspect.signature(estimate).parameters),['ledger','minimum_ess'])
        self.assertNotIn('y',inspect.signature(exploration_plan).parameters)
        ledger,plan,y=self.make()
        oracle=pd.DataFrame(dict(SK_ID_CURR=plan.SK_ID_CURR,TARGET=y)).sample(frac=1,random_state=7)
        r=oracle_metrics(oracle,plan)
        self.assertAlmostEqual(r[0],1/3)
        self.assertEqual(estimate(ledger,1)[0]['ipw_frr'],r[0])

    def test_exploration_actual_risk(self):
        ledger,_,y=self.make(); r=risk_audit(ledger)
        self.assertEqual(r['actual_n'],8); self.assertEqual(r['exploration_n'],4)
        self.assertEqual(r['actual_risk'],y.mean()); self.assertEqual(r['regular_risk'],0)
        self.assertAlmostEqual(r['actual_risk'],(r['regular_n']*r['regular_risk']+r['exploration_n']*r['exploration_risk'])/r['actual_n'])

    def test_positive_prob_no_visible_good(self):
        l,_,_=self.make(.01); l.loc[l.visible==1,'outcome']=1
        self.assertEqual(estimate(l,1)[0]['status'],'INSUFFICIENT_SUPPORT')

    def test_id_alignment(self):
        l,plan,y=self.make()
        with self.assertRaises(ValueError): reveal(plan,[1,1],[0,0])
        with self.assertRaises(ValueError): validate(pd.DataFrame(dict(SK_ID_CURR=[1,1],TARGET=[0,1],thin=[1,0],probability=[.1,.2])))


class StatisticsTests(unittest.TestCase):
    def test_auc_ties(self):
        from sklearn.metrics import roc_auc_score
        y=np.tile([0,1],100); p=np.linspace(0,1,200).round(1)
        self.assertAlmostEqual(auc_interval(y,p)['auc'],roc_auc_score(y,p))

    def test_joint_pairing(self):
        c,d=joint_bootstrap([[0,0,1,1],[1,1,0,0]],100,17)
        np.testing.assert_array_equal(d.sum(axis=1),4)
        self.assertEqual(len(c),2)


if __name__=='__main__': unittest.main()
