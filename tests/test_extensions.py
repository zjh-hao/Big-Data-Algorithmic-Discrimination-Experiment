import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from extensions import fair_rule, apply_rule, accuracy_rule, reveal_candidates


class ExtensionTests(unittest.TestCase):
    def test_fair_rule_matches_exhaustive_enumeration(self):
        p=np.array([.1,.2,.3,.4,.05,.15,.25,.35]); y=np.array([0,0,1,1,0,1,0,1]);g=np.repeat([0,1],4)
        rule=fair_rule(p,y,g,.3,.5,2,512,.15)
        candidates=[]
        for t0 in np.r_[-1,p[g==0],.15]:
            for t1 in np.r_[-1,p[g==1],.15]:
                a=p<=np.where(g==0,t0,t1)
                gap=abs((~a[(g==0)&(y==0)]).mean()-(~a[(g==1)&(y==0)]).mean())
                if a.sum()>=2 and y[a].mean()<=.3 and gap<=.5: candidates.append(a.sum())
        a=apply_rule(p,g,rule)
        self.assertEqual(a.sum(),max(candidates))
        self.assertLessEqual(y[a].mean(),.3)

    def test_infeasible_policy_is_explicit(self):
        p=np.array([.1,.2,.3,.4]);y=np.array([1,0,1,0]);g=np.array([0,0,1,1])
        rule=fair_rule(p,y,g,0,.01,2,512,.2)
        self.assertFalse(rule['feasible'])
        self.assertFalse(apply_rule(p,g,rule).any())

    def test_accuracy_rule_optimizes_errors_and_respects_ties(self):
        p=np.array([.1,.1,.2,.3]);y=np.array([0,1,1,1])
        r=accuracy_rule(p,y);a=apply_rule(p,np.zeros(4,dtype=int),r)
        self.assertEqual(np.sum((~a)!=y),1)
        self.assertEqual(a[0],a[1])

    def test_exploration_and_ipw_are_known_probabilities(self):
        a=np.array([True,False,False,False]);u=np.array([.99,.01,.2,.05])
        visible,explored,pi=reveal_candidates(a,u,.1)
        np.testing.assert_array_equal(visible,[True,True,False,True])
        np.testing.assert_array_equal(explored,[False,True,False,True])
        np.testing.assert_allclose(1/pi[visible],[1,10,10])
        # A denied observation cannot appear in training at zero exploration.
        obs,_,prob=reveal_candidates(a,u,0)
        np.testing.assert_array_equal(obs,a)
        self.assertTrue(np.all(prob[~obs]==0))


if __name__=='__main__':unittest.main()
