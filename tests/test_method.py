import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from experiment import biased_indices, correction_weights, choose_threshold, split_indices, metrics


class MethodTests(unittest.TestCase):
    def test_selection_changes_only_thin_good_count(self):
        g = np.repeat([0,0,1,1],100)
        y = np.repeat([0,1,0,1],100)
        rows = biased_indices(y,g,.25,17)
        self.assertEqual(np.bincount(2*g[rows]+y[rows],minlength=4).tolist(),[100,100,25,100])
        self.assertEqual(len(rows),len(set(rows)))

    def test_weights_restore_joint_not_independence(self):
        reference = np.repeat(np.arange(4),[600,100,200,100])
        biased = np.repeat(np.arange(4),[600,100,50,100])
        w, _ = correction_weights(biased%2,biased//2,reference%2,reference//2)
        actual = np.bincount(biased,weights=w)/w.sum()
        np.testing.assert_allclose(actual,np.array([.6,.1,.2,.1]))
        self.assertNotAlmostEqual(actual[1]/sum(actual[:2]),actual[3]/sum(actual[2:]))

    def test_split_disjoint_and_exhaustive(self):
        cell = np.tile(np.arange(4),1000)
        split = split_indices(cell%2,cell//2,123)
        joined = np.concatenate(list(split.values()))
        self.assertEqual(len(joined),len(np.unique(joined)))
        self.assertEqual(set(joined),set(range(len(cell))))
        self.assertEqual([len(v) for v in split.values()],[2000,400,400,400,800])

    def test_threshold_respects_ties(self):
        p = np.array([.1,.1,.2,.2,.3])
        y = np.array([0,1,0,0,1])
        rule = choose_threshold(p,y,.26,1)
        self.assertEqual(rule['threshold'],.2)
        self.assertEqual(rule['validation_approvals'],4)

    def test_infeasible_threshold_rejects_all(self):
        r = choose_threshold(np.array([.1,.2]),np.array([1,1]),.05,1)
        self.assertFalse(r['feasible'])
        self.assertEqual(r['validation_approvals'],0)

    def test_frr_label_direction(self):
        y=np.tile([0,1,0,1],5); g=np.tile([1,1,0,0],5)
        m=metrics(y,g,np.tile([.2,.8,.1,.9],5),np.tile([False,False,True,False],5))
        self.assertEqual(m['thin_frr'],1)
        self.assertEqual(m['thick_frr'],0)
        self.assertEqual(m['signed_frr_gap'],1)
        self.assertEqual(m['approved_risk'],0)


if __name__ == '__main__': unittest.main()
