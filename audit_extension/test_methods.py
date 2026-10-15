import unittest
import numpy as np
from run_experiment import sampling_conditions,loss_weights,information_sets

class Methods(unittest.TestCase):
    def setUp(self):
        self.y=np.tile([0,0,0,0,1],200)
        self.g=np.repeat([0,1],500)

    def test_group_count_and_label_ratio(self):
        for rate in [.25,.5]:
            rows=sampling_conditions(self.y,self.g,rate,17)
            sizes=[len(a) for a in rows.values()]
            self.assertEqual(len(set(sizes)),1)
            for group in [0,1]:
                a=rows['group_only'];b=rows['biased']
                self.assertEqual((self.g[a]==group).sum(),(self.g[b]==group).sum())
                n=(self.g[a]==group).sum()
                self.assertLessEqual(abs(self.y[a][self.g[a]==group].sum()-n*.2),.5)
            self.assertGreater(self.y[rows['biased']][self.g[rows['biased']]==1].mean(),.2)

    def test_weighted_loss_is_declared_objective(self):
        rows=sampling_conditions(self.y,self.g,.5,17)['biased']
        y=self.y[rows];g=self.g[rows];loss=np.linspace(.01,2,len(y))
        for c,lam in [(1,0),(5,0),(1,1),(1,4),(5,1)]:
            raw=np.where(y==1,c,1.).astype(float)
            target=np.mean(raw*loss)
            for group in [0,1]:
                mask=(g==group)&(y==0)
                target+=lam/2*loss[mask].mean()
                raw[mask]+=lam*len(y)/(2*mask.sum())
            w=loss_weights(y,g,c,lam)
            self.assertAlmostEqual(np.mean(w*loss),target/raw.mean())
            self.assertAlmostEqual(w.mean(),1)

    def test_information_sources(self):
        roles=np.array([0,1,2,0,1,2,0,1,2])
        initial=np.arange(3);observed=np.arange(6);ext=(np.array([100,101,102]),np.array([200,201,202]))
        frozen=information_sets('frozen',initial,observed,roles,ext)
        visible=information_sets('observed',initial,observed,roles,ext)
        external=information_sets('external',initial,observed,roles,ext)
        self.assertEqual([a.tolist() for a in frozen],[[1],[2]])
        self.assertEqual([a.tolist() for a in visible],[[1,4],[2,5]])
        self.assertEqual([a.tolist() for a in external],[[100,101],[200,201]])
        self.assertFalse(set(np.concatenate(visible))&{6,7,8})

if __name__=='__main__':unittest.main()
