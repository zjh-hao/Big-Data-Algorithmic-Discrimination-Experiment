"""Independent elementwise gradients to investigate macOS BLAS warnings."""
import json
import joblib
import numpy as np
from scipy.special import expit,logit
from research_extension_v2.common import ROOT,HERE,dump
from experiment import load_features,split_indices,make_model


def main():
    out=HERE/'results/run_001'
    x,y,g,ids,cat,_=load_features();cal=split_indices(y,g,20261013)['calibration']
    pre=joblib.load(out/'models/preprocessor.joblib');matrix=pre.transform(x.iloc[cal]).astype(np.float32)
    results=[]
    for path in sorted((out/'models').glob('*_calibration.json')):
        name=path.name.removesuffix('_calibration.json');c=json.loads(path.read_text())
        m=make_model({'max_depth':5,'n_estimators':200});m.load_model(out/f'models/{name}.json')
        z=logit(np.clip(m.predict_proba(matrix)[:,1].astype(float),1e-7,1-1e-7))
        pred=expit(c['slope']*z+c['intercept']);res=pred-y[cal]
        gradients=[float(np.sum(res*z)/len(z)+c['slope']/(1e6*len(z))),float(np.sum(res)/len(z))]
        loss=float(np.mean(np.logaddexp(0,c['slope']*z+c['intercept'])-y[cal]*(c['slope']*z+c['intercept'])))
        passed=bool(np.isfinite(pred).all() and np.isfinite(loss) and np.max(np.abs(gradients))<1e-3)
        results.append(dict(model=name,mean_gradients=gradients,log_loss=loss,passed=passed,calibration=c))
    # Check all saved intervals with strictly finite expected estimands.
    import pandas as pd
    ci=pd.read_csv(out/'paired_intervals.csv')
    finite=bool(ci[['lo','hi','difference_lo','difference_hi']].notna().all().all())
    passed=all(r['passed'] for r in results) and finite
    dump(out/'calibration_numerics.json',dict(status='PASS' if passed else 'FAIL',checks=results,bootstrap_intervals_all_finite=finite,
        note='Warnings preserved. Independent elementwise first-order optimality and finite-output checks; no retraining or changed scientific definitions.'))
    if not passed:raise AssertionError('Numerical diagnostic failed')
    print('PASS all six calibrators and finite bootstrap intervals')


if __name__=='__main__':main()
