"""Independent reconstruction of exported decisions and metrics (no retraining)."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss
from experiment import ROOT, CFG, dump, sha


def main():
    checks=[]
    def check(condition,name):
        if not bool(condition): raise AssertionError(name)
        checks.append(name)
    manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv')
    check(manifest.SK_ID_CURR.is_unique,'Customer IDs never overlap across five splits')
    check(len(manifest)==307511,'All labeled applications retained in split manifest')
    index=manifest.set_index('SK_ID_CURR')
    training=index[index.split=='train']
    reference=index[index.split=='reference']
    test=index[index.split=='test']
    val=index[index.split=='validation']
    results=pd.read_csv(ROOT/'results/metrics.csv')
    decisions=json.loads((ROOT/'results/frozen_decisions.json').read_text())
    check(len(decisions)==31 and len(results)==93,'31 models, 3 risk budgets each')
    features=json.loads((ROOT/'results/features.json').read_text())
    check(not any(c in ['SK_ID_CURR','TARGET'] or c.startswith('CODE_GENDER') for c in features),'IDs, target and gender excluded from model inputs')
    audit=json.loads((ROOT/'results/data_audit.json').read_text())
    for filename,record in audit['source_files'].items():
        check(sha(ROOT/'data/raw'/filename)==record['sha256'],f'Input hash unchanged: {filename}')
    for d in decisions:
        name=d['name']; source=ROOT/f'results/predictions/{name}_test.csv'
        p=pd.read_csv(source,float_precision='round_trip')
        validation=pd.read_csv(ROOT/f'results/predictions/{name}_validation.csv',float_precision='round_trip')
        # XGBoost + numpy/scipy calibration operates in float32. CSV emits its
        # shortest round-trip decimal; restore source dtype before boundary tests.
        p['probability']=p.probability.astype(np.float32)
        validation['probability']=validation.probability.astype(np.float32)
        used=pd.read_csv(ROOT/f'data/processed/{name}_training.csv')
        check(set(p.SK_ID_CURR)==set(test.index) and p.SK_ID_CURR.is_unique,name+' test membership')
        check(set(validation.SK_ID_CURR)==set(val.index),name+' validation membership')
        check(set(used.SK_ID_CURR).issubset(set(training.index)),name+' training-only membership')
        check(used.SK_ID_CURR.is_unique and len(used)==d['train_size'],name+' unique training size')
        check(np.array_equal(p.TARGET,index.loc[p.SK_ID_CURR,'TARGET']),name+' real labels unchanged')
        check(np.array_equal(p.thin,index.loc[p.SK_ID_CURR,'thin']),name+' group labels unchanged')
        check(np.isfinite(p.probability).all() and p.probability.between(0,1).all(),name+' valid probabilities')
        check(source.stat().st_mtime >= (ROOT/'results/frozen_decisions.json').stat().st_mtime,name+' evaluated after decision freeze')
        labels=index.loc[used.SK_ID_CURR]
        cells=(2*labels.thin+labels.TARGET).to_numpy()
        counts=np.bincount(cells,minlength=4)
        check(np.array_equal(counts,d['cell_counts']),name+' cell counts')
        if d['condition'] in ['biased','corrected']:
            original=np.bincount((2*training.thin+training.TARGET).to_numpy(),minlength=4)
            expected=original.copy(); expected[2]=int(np.floor(original[2]*d['retention']))
            check(np.array_equal(counts,expected),name+' only thin TARGET=0 selected away')
        if d['condition']=='corrected':
            bias=pd.read_csv(ROOT/f'data/processed/{name.replace("corrected","biased")}_training.csv')
            random=pd.read_csv(ROOT/f'data/processed/{name.replace("corrected","random")}_training.csv')
            check(np.array_equal(used.SK_ID_CURR,bias.SK_ID_CURR),name+' exactly same rows as biased')
            check(len(random)==len(used),name+' equal-size random control')
            ref=np.bincount((2*reference.thin+reference.TARGET).to_numpy(),minlength=4)
            proportions=np.bincount(cells,weights=used.weight,minlength=4)/used.weight.sum()
            check(np.allclose(proportions,ref/ref.sum()),name+' weights restore reference joint proportions')
        for budget in CFG['risk_budgets']:
            row=results[(results.name==name)&np.isclose(results.budget,budget)].iloc[0]
            threshold=d['rules'][str(budget)]['threshold']
            accepted=(p.probability<=threshold).to_numpy()
            check(np.array_equal(accepted,p[f'approved_{budget}'].astype(bool)),f'{name} {budget} decisions')
            approved_val=validation.probability<=threshold
            check(approved_val.sum()==d['rules'][str(budget)]['validation_approvals'],f'{name} {budget} validation count')
            if approved_val.any():
                check(validation.loc[approved_val,'TARGET'].mean()<=budget+1e-12,f'{name} {budget} validation risk')
            good_thin=(p.TARGET==0)&(p.thin==1); good_thick=(p.TARGET==0)&(p.thin==0)
            thin=np.mean(~accepted[good_thin]); thick=np.mean(~accepted[good_thick])
            manual={'thin_frr':thin,'thick_frr':thick,'signed_frr_gap':thin-thick,'abs_frr_gap':abs(thin-thick),
                    'approval_rate':accepted.mean(),'approved_risk':p.loc[accepted,'TARGET'].mean(),
                    'auc':roc_auc_score(p.TARGET,p.probability),'brier':brier_score_loss(p.TARGET,p.probability)}
            check(all(np.isclose(row[key],value,atol=1e-10,equal_nan=True) for key,value in manual.items()),f'{name} {budget} independently recomputed metrics')
    check(len(pd.read_csv(ROOT/'results/paired_intervals.csv'))==60,'All preset comparisons and intervals retained')
    for number,name in enumerate(['mechanism','risk_fairness','correction_effect','explanation','calibration'],1):
        for ext in ['png','pdf']:
            check((ROOT/f'figures/{number:02d}_{name}.{ext}').stat().st_size>1000,f'Figure {number} {ext} exported')
    dump(ROOT/'results/verification.json',{'passed':True,'check_count':len(checks),'checks':checks})
    print(f'{len(checks)} independent checks passed.')


if __name__=='__main__': main()
