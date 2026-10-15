"""Audit saved training membership, visibility, validation provenance and metrics."""
import json
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import roc_auc_score,brier_score_loss
from run_experiment import HERE,ROOT,OUT,MODELS,CFG,sha,dump


def main():
    checks=[]
    def check(ok,message):
        if not bool(ok):raise AssertionError(message)
        checks.append(message)
    def same(a,b,message):check(np.array_equal(np.sort(a),np.sort(b)),message)
    identity=json.loads((OUT/'identity.json').read_text(encoding='utf-8'))
    for name,h in identity['hashes'].items():check(sha(name)==h,'frozen '+name)
    manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv').set_index('SK_ID_CURR')
    population=pd.read_csv(OUT/'test_population.csv')
    same(population.SK_ID_CURR,manifest[manifest.split=='test'].index,'test membership')
    y=population.TARGET.to_numpy();g=population.thin.to_numpy()
    frozen=json.loads((OUT/'frozen_all.json').read_text(encoding='utf-8'))['records']
    data=pd.read_csv(OUT/'metrics.csv')
    check(len(data)==701 and len(frozen)==701,'701 evaluation rows')
    check(data.model.nunique()==220,'220 unique models')
    # Static sample sizes, groups, and weights verified against the original pool.
    base_ids=manifest[manifest.split=='train'].index.to_numpy()
    for rate in CFG['retentions']:
        for seed in CFG['seeds']:
            records={}
            for condition in ['random','group_only','biased','corrected']:
                name=f'a_r{int(rate*100)}_s{seed}_{condition}'
                with np.load(OUT/'training'/f'{name}.npz') as f:
                    selected=manifest.loc[f['ids']];w=f['weights'];records[condition]=selected
                    check(selected.index.is_unique and (selected.split=='train').all(),name+' eligible unique train')
                    check(np.isclose(w.mean(),1),name+' normalized weights')
                    if condition!='corrected':check(np.all(w==1),name+' equal weights')
                    else:
                        ref=manifest[manifest.split=='reference'];cells=2*selected.thin+selected.TARGET
                        counts=np.bincount(cells,minlength=4)/len(cells)
                        reference=np.bincount(2*ref.thin+ref.TARGET,minlength=4)/len(ref)
                        expected=(reference/counts)[cells]
                        check(np.allclose(w,expected/expected.mean()),name+' reference weights')
            check(len({len(v) for v in records.values()})==1,f'A sizes {rate} {seed}')
            b=records['biased'];o=records['group_only']
            for group in [0,1]:
                check((b.thin==group).sum()==(o.thin==group).sum(),f'A group counts {rate} {seed} {group}')
                fraction=manifest.loc[base_ids].query('thin == @group').TARGET.mean()
                check(abs(o[o.thin==group].TARGET.sum()-(o.thin==group).sum()*fraction)<=.500001,f'A label ratio {rate} {seed} {group}')
            same(b.index,records['corrected'].index,'correction uses same records')
            base=manifest.loc[base_ids]
            check(len(b[(b.thin==1)&(b.TARGET==0)])==int(np.floor(len(base[(base.thin==1)&(base.TARGET==0)])*rate)),'biased retained good count')
            check(len(b)-len(b[(b.thin==1)&(b.TARGET==0)])==len(base)-len(base[(base.thin==1)&(base.TARGET==0)]),'other cells not removed')
    for row in [r for r in frozen if r['module']=='B']:
        with np.load(OUT/'training'/f'{row["model"]}.npz') as f:
            selected=manifest.loc[f['ids']];w=f['weights'];c,lam=CFG['losses'][row['loss']]
            target=np.where(selected.TARGET==1,float(c),1.)
            for group in [0,1]:
                mask=(selected.TARGET==0)&(selected.thin==group)
                target[mask]+=lam*len(selected)/(2*mask.sum())
            check(np.allclose(w,target/target.mean()),row['model']+' objective weights')
            base='a_full' if row['dataset']=='full' else f'a_r50_s{row["seed"]}_biased'
            with np.load(OUT/'training'/f'{base}.npz') as a:same(f['ids'],a['ids'],'B same training records')
    # Replay only visible ledger outcomes to reconstruct all permitted information.
    for seed in CFG['seeds']:
        stream=pd.read_csv(OUT/f'c_stream_{seed}.csv').set_index('SK_ID_CURR')
        base=stream.index[stream.initial_visible==1].to_numpy()
        roles=stream.role
        role_map=roles.to_dict()
        with np.load(OUT/f'c_external_{seed}.npz') as f:external=[f['cal_ids'],f['val_ids']]
        for ext,kind in zip(external,['calibration','validation']):same(ext,manifest[manifest.split==kind].index,'external origin')
        with np.load(OUT/'training'/f'c_s{seed}_initial.npz') as f:
            same(f['ids'],[i for i in base if role_map[i]==0],'initial training roles')
            check(np.all(f['weights']==1),'initial equal weights')
        with np.load(OUT/'policies'/f'c_s{seed}_initial.npz') as f:
            same(f['cal_ids'],[i for i in base if role_map[i]==1],'initial calibration roles')
            same(f['val_ids'],[i for i in base if role_map[i]==2],'initial validation roles')
        for mode in CFG['modes']:
            for acq in CFG['acquisitions']:
                observed=set(base);weights={int(i):1. for i in base if roles.loc[i]==0}
                previous=f'c_s{seed}_initial'
                for r in range(1,6):
                    name=f'c_s{seed}_{mode}_{acq}_r{r}'
                    ledger=pd.read_csv(OUT/'ledgers'/f'{name}.csv.gz',float_precision='round_trip').set_index('SK_ID_CURR')
                    same(ledger.index,stream[stream.arrival==r].index,name+' correct batch')
                    check(not (set(ledger.index)&observed),name+' no repeated arrivals')
                    prev=json.loads((MODELS/f'{previous}_decision.json').read_text(encoding='utf-8'))
                    check(np.array_equal(ledger.proposed,ledger.probability<=prev['rule']['threshold']),name+' prior policy')
                    eps=.1 if acq=='explore_ipw' else 0.
                    u=np.random.default_rng(seed*100+r).random(len(ledger))
                    check(np.array_equal(ledger.explored,(ledger.proposed==0)&(u<eps)),name+' exploration draw')
                    check(np.array_equal(ledger.visible,(ledger.proposed==1)|(ledger.explored==1)),name+' visibility')
                    check(np.array_equal(ledger.pi,np.where(ledger.proposed==1,1.,eps)),name+' known propensities')
                    check(ledger.loc[ledger.visible==0,'outcome'].isna().all(),name+' hidden outcomes absent')
                    visible=ledger[ledger.visible==1]
                    check(np.array_equal(visible.outcome,manifest.loc[visible.index,'TARGET']),name+' observed labels correct')
                    check(np.array_equal(ledger.role,roles.loc[ledger.index]),name+' ex ante roles')
                    observed.update(visible.index.tolist())
                    training_visible=visible[visible.role==0]
                    weights.update(zip(training_visible.index.tolist(),(1/training_visible.pi if acq=='explore_ipw' else np.ones(len(training_visible))).tolist()))
                    with np.load(OUT/'training'/f'{name}.npz') as f:
                        same(f['ids'],list(weights),name+' only observed training roles')
                        expected=np.array([weights[int(i)] for i in f['ids']]);expected/=expected.mean()
                        check(np.allclose(expected,f['weights']),name+' IPW weights')
                    internal=[np.array([i for i in observed if role_map[i]==role]) for role in [1,2]]
                    initial=[np.array([i for i in base if role_map[i]==role]) for role in [1,2]]
                    for info in CFG['modes']:
                        key=name if info==mode else name+'_shadow_'+info
                        wanted=internal if info=='observed' else initial if info=='frozen' else [external[j][:len(internal[j])] for j in [0,1]]
                        with np.load(OUT/'policies'/f'{key}.npz') as f:
                            same(f['cal_ids'],wanted[0],key+' calibration provenance')
                            same(f['val_ids'],wanted[1],key+' validation provenance')
                            check(not set(f['cal_ids'])&set(weights) and not set(f['val_ids'])&set(weights),key+' holdout exclusion')
                    previous=name
    # Every frozen policy: independent threshold feasibility/maximality and held-out metrics.
    done=set()
    for row in frozen:
        d=row['policy'];key=d['key']
        if key in done:continue
        done.add(key)
        with np.load(OUT/'policies'/f'{key}.npz') as f:
            ids=f['val_ids'];pv=f['val_probability'];labels=manifest.loc[ids,'TARGET'].to_numpy()
            idx=np.argsort(pv,kind='stable');scores=pv[idx];ordered=labels[idx]
            end=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
            counts=end+1;bad=np.cumsum(ordered)[end]
            minimum=CFG['feedback_minimum'] if row['module'].startswith('C') else CFG['static_minimum']
            ok=(counts>=minimum)&(bad<=.05*counts+1e-12)
            expected=float(scores[end[np.flatnonzero(ok)[-1]]]) if ok.any() else -1.
            check(d['rule']['threshold']==expected,key+' optimal validation prefix')
        with np.load(OUT/'predictions'/f'{key}.npz') as f:p=f['probability'];a=f['approved']
        check(np.array_equal(a,p<=d['rule']['threshold']),key+' test policy fixed')
        with np.load(OUT/'predictions'/f'{row["model"]}_raw.npz') as f:raw=f['probability']
        coef=d['calibration'];expected=expit(coef['slope']*logit(np.clip(raw,1e-7,1-1e-7))+coef['intercept'])
        check(np.allclose(p,expected),key+' score transformation')
        f1=(~a[(g==1)&(y==0)]).mean();f0=(~a[(g==0)&(y==0)]).mean()
        values=dict(auc=roc_auc_score(y,p),brier=brier_score_loss(y,p),thin_frr=f1,thick_frr=f0,abs_frr_gap=abs(f1-f0),approval_rate=a.mean(),approved_risk=y[a].mean() if a.any() else np.nan)
        for _,saved in data[data.policy_key==key].iterrows():
            for k,v in values.items():check(np.isclose(saved[k],v,equal_nan=True),key+' '+k)
    dump(OUT/'verification.json',{'passed':True,'checks':len(checks),'unique_policies':len(done),'scope':'Saved input provenance, role/visibility replay, weights, validation thresholds, test metrics; not proof of real-world validity.'})
    print(f'Independent audit passed: {len(checks)} checks; {len(done)} policies.')


if __name__=='__main__':main()
