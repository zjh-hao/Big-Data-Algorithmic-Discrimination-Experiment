"""Independent record-level arithmetic, threshold optimum and provenance audit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from research_extension_v2.common import ROOT, HERE, sha, dump, CFG


def main(out):
    checks = []
    def check(ok, message):
        if not bool(ok): raise AssertionError(message)
        checks.append(message)
    def close(a,b,message): check(np.isclose(a,b,atol=1e-12,rtol=1e-10,equal_nan=True),message)
    identity=json.loads((out/'identity.json').read_text())
    for f,h in identity['hashes'].items(): check(sha(ROOT/f)==h,'frozen input '+f)
    old=json.loads((out/'legacy_hashes.json').read_text())
    for f,h in old.items(): check(sha(ROOT/f)==h,'legacy unchanged '+f)
    frozen=json.loads((out/'frozen_policies.json').read_text())['policies']
    table=pd.read_csv(out/'COMPARISON_TABLES.csv')
    manifest=pd.read_csv(out/'split_manifest.csv').set_index('SK_ID_CURR')
    check(manifest.index.is_unique,'disjoint/exhaustive manifest IDs')
    raw=pd.read_parquet(ROOT/'data/raw/app_train.parquet').set_index('SK_ID_CURR')
    check(set(raw.index)==set(manifest.index),'full manifest membership')
    check(np.array_equal(raw.loc[manifest.index].TARGET,manifest.TARGET),'manifest original labels')
    cuts=np.array(json.loads((out/'income_boundaries.json').read_text())['cuts'])
    expected=np.quantile(raw.loc[manifest.index[manifest.split=='train'],'AMT_INCOME_TOTAL'].astype(np.float32),CFG['income_quantiles'])
    check(np.array_equal(cuts,expected),'train-only income boundaries')
    sub=pd.read_csv(out/'subgroups.csv')
    for model,policies in frozen.items():
        v=pd.read_csv(out/f'scores/{model}/validation.csv',float_precision='round_trip')
        p=v.probability.to_numpy(); y=v.TARGET.to_numpy(); g=v.thin.to_numpy()
        check(set(v.SK_ID_CURR)==set(manifest.index[manifest.split=='validation']),model+' validation membership')
        check(np.array_equal(y,manifest.loc[v.SK_ID_CURR].TARGET),model+' validation labels')
        # Independently generate legacy grid, including the common risk threshold.
        order=np.argsort(p,kind='stable'); sp=p[order]; sy=y[order]
        ends=np.r_[np.flatnonzero(sp[:-1]!=sp[1:])+1,len(sp)]
        n=ends; bad=np.cumsum(sy)[ends-1]
        ok=(n>=CFG['minimum_approvals'])&(bad<=CFG['risk_budget']*n+1e-12)
        common=float(sp[ends[np.flatnonzero(ok)[-1]]-1]) if ok.any() else -1.
        arrays=[]
        for group in [0,1]:
            mask=g==group; order=np.argsort(p[mask],kind='stable'); sp=p[mask][order]; sy=y[mask][order]
            ends=np.r_[np.flatnonzero(sp[:-1]!=sp[1:])+1,len(sp)]
            if len(ends)>CFG['grid']: ends=ends[np.unique(np.rint(np.linspace(0,len(ends)-1,CFG['grid'])).astype(int))]
            t=np.unique(np.r_[-1.,sp[ends-1],common]); n=np.searchsorted(sp,t,side='right')
            b=np.r_[0,np.cumsum(sy)][n]; f=1-(n-b)/(sy==0).sum(); arrays.append((t,n,b,f))
        t0,n0,b0,f0=arrays[0];t1,n1,b1,f1=arrays[1]
        n=n0[:,None]+n1[None,:];bad=b0[:,None]+b1[None,:];gap=abs(f0[:,None]-f1[None,:])
        base=(n>=CFG['minimum_approvals'])&(bad<=CFG['risk_budget']*n+1e-12)
        digest=hashlib.sha256(t0.astype('<f8').tobytes()+t1.astype('<f8').tobytes()).hexdigest()
        baseline=policies['business']
        test=pd.read_csv(out/f'{model}_test_decisions.csv.gz',float_precision='round_trip')
        check(set(test.SK_ID_CURR)==set(manifest.index[manifest.split=='test']),model+' test membership')
        check(np.array_equal(test.TARGET,raw.loc[test.SK_ID_CURR].TARGET),model+' original test labels')
        for key,r in policies.items():
            possible=base.copy()
            if r['epsilon'] is not None: possible &= gap<=r['epsilon']+1e-12
            if r['delta'] is not None:
                possible &= f0[:,None]<=baseline['validation_thick_frr']+r['delta']+1e-12
                possible &= f1[None,:]<=baseline['validation_thin_frr']+r['delta']+1e-12
            check(r['candidate_fingerprint']==digest,model+' '+key+' same candidates')
            check(r['feasible']==possible.any(),model+' '+key+' feasibility')
            if not possible.any():
                check(r['thresholds'] is None,model+' '+key+' no fake reject-all');continue
            ii,jj=np.where(possible)
            best=min(zip(ii,jj),key=lambda z:(-n[z],bad[z]/n[z],gap[z],z[0],z[1]))
            check(r['indices']==list(best),model+' '+key+' independent grid optimum')
            a=test.probability.to_numpy()<=np.array(r['thresholds'])[test.thin.to_numpy()]
            check(np.array_equal(a,test[key]),model+' '+key+' frozen application')
            saved=table[(table.model==model)&(table.policy==key)].iloc[0]
            frr=[]
            for group,name in [(0,'thick'),(1,'thin')]:
                m=(test.thin==group)&(test.TARGET==0);value=float((~a[m]).mean());frr.append(value)
                close(saved[name+'_frr'],value,model+' '+key+' '+name+' FRR')
            close(saved.abs_frr_gap,abs(frr[0]-frr[1]),model+' '+key+' gap')
            close(saved.approval_rate,a.mean(),model+' '+key+' approval rate')
            close(saved.approved_count,a.sum(),model+' '+key+' approval count')
            close(saved.approved_risk,test.loc[a,'TARGET'].mean(),model+' '+key+' risk')
            close(saved.auc,roc_auc_score(test.TARGET,test.probability),model+' '+key+' AUC')
        if (out/f'models/{model}_training.csv').exists():
            train=pd.read_csv(out/f'models/{model}_training.csv')
            check(train.SK_ID_CURR.is_unique and (manifest.loc[train.SK_ID_CURR].split=='train').all(),model+' training exclusion')
        bands=np.searchsorted(cuts,test.income.to_numpy(),side='left')
        for _,r in sub[sub.model==model].iterrows():
            group=1 if r.group=='thin' else 0; band=['low','middle','high'].index(r.income)
            mask=(test.thin==group)&(bands==band);good=mask&(test.TARGET==0);a=test[r.policy].to_numpy(bool)
            close(r.n,mask.sum(),model+' subgroup size');close(r.normal_count,good.sum(),model+' subgroup normal size')
            close(r.frr,(~a[good]).mean(),model+' subgroup FRR')
            close(r.frr_change,((~a[good]).astype(float)-(~test.business.to_numpy(bool)[good]).astype(float)).mean(),model+' subgroup paired change')
    estimates=pd.read_csv(out/'ipw_estimates.csv');risk=pd.read_csv(out/'exploration_risk.csv')
    for path in sorted((out/'ledgers').glob('*.csv.gz')):
        key=path.name.removesuffix('.csv.gz');model,seed,rate=key.rsplit('_',2);seed=int(seed[1:]);rate=float(rate[1:])
        l=pd.read_csv(path,float_precision='round_trip');plan=pd.read_csv(out/'plans'/path.name,float_precision='round_trip')
        check(np.array_equal(l.SK_ID_CURR,plan.SK_ID_CURR),key+' plan IDs')
        for col in plan: check(np.array_equal(l[col],plan[col]),key+' frozen plan '+col)
        u=np.random.default_rng(CFG['simulation_seed_offset']+seed).random(len(l))
        expected_explore=(l.proposed_approval==0)&(u<rate)
        check(np.array_equal(l.explored,expected_explore),key+' label-free exploration')
        check(np.array_equal(l.actual_approval,(l.proposed_approval==1)|expected_explore),key+' actual approvals')
        check(np.array_equal(l.visible,l.actual_approval),key+' mature visibility')
        check(np.array_equal(l.inclusion_probability,np.where(l.proposed_approval==1,1.,rate)),key+' propensities')
        check(l.loc[l.visible==0,'outcome'].isna().all(),key+' hidden labels absent')
        vis=l[l.visible==1];check(np.array_equal(vis.outcome,raw.loc[vis.SK_ID_CURR].TARGET),key+' visible labels')
        for group in [0,1]:
            pop=l[l.thin==group];good=pop[(pop.visible==1)&(pop.outcome==0)]
            r=estimates[(estimates.model==model)&(estimates.seed==seed)&(estimates.exploration_rate==rate)&(estimates.group==group)].iloc[0]
            close(r.naive_frr,(1-good.proposed_approval).mean(),key+' naive')
            if (pop.inclusion_probability==0).any():check(r.status=='NOT_IDENTIFIABLE' and pd.isna(r.ipw_frr),key+' nonidentifiability')
            else:
                w=1/good.inclusion_probability;close(r.ipw_frr,(w*(1-good.proposed_approval)).sum()/w.sum(),key+' independent IPW')
                close(r.ess_good,w.sum()**2/(w*w).sum(),key+' ESS')
            oracle_y=raw.loc[pop.SK_ID_CURR].TARGET.to_numpy();close(r.oracle_frr,(1-pop.proposed_approval.to_numpy())[oracle_y==0].mean(),key+' oracle')
        r=risk[(risk.model==model)&(risk.seed==seed)&(risk.exploration_rate==rate)].iloc[0]
        for name,m in [('regular',l.proposed_approval==1),('exploration',l.explored==1),('actual',l.actual_approval==1)]:
            close(r[name+'_n'],m.sum(),key+' '+name+' n')
            close(r[name+'_risk'],l.loc[m,'outcome'].mean(),key+' '+name+' risk')
    dump(out/'verification.json',dict(status='PASS',checks=len(checks),policy_rows=len(table),ledger_files=len(list((out/'ledgers').glob('*.csv.gz'))),
        scope='Independent grid optimum and record arithmetic, observed-label isolation and provenance; conditional scientific validity only'))
    print('PASS',len(checks),'independent checks',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,default=HERE/'results/run_001');args=parser.parse_args();main(args.run.resolve())
