"""Post-primary matched-size random-feedback reference; explicitly diagnostic."""
import json
import time
import hashlib
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from experiment import ROOT, load_features, make_model, sha
from extensions import OUT,MOD,CONFIG,fit_and_freeze,cal_predict,score_metrics,safe_dump


def main():
    start=time.perf_counter()
    inputs={str(p.relative_to(ROOT)):sha(p) for p in [ROOT/'FEEDBACK_DIAGNOSTIC.md',ROOT/'src/feedback_diagnostic.py',OUT/'identity.json',OUT/'feedback/stream_metrics.csv']}
    signature=hashlib.sha256(json.dumps(inputs,sort_keys=True).encode()).hexdigest()
    signature_path=OUT/'feedback/diagnostic_identity.json'
    if signature_path.exists() and json.loads(signature_path.read_text())['signature']!=signature:
        raise RuntimeError('Diagnostic protocol/input changed; preserve old results first.')
    safe_dump(signature_path,{'signature':signature,'hashes':inputs,'post_primary_diagnostic':True})
    x,y,g,ids,cat,_=load_features()
    location=pd.Series(np.arange(len(ids)),index=ids)
    manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv').set_index('SK_ID_CURR')
    assignment=manifest.loc[ids,'split'].to_numpy()
    cal=np.flatnonzero(assignment=='calibration');val=np.flatnonzero(assignment=='validation');test=np.flatnonzero(assignment=='test')
    events=pd.read_csv(OUT/'feedback/stream_metrics.csv')
    references=[]
    for seed in CONFIG['feedback']['seeds']:
        pre=joblib.load(MOD/f'seed_{seed}_preprocessor.joblib')
        matrix=pre.transform(x).astype(np.float32)
        arrivals=pd.read_csv(OUT/f'feedback/stream_{seed}.csv')
        with np.load(OUT/f'feedback/training/s{seed}_initial.npz') as f:observed=location.loc[f['ids']].to_numpy()
        references.append({'seed':seed,'round':0,'branch':'random_reference','model':f's{seed}_initial'})
        for r in range(1,6):
            batch_ids=arrivals.loc[arrivals.arrival_round==r,'SK_ID_CURR'].to_numpy()
            count=int(events[(events.seed==seed)&(events.branch=='selected_only')&(events['round']==r)].actual_approved.iloc[0])
            chosen=np.random.default_rng(seed*1000+r+70000).choice(batch_ids,count,replace=False)
            observed=np.r_[observed,location.loc[chosen].to_numpy()]
            name=f's{seed}_random_reference_r{r}'
            if not (MOD/f'{name}_decision.json').exists():fit_and_freeze(name,matrix,observed,y,g,np.ones(len(observed)),cal,val,ids)
            references.append({'seed':seed,'round':r,'branch':'random_reference','model':name})
        print(f'Matched-size diagnostic seed {seed} frozen.',flush=True)
    safe_dump(OUT/'feedback/diagnostic_frozen.json',references)
    results=[]
    for seed in CONFIG['feedback']['seeds']:
        pre=joblib.load(MOD/f'seed_{seed}_preprocessor.joblib'); xt=pre.transform(x.iloc[test]).astype(np.float32)
        for ref in [v for v in references if v['seed']==seed]:
            name=ref['model'];d=json.loads((MOD/f'{name}_decision.json').read_text())
            model=make_model(CONFIG['feedback']['parameters']);model.load_model(MOD/f'{name}.json')
            p=cal_predict(model,xt,d['calibration']);a=p<=d['rule']['threshold']
            # Initial predictions already exist from main experiment; preserve timestamps.
            if ref['round']:np.savez_compressed(OUT/f'feedback/predictions/{name}.npz',probability=p,approved=a)
            results.append({**ref,**score_metrics(y[test],g[test],p,a),**{k:d[k] for k in ['train_size','thin_good_share','weighted_thin_good_share','effective_sample_size','max_normalized_weight']}})
    pd.DataFrame(results).to_csv(OUT/'feedback/diagnostic_metrics.csv',index=False)
    safe_dump(OUT/'feedback/diagnostic_completion.json',{'additional_fits':25,'seconds':time.perf_counter()-start,'signature':signature,'completed_unix':time.time()})
    print('Diagnostic reference: 25 additional models, 30 round evaluations complete.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=CONFIG['threads']):main()
