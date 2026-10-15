"""Frozen supplementary protocol; CPU only, all policies frozen before testing."""
from pathlib import Path
import sys, json, time, hashlib
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent/'src'))
from experiment import ROOT, load_features, make_model, biased_indices, correction_weights, choose_threshold, metrics, sha, dump
import numpy as np
import pandas as pd
import joblib
from scipy.special import logit, expit
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from threadpoolctl import threadpool_limits

CFG=json.loads((HERE/'config.json').read_text())
OUT=HERE/'results'
MODELS=HERE/'models'


def sampling_conditions(y,g,rate,seed):
    biased=biased_indices(y,g,rate,seed)
    random=np.sort(np.random.default_rng(seed+10000).choice(len(y),len(biased),replace=False))
    rng=np.random.default_rng(seed+20000)
    selected=[]
    for group in [0,1]:
        n=int((g[biased]==group).sum())
        bad=int(round(n*float(y[g==group].mean())))
        for label,count in [(0,n-bad),(1,bad)]:
            cell=np.flatnonzero((g==group)&(y==label))
            selected.extend(rng.choice(cell,count,replace=False))
    group_only=np.sort(np.asarray(selected,dtype=int))
    return dict(random=random,group_only=group_only,biased=biased,corrected=biased.copy())


def loss_weights(y,g,c,lam):
    w=np.where(y==1,float(c),1.)
    for group in [0,1]:
        mask=(g==group)&(y==0)
        if not mask.any():raise ValueError('Empty normal-label group')
        w[mask]+=lam*len(y)/(2*mask.sum())
    return w/w.mean()


def predict(model,matrix,index,coef):
    raw=model.predict_proba(matrix[index])[:,1].astype(float)
    return expit(coef['slope']*logit(np.clip(raw,1e-7,1-1e-7))+coef['intercept'])


def freeze_rule(model,matrix,y,ids,cal,val,key,minimum):
    if len(np.unique(y[cal]))!=2:raise ValueError('Calibration requires two classes')
    raw=model.predict_proba(matrix[cal])[:,1].astype(float)
    lr=LogisticRegression(C=1e6,max_iter=1000).fit(logit(np.clip(raw,1e-7,1-1e-7)).reshape(-1,1),y[cal])
    coef={'slope':float(lr.coef_[0,0]),'intercept':float(lr.intercept_[0])}
    if coef['slope']<=0:raise ValueError('Calibration reversed score ordering')
    pv=predict(model,matrix,val,coef)
    rule=choose_threshold(pv,y[val],CFG['risk_budget'],minimum)
    np.savez_compressed(OUT/'policies'/f'{key}.npz',cal_ids=ids[cal],val_ids=ids[val],val_probability=pv)
    return {'key':key,'calibration':coef,'rule':rule,'cal_size':len(cal),'val_size':len(val)}


def fit(name,matrix,y,g,ids,train,weights,cal,val,minimum):
    file=MODELS/f'{name}.json'; decision=MODELS/f'{name}_decision.json'
    if file.exists() and decision.exists():
        m=make_model(CFG['parameters']);m.load_model(file)
        return m,json.loads(decision.read_text())
    assert not np.intersect1d(train,cal).size and not np.intersect1d(train,val).size
    assert not np.intersect1d(cal,val).size and len(np.unique(train))==len(train)
    w=np.asarray(weights)/np.mean(weights)
    tick=time.perf_counter()
    model=make_model(CFG['parameters']).fit(matrix[train],y[train],sample_weight=w)
    d=freeze_rule(model,matrix,y,ids,cal,val,name,minimum)
    d.update(model=name,train_size=len(train),train_cells=np.bincount(2*g[train]+y[train],minlength=4).tolist(),
             ess=float(w.sum()**2/(w@w)),max_weight=float(w.max()),seconds=time.perf_counter()-tick)
    model.save_model(file)
    np.savez_compressed(OUT/'training'/f'{name}.npz',ids=ids[train],weights=w)
    dump(decision,d)
    print(f'FIT {name}: n={len(train)} cal={len(cal)} val={len(val)} {d["seconds"]:.1f}s',flush=True)
    return model,d


def static_experiments(x,y,g,ids,cat,split):
    train,ref,cal,val=[split[k] for k in ['train','reference','calibration','validation']]
    pre=joblib.load(ROOT/'models/preprocessor.joblib')
    matrix=pre.transform(x).astype(np.float32)
    joblib.dump(pre,MODELS/'static_preprocessor.joblib')
    rows=[]
    def add(name,used,w,module,**meta):
        _,d=fit(name,matrix,y,g,ids,used,w,cal,val,CFG['static_minimum'])
        rows.append(dict(module=module,model=name,policy=d,preprocessor='static',**meta))
    add('a_full',train,np.ones(len(train)),'A',seed=0,retention=1.,condition='full')
    bases=[('full',0,train,'a_full')]
    for rate in CFG['retentions']:
        for seed in CFG['seeds']:
            selections=sampling_conditions(y[train],g[train],rate,seed)
            for condition,local in selections.items():
                used=train[local];w=np.ones(len(used))
                if condition=='corrected':w,_=correction_weights(y[used],g[used],y[ref],g[ref])
                name=f'a_r{int(rate*100)}_s{seed}_{condition}'
                add(name,used,w,'A',seed=seed,retention=rate,condition=condition)
                if rate==.5 and condition=='biased':bases.append(('biased',seed,used,name))
    for dataset,seed,used,base_name in bases:
        base=json.loads((MODELS/f'{base_name}_decision.json').read_text())
        rows.append(dict(module='B',model=base_name,policy=base,preprocessor='static',dataset=dataset,seed=seed,loss='standard'))
        for loss,(c,lam) in CFG['losses'].items():
            if loss=='standard':continue
            add(f'b_{dataset}_s{seed}_{loss}',used,loss_weights(y[used],g[used],c,lam),'B',dataset=dataset,seed=seed,loss=loss)
    dump(OUT/'static_frozen.json',rows)
    return rows


def information_sets(mode,initial,observed,roles,external):
    if mode=='frozen':return tuple(initial[roles[initial]==role] for role in [1,2])
    internal=tuple(observed[roles[observed]==role] for role in [1,2])
    if mode=='observed':return internal
    if mode!='external':raise ValueError(mode)
    if any(len(a)>len(b) for a,b in zip(internal,external)):raise ValueError('Insufficient external pool')
    return tuple(b[:len(a)] for a,b in zip(internal,external))


def feedback(x,y,g,ids,cat,split):
    refs=[];events=[];shadows=[]
    pool=split['train']
    for seed in CFG['seeds']:
        done=OUT/f'c_seed_{seed}_complete.json'
        if done.exists():
            saved=json.loads(done.read_text());refs+=saved['refs'];events+=saved['events'];shadows+=saved['shadows']
            print('Resume C seed',seed,flush=True);continue
        rng=np.random.default_rng(seed)
        order=rng.permutation(pool); n0=int(len(pool)*CFG['initial_fraction'])
        initial=order[:n0]; batches=np.array_split(order[n0:],CFG['rounds'])
        base=initial[biased_indices(y[initial],g[initial],.5,seed+500)]
        u=np.random.default_rng(seed+30000).random(len(y))
        roles=np.where(u<.8,0,np.where(u<.9,1,2))
        train=base[roles[base]==0];cal=base[roles[base]==1];val=base[roles[base]==2]
        ext=tuple(np.random.default_rng(seed+40000+j).permutation(split[k]) for j,k in enumerate(['calibration','validation']))
        pre=ColumnTransformer([('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False,dtype=np.float32),cat)],remainder='passthrough')
        pre.fit(x.iloc[train]);matrix=pre.transform(x).astype(np.float32)
        joblib.dump(pre,MODELS/f's{seed}_preprocessor.joblib')
        arrival=np.full(len(y),-1);arrival[initial]=0
        for r,batch in enumerate(batches,1):arrival[batch]=r
        pd.DataFrame({'SK_ID_CURR':ids[pool],'arrival':arrival[pool],'role':roles[pool],'initial_visible':np.isin(pool,base).astype(int)}).to_csv(OUT/f'c_stream_{seed}.csv',index=False)
        np.savez_compressed(OUT/f'c_external_{seed}.npz',cal_ids=ids[ext[0]],val_ids=ids[ext[1]])
        name=f'c_s{seed}_initial'
        model,decision=fit(name,matrix,y,g,ids,train,np.ones(len(train)),cal,val,CFG['feedback_minimum'])
        states={};sr=[];se=[];ss=[]
        for mode in CFG['modes']:
            for acq in CFG['acquisitions']:
                states[(mode,acq)]={'model':model,'decision':decision,'observed':base.copy(),'train':train.copy(),'weights':np.ones(len(train))}
                sr.append(dict(module='C',model=name,preprocessor=f's{seed}',policy=decision,seed=seed,mode=mode,acquisition=acq,round=0))
        for r,batch in enumerate(batches,1):
            uniform=np.random.default_rng(seed*100+r).random(len(batch))
            for (mode,acq),s in states.items():
                p=predict(s['model'],matrix,batch,s['decision']['calibration'])
                proposed=p<=s['decision']['rule']['threshold']
                eps=CFG['epsilon'] if acq=='explore_ipw' else 0.
                explore=(~proposed)&(uniform<eps); visible=proposed|explore
                pi=np.where(proposed,1.,eps);revealed=batch[visible]
                key=f'c_s{seed}_{mode}_{acq}_r{r}'
                pd.DataFrame({'SK_ID_CURR':ids[batch],'role':roles[batch],'probability':p,'proposed':proposed.astype(int),'explored':explore.astype(int),'visible':visible.astype(int),'pi':pi,'outcome':np.where(visible,y[batch],np.nan)}).to_csv(OUT/'ledgers'/f'{key}.csv.gz',index=False,compression='gzip')
                se.append(dict(seed=seed,round=r,mode=mode,acquisition=acq,model_before=s['decision']['model'],arrivals=len(batch),approved=int(visible.sum()),exploration=int(explore.sum()),bad=int(y[revealed].sum()),observed_cal=int((roles[revealed]==1).sum()),observed_val=int((roles[revealed]==2).sum())))
                s['observed']=np.r_[s['observed'],revealed]
                new_train=visible&(roles[batch]==0)
                s['train']=np.r_[s['train'],batch[new_train]]
                s['weights']=np.r_[s['weights'],1/pi[new_train] if acq=='explore_ipw' else np.ones(new_train.sum())]
                cal,val=information_sets(mode,base,s['observed'],roles,ext)
                s['model'],s['decision']=fit(key,matrix,y,g,ids,s['train'],s['weights'],cal,val,CFG['feedback_minimum'])
                sr.append(dict(module='C',model=key,preprocessor=f's{seed}',policy=s['decision'],seed=seed,mode=mode,acquisition=acq,round=r))
                # Same model, alternate validation information; never fed into the trajectory.
                for alt in CFG['modes']:
                    ca,va=information_sets(alt,base,s['observed'],roles,ext)
                    pol=s['decision'] if alt==mode else freeze_rule(s['model'],matrix,y,ids,ca,va,key+'_shadow_'+alt,CFG['feedback_minimum'])
                    ss.append(dict(module='C_shadow',model=key,preprocessor=f's{seed}',policy=pol,seed=seed,mode=mode,information=alt,acquisition=acq,round=r))
            print(f'C seed {seed} round {r} all six paths frozen',flush=True)
        dump(done,{'refs':sr,'events':se,'shadows':ss})
        refs+=sr;events+=se;shadows+=ss
        del matrix
    pd.DataFrame(events).to_csv(OUT/'stream_metrics.csv',index=False)
    return refs+shadows


def evaluate(refs,x,y,g,ids,split):
    test=split['test']; records=[]
    pd.DataFrame({'SK_ID_CURR':ids[test],'TARGET':y[test],'thin':g[test]}).to_csv(OUT/'test_population.csv',index=False)
    for pre_name in dict.fromkeys(r['preprocessor'] for r in refs):
        pre=joblib.load(MODELS/f'{pre_name}_preprocessor.joblib');matrix=pre.transform(x.iloc[test]).astype(np.float32)
        for model_name in dict.fromkeys(r['model'] for r in refs if r['preprocessor']==pre_name):
            model=make_model(CFG['parameters']);model.load_model(MODELS/f'{model_name}.json')
            raw=model.predict_proba(matrix)[:,1].astype(float)
            np.savez_compressed(OUT/'predictions'/f'{model_name}_raw.npz',probability=raw)
            for row in [r for r in refs if r['model']==model_name]:
                d=row['policy'];coef=d['calibration']
                p=expit(coef['slope']*logit(np.clip(raw,1e-7,1-1e-7))+coef['intercept'])
                a=p<=d['rule']['threshold']
                v=metrics(y[test],g[test],p,a)
                v.update(raw_brier=float(np.mean((raw-y[test])**2)),bad_reject=float((~a[y[test]==1]).mean()),
                         feasible=d['rule']['feasible'],threshold=d['rule']['threshold'],cal_size=d['cal_size'],val_size=d['val_size'])
                v['policy_key']=d['key'];v.update({k:z for k,z in row.items() if k not in ['policy','preprocessor']})
                records.append(v)
                np.savez_compressed(OUT/'predictions'/f'{d["key"]}.npz',probability=p,approved=a)
        print('Evaluated frozen test scores:',pre_name,flush=True)
    pd.DataFrame(records).to_csv(OUT/'metrics.csv',index=False)


def main():
    started=time.perf_counter()
    for folder in [MODELS,OUT,OUT/'training',OUT/'policies',OUT/'predictions',OUT/'ledgers',HERE/'figures']:
        folder.mkdir(parents=True,exist_ok=True)
    paths=[HERE/'PROTOCOL.md',HERE/'config.json',Path(__file__),ROOT/'src/experiment.py',ROOT/'data/processed/split_manifest.csv',ROOT/'models/preprocessor.joblib']
    paths+=list((ROOT/'data/raw').glob('*.parquet'))
    hashes={str(p.resolve()):sha(p) for p in paths}
    identity=OUT/'identity.json'
    if identity.exists() and json.loads(identity.read_text())['hashes']!=hashes:raise RuntimeError('Frozen inputs changed; use a fresh output folder')
    if not identity.exists():dump(identity,{'hashes':hashes,'started_unix':time.time()})
    x,y,g,ids,cat,audit=load_features();dump(OUT/'data_audit.json',audit)
    manifest=pd.read_csv(ROOT/'data/processed/split_manifest.csv').set_index('SK_ID_CURR')
    assignment=manifest.loc[ids,'split'].to_numpy()
    split={k:np.flatnonzero(assignment==k) for k in ['train','reference','calibration','validation','test']}
    refs=static_experiments(x,y,g,ids,cat,split)+feedback(x,y,g,ids,cat,split)
    dump(OUT/'frozen_all.json',{'frozen_unix':time.time(),'records':refs})
    assert len(set(r['model'] for r in refs))==220
    evaluate(refs,x,y,g,ids,split)
    dump(OUT/'completion.json',{'seconds':time.perf_counter()-started,'models':220,'metric_rows':len(refs),'completed_unix':time.time()})
    print('All 220 supplementary models and policies completed.',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=4):main()
