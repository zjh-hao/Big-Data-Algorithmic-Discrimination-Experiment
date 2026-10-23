"""Small, reference-only searches; candidate definitions precede test evaluation."""
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .common import dump, now, sha, HERE
from .advanced_models import train_tabm, train_m2fgb

def selection_fingerprint(name,cfg,run):
    value=dict(name=name, search_definition='v3_reference_candidates_20261022_v1',
               config=cfg, data_audit_sha256=sha(Path(run)/'data_audit.json'),
               adapter_sha256=sha(HERE/'advanced_models.py'))
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()

def _load_tabm(artifact):
    import torch
    import joblib
    from .advanced_models import _load_tabm as load_official
    artifact=Path(artifact)
    pre=joblib.load(artifact/'preprocessor.joblib')
    data=torch.load(artifact/'best_checkpoint.pt',map_location='cpu',weights_only=True)
    official,_=load_official()
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=official.TabM.make(**data['model_params']).to(device)
    model.load_state_dict(data['state_dict']);model.eval()
    def predict(X):
        z=np.array(X,dtype=np.float32,copy=True)
        cols=pre['continuous_columns']
        if pre['normalization']=='quantile':
            if cols.any(): z[:,cols]=pre['transformer'].transform(z[:,cols])
        else:
            z=pre['transformer'].transform(z).astype(np.float32)
            z=np.clip(z,-pre['clip'],pre['clip'])
        parts=[]
        with torch.inference_mode():
            for start in range(0,len(z),2048):
                value=torch.as_tensor(z[start:start+2048],device=device)
                parts.append(model(value).squeeze(-1).sigmoid().mean(dim=1).cpu().numpy())
        return np.concatenate(parts)
    meta=json.loads((artifact/'metadata.json').read_text(encoding='utf-8'))
    return model,predict,meta

def train_advanced(name,X,y,Xstop,ystop,cfg,seed,artifact,**kwargs):
    artifact=Path(artifact)
    tuning=artifact.parents[1]/'tuning'
    tuning.mkdir(exist_ok=True)
    selected_path=tuning/f'{name}_selection.json'
    selected=None
    if selected_path.exists():
        selected=json.loads(selected_path.read_text(encoding='utf-8'))
        if selected.get('input_fingerprint') != selection_fingerprint(name,cfg,artifact.parents[1]):
            raise ValueError('Reference-selection data/config/adapter differs; use a fresh output directory')
    first_seed=int(cfg['seeds'][0])
    if selected is None and seed!=first_seed:
        raise ValueError('Run first configured seed before reusing reference-selected parameters')
    if name=='tabm':
        import torch
        base=copy.deepcopy(cfg['tabm'])
        candidates=[base,{**base,'n_blocks':3,'d_block':128,'learning_rate':0.002},
                    {**base,'n_blocks':3,'d_block':192,'learning_rate':0.001}]
        if selected is not None:
            candidates=[selected['parameters']]
        started=time.perf_counter();trials=[]
        for index,parameters in enumerate(candidates):
            trial=artifact/f'trial_{index:02d}'
            print(f'TabM reference-only candidate {index+1}/{len(candidates)}: {parameters}',flush=True)
            model,predict,meta=train_tabm(X,y,Xstop,ystop,{**cfg,'tabm':parameters},seed,trial)
            trials.append(dict(parameters=parameters,reference_auc=meta['best_stop_auc'],
                               artifact=str(trial),training_seconds=meta['training_seconds']))
            del model,predict
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        winner=max(trials,key=lambda row:row['reference_auc'])
        if selected is None:
            selected=dict(model=name,selection_split='reference',selection_metric='ROC AUC',
                          seed=seed,parameters=winner['parameters'],trials=trials,frozen_utc=now(),
                          input_fingerprint=selection_fingerprint(name,cfg,artifact.parents[1]))
            dump(selected_path,selected)
        model,predict,meta=_load_tabm(winner['artifact'])
        meta.update(reference_auc=winner['reference_auc'],train_seconds=time.perf_counter()-started,
                    input_features=X.shape[1],reference_search=trials,selected_configuration=selected,
                    selected_artifact=winner['artifact'])
        dump(artifact/'search_metadata.json',meta)
        return model,predict,meta
    if name=='m2fgb_tuned':
        base=copy.deepcopy(cfg['m2fgb'])
        candidates=[{**base,'fair_weight':weight} for weight in [.05,.1,.5]]
        if selected is not None: candidates=[selected['parameters']]
        started=time.perf_counter();trials=[];objects=[]
        for index,parameters in enumerate(candidates):
            trial=artifact/f'trial_{index:02d}'
            model,predict,meta=train_m2fgb(X,y,Xstop,ystop,{**cfg,'m2fgb':parameters},seed,trial,**kwargs)
            trials.append(dict(parameters=parameters,reference_auc=meta['stop_auc'],artifact=str(trial)))
            objects.append((model,predict,meta))
        index=max(range(len(trials)),key=lambda j:trials[j]['reference_auc'])
        winner=trials[index];model,predict,meta=objects[index]
        if selected is None:
            selected=dict(model=name,selection_split='reference',selection_metric='ROC AUC',seed=seed,
                          parameters=winner['parameters'],trials=trials,frozen_utc=now(),
                          input_fingerprint=selection_fingerprint(name,cfg,artifact.parents[1]))
            dump(selected_path,selected)
        meta.update(reference_auc=winner['reference_auc'],train_seconds=time.perf_counter()-started,
                    input_features=X.shape[1],reference_search=trials,selected_configuration=selected)
        dump(artifact/'search_metadata.json',meta)
        return model,predict,meta
    return train_m2fgb(X,y,Xstop,ystop,cfg,seed,artifact,**kwargs)
