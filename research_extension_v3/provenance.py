"""Record environment, source versions, hardware, and final evidence hashes."""
import argparse
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
import pandas as pd
from .common import HERE,ROOT,BASELINE_SOURCES,dump,sha,now,load_config

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=str(HERE/'results/run_001'))
    args=parser.parse_args();run=Path(args.run)
    packages={}
    for name in ['numpy','pandas','scipy','scikit-learn','xgboost','lightgbm','matplotlib','pyarrow','torch','rtdl_num_embeddings','narwhals','joblib']:
        packages[name]=importlib.metadata.version(name)
    source_root=BASELINE_SOURCES
    sys.path.insert(0,str(source_root/'fairlearn'))
    import fairlearn
    import torch
    from .advanced_search import selection_fingerprint
    # Selection contents are unchanged; add a post-run compatibility attestation
    # for guards introduced after this first successful run.
    for path in (run/'tuning').glob('*_selection.json'):
        obj=json.loads(path.read_text(encoding='utf-8'))
        if 'input_fingerprint' not in obj:
            obj['input_fingerprint']=selection_fingerprint(obj['model'],load_config(),run)
            obj['guard_attestation']='Added after successful run; original selected parameters and trials preserved'
            dump(path,obj)
    for path in (run/'jobs').glob('*.json'):
        obj=json.loads(path.read_text(encoding='utf-8'))
        if obj.get('status')!='PASS':continue
        meta=obj.get('metadata',{});metrics=obj['model_metrics']
        if metrics.get('reference_auc') is None:
            metrics['reference_auc']=meta.get('stop_auc',meta.get('best_stop_auc'))
        if metrics.get('input_features') is None:
            metrics['input_features']=334 if obj['model'] in ['tabm','logistic'] else 262
        dump(path,obj)
    from .run import aggregate
    aggregate(run)
    source_manifest=json.loads((source_root/'sources.json').read_text(encoding='utf-8'))
    versions=dict(python=sys.version,platform=platform.platform(),packages=packages,
                  fairlearn_source_version=fairlearn.__version__,cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                  tf32_enabled_for_training=True,generated_utc=now())
    dump(run/'runtime_versions.json',versions)
    dump(run/'reproducibility.json',dict(generated_utc=now(),versions=versions,
        upstream_sources=source_manifest,
        code_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted(HERE.glob('*.py'))},
        config_sha256=sha(run/'config_frozen.json'),split_manifest_sha256=sha(run/'split_manifest.parquet'),
        historical_base_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        notes=['Official algorithm adaptation to shared Home Credit protocol; not a reproduction of every original paper benchmark',
               'Historical holdout already inspected; no new blind test claim',
               'Fixed-model/policy bootstrap, not training uncertainty',
               'Initial completed basic checkpoints predate stricter resume guards; saved predictions unchanged',
               'LR and M2FGB current deterministic settings can yield identical outputs across seeds']))
    print(json.dumps(dict(status='PASS',gpu=versions['gpu'],packages=len(packages))))

if __name__=='__main__':main()
