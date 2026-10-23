"""Shared historical split and train-only preprocessing for every V3 predictor."""
from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from .common import HERE, ROOT, dump, sha, now, load_config

def prepare(config=None):
    cfg = config or load_config()
    cache = HERE / 'cache'
    manifest = cache / 'prepared.json'
    if manifest.exists():
        info = json.loads(manifest.read_text(encoding='utf-8'))
        if info['split_seed'] != cfg['split_seed']:
            raise ValueError('Cached split seed differs')
        if info['legacy_feature_loader_sha256'] != sha(ROOT/'src/experiment.py'):
            raise ValueError('Feature loader changed; prepare a fresh cache/output')
        for name, source in info['audit']['source_files'].items():
            if sha(ROOT/'data/raw'/name) != source['sha256']:
                raise ValueError('Raw data changed: '+name)
        for name, checksum in info['cache_hashes'].items():
            if sha(cache/name) != checksum: raise ValueError('Cache hash mismatch: '+name)
        cached_records = pd.read_parquet(cache/'records.parquet', columns=['AMT_INCOME_TOTAL'])
        cached_train = np.load(cache/'splits.npz')['train']
        income = cached_records.AMT_INCOME_TOTAL.to_numpy()[cached_train]
        boundaries = np.quantile(income[np.isfinite(income)], cfg['income_quantiles'])
        if not np.array_equal(boundaries, info['income_boundaries']):
            raise ValueError('Income subgroup definition changed; use a fresh cache/output')
        return cache, info
    from src.experiment import load_features, split_indices
    print('Preparing full application + time-filtered bureau features', flush=True)
    x, y, g, ids, cat, audit = load_features()
    split = split_indices(y, g, cfg['split_seed'])
    cache.mkdir(parents=True, exist_ok=True)
    pre = ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float32), cat)],
                            remainder='passthrough', verbose_feature_names_out=False)
    pre.fit(x.iloc[split['train']])
    matrix = pre.transform(x).astype(np.float32)
    np.save(cache/'native.npy', matrix)
    train = split['train']
    missing_cols = np.flatnonzero(np.isnan(matrix[train]).any(axis=0))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        medians = np.nanmedian(matrix[train], axis=0)
    medians = np.nan_to_num(medians, nan=0.0)
    dense = np.where(np.isnan(matrix), medians, matrix)
    if len(missing_cols):
        dense = np.concatenate([dense, np.isnan(matrix[:, missing_cols]).astype(np.float32)], axis=1)
    np.save(cache/'dense.npy', dense.astype(np.float32))
    np.savez(cache/'splits.npz', **split)
    np.savez(cache/'imputation.npz', medians=medians, missing_columns=missing_cols)
    joblib.dump(pre, cache/'preprocessor.joblib')
    income = x.AMT_INCOME_TOTAL.to_numpy(dtype=float)
    bounds = np.quantile(income[train][np.isfinite(income[train])], cfg['income_quantiles'])
    records = pd.DataFrame(dict(SK_ID_CURR=ids, TARGET=y, thin=g,
                                AMT_CREDIT=x.AMT_CREDIT.to_numpy(dtype=float),
                                AMT_INCOME_TOTAL=income,
                                income_band=np.searchsorted(bounds, income)))
    if not records.AMT_CREDIT.gt(0).all(): raise ValueError('Nonpositive principal')
    assignment = np.empty(len(records), dtype='U16')
    for name, index in split.items(): assignment[index] = name
    records['split'] = assignment
    records.to_parquet(cache/'records.parquet', index=False)
    records[['SK_ID_CURR','TARGET','thin','split']].to_parquet(cache/'split_manifest.parquet', index=False)
    info = dict(status='PASS', created_utc=now(), split_seed=cfg['split_seed'],
                audit=audit, n_rows=len(records), native_shape=list(matrix.shape), dense_shape=list(dense.shape),
                income_boundaries=bounds.tolist(), split_counts={k: len(v) for k,v in split.items()},
                preprocessing='Training-only OHE; native missing values for trees. Median imputation and training-missing indicators for LR/TabM; adapter fits further transforms on train only.',
                legacy_feature_loader_sha256=sha(ROOT/'src/experiment.py'), prepare_code_sha256=sha(__file__),
                feature_names=list(pre.get_feature_names_out()),
                cache_hashes={name: sha(cache/name) for name in ['native.npy','dense.npy','records.parquet','splits.npz','split_manifest.parquet']})
    dump(manifest, info)
    print(f'Prepared {len(records):,} records; native {matrix.shape}; dense {dense.shape}', flush=True)
    return cache, info

def load_prepared(config=None):
    cache, info = prepare(config)
    return (np.load(cache/'native.npy', mmap_mode='r'), np.load(cache/'dense.npy', mmap_mode='r'),
            pd.read_parquet(cache/'records.parquet'), dict(np.load(cache/'splits.npz')), info)

if __name__ == '__main__':
    prepare()
