from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / 'src'))
CFG = json.loads((HERE / 'config.json').read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1048576), b''): h.update(block)
    return h.hexdigest()


def dump(path, obj):
    def clean(v):
        if isinstance(v, dict): return {str(k): clean(z) for k, z in v.items()}
        if isinstance(v, (list, tuple)): return [clean(z) for z in v]
        if isinstance(v, np.ndarray): return clean(v.tolist())
        if isinstance(v, np.generic): return clean(v.item())
        if isinstance(v, float) and not np.isfinite(v): return None
        if isinstance(v, Path): return str(v)
        return v
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(clean(obj), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def validate(frame, score=True):
    required = ['SK_ID_CURR', 'TARGET', 'thin'] + (['probability'] if score else [])
    if set(required) - set(frame): raise ValueError('Missing fields')
    if frame[required].isna().any().any(): raise ValueError('Missing required values')
    if not frame.SK_ID_CURR.is_unique: raise ValueError('Duplicate record ID')
    if not frame.TARGET.isin([0, 1]).all() or not frame.thin.isin([0, 1]).all(): raise ValueError('Invalid labels/groups')
    if score and not frame.probability.between(0, 1).all(): raise ValueError('Invalid score')


def metrics(y, g, p, a):
    y, g, p, a = np.asarray(y), np.asarray(g), np.asarray(p), np.asarray(a, dtype=bool)
    result = dict(n=len(y), approved_count=int(a.sum()), approval_rate=float(a.mean()),
                  approved_risk=float(y[a].mean()) if a.any() else None,
                  auc=float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None)
    for group, name in [(0, 'thick'), (1, 'thin')]:
        m = (g == group) & (y == 0)
        result[name + '_frr'] = float((~a[m]).mean()) if m.any() else None
    result['abs_frr_gap'] = (abs(result['thin_frr'] - result['thick_frr'])
                             if all(result[k + '_frr'] is not None for k in ['thin', 'thick']) else None)
    return result


def read_scores(path):
    f = pd.read_csv(path, float_precision='round_trip')
    validate(f)
    return f
