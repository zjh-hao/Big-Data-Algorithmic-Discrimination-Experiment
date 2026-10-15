"""Fetch fixed public GitHub dataset mirrors, with expected content hashes."""
from pathlib import Path
from urllib.request import Request, urlopen
import hashlib
import json
import shutil
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
BASE = 'https://raw.githubusercontent.com/qwyt/HomeCreditDefaultRisk/920bbfca12ac236a10ca6b9772286fa15f7a9cca'
FILES = {
    'app_train.parquet': (BASE+'/dataset/full/app_train.parquet', '91cc5fd191a67fef84edf4c99318ea52b53966ebc02591152a31e1e68324649e'),
    'bureau.parquet': (BASE+'/dataset/full/bureau.parquet', 'ec8694dac827b2ab8202398ba9ce48cf0aa9fbc98b2c472d417a4d7b096b0629'),
    'HomeCredit_columns_description.csv': ('https://raw.githubusercontent.com/tejase13/Home-Credit-Default-Risk/d547460fce1d5345cb1cd4b00b9b99fb8cbcb11c/HomeCredit_columns_description.csv', 'eef7665398228a80f7367c9258220c5fbe1038f3f54094244f354d54e2d4fb03')
}


def main():
    raw = ROOT/'data/raw'
    raw.mkdir(parents=True, exist_ok=True)
    manifest = {'original_source':'https://www.kaggle.com/competitions/home-credit-default-risk/data',
                'retrieval':'Third-party public GitHub mirrors, not an official Home Credit distribution',
                'conversion_source':BASE+'/src_draft/load_dataset.py',
                'verified_utc':datetime.now(timezone.utc).isoformat(),
                'limitations':'Original Kaggle byte hashes are unavailable. Structural checks do not prove exact equivalence. Mirror repository licenses are not asserted to relicense underlying data.',
                'files':{}}
    for name, (url, expected) in FILES.items():
        path = raw/name
        if not path.exists():
            temp = path.with_suffix(path.suffix+'.part')
            with urlopen(Request(url, headers={'User-Agent':'CourseResearch/1.0'}), timeout=120) as r, temp.open('wb') as f:
                shutil.copyfileobj(r,f)
            if hashlib.sha256(temp.read_bytes()).hexdigest() != expected: raise ValueError('Hash mismatch: '+name)
            temp.replace(path)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected: raise ValueError('Existing file differs: '+name)
        manifest['files'][name] = {'url':url,'sha256':actual,'bytes':path.stat().st_size}
        print(name, 'verified')
    (raw/'source.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    refs = ROOT/'references'
    refs.mkdir(exist_ok=True)
    for name, url in [('mirror_conversion.py.txt',BASE+'/src_draft/load_dataset.py'),
                      ('aws_bias_mitigation_readme.md','https://raw.githubusercontent.com/aws-samples/amazon-sagemaker-credit-risk-prediction-explainability-bias-detection/main/bias-mitigation/README.md')]:
        with urlopen(url,timeout=60) as r: (refs/name).write_bytes(r.read())


if __name__ == '__main__': main()
