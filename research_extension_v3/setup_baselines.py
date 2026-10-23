"""Restore pinned official sources without replacing an existing checkout.

The experiment adapters read these source trees directly. Git commits and
critical source hashes are recorded in upstream_sources.json. By default only
the three required runtime repositories are fetched; --all includes references.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = Path(os.environ.get('CREDIT_FAIRNESS_BASELINES', str(HERE.parent.parent/'credit-fairness-baselines'))).expanduser().resolve()


def verify_files(target, source):
    for relative, expected in source['verification_files'].items():
        path = target/relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Pinned upstream source differs: {source["name"]}/{relative}; existing files were not replaced')


def git(target, *arguments):
    return subprocess.check_output(['git','-C',str(target),*arguments],text=True,stderr=subprocess.STDOUT).strip()


def prepare_source(root, source, check_only=False):
    target=root/source['name']
    if target.exists():
        if (target/'.git').exists() and git(target,'rev-parse','HEAD') != source['commit']:
            raise ValueError(f'Existing checkout has another commit: {target}; no reset performed')
        verify_files(target,source)
        return 'VERIFIED_EXISTING'
    if check_only:
        raise FileNotFoundError(f'Missing pinned source: {target}')
    target.mkdir(parents=True)
    git(target,'init','--quiet')
    git(target,'config','core.autocrlf','false')
    git(target,'config','core.eol','lf')
    git(target,'remote','add','origin',source['repository']+'.git')
    git(target,'fetch','--quiet','--depth','1','origin',source['commit'])
    git(target,'checkout','--quiet','--detach',source['commit'])
    verify_files(target,source)
    return 'FETCHED_AND_VERIFIED'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--all',action='store_true',help='Also restore the two optional reference repositories')
    parser.add_argument('--check-only',action='store_true',help='Only verify existing sources; no network or file writes')
    args=parser.parse_args()
    root=args.root.expanduser().resolve()
    manifest=json.loads((HERE/'upstream_sources.json').read_text(encoding='utf-8'))
    if not args.check_only:
        root.mkdir(parents=True,exist_ok=True)
    for source in manifest['sources']:
        if source['runtime_required'] or args.all:
            status=prepare_source(root,source,args.check_only)
            print(json.dumps(dict(name=source['name'],status=status,commit=source['commit'])),flush=True)
    # Older adapters read this small manifest. Preserve the original downloaded
    # snapshot provenance when it already exists at the destination.
    if not args.check_only and not (root/'sources.json').exists():
        output={'sources':[{**s,'local_path':str(root/s['name']),
                           'status':'PINNED_SOURCE_READY' if (root/s['name']).exists() else 'OPTIONAL_NOT_FETCHED'}
                          for s in manifest['sources']]}
        (root/'sources.json').write_text(json.dumps(output,indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    main()
