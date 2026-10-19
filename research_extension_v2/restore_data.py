"""Use legacy fixed URLs and hashes, without rewriting its source manifest."""
from urllib.request import urlopen, Request
import shutil
from research_extension_v2.common import ROOT, sha, dump
from prepare_data import FILES


def main():
    evidence = {}
    for name, (url, expected) in FILES.items():
        path = ROOT / 'data/raw' / name
        if not path.exists():
            temp = path.with_suffix(path.suffix + '.part')
            with urlopen(Request(url, headers={'User-Agent': 'CourseResearch/2.0'}), timeout=120) as response, temp.open('wb') as f:
                shutil.copyfileobj(response, f)
            if sha(temp) != expected: raise ValueError('Hash mismatch: ' + name)
            temp.rename(path)
        if sha(path) != expected: raise ValueError('Existing input differs: ' + name)
        evidence[name] = dict(url=url, sha256=expected, bytes=path.stat().st_size)
        print('VERIFIED', name, flush=True)
    dump(ROOT / 'research_extension_v2/results/data_recovery.json', evidence)


if __name__ == '__main__': main()
