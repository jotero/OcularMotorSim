"""Show pass/fail for gravity/tVOR metrics against the (new) bands."""
import json

base = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\web\benchmarks'
data = json.load(open(base + r'\benchmarks_data.json', encoding='utf-8'))
rng = json.load(open(base + r'\metrics_ranges.json', encoding='utf-8'))

vals = {}
def walk(o):
    if isinstance(o, dict):
        if 'name' in o and isinstance(o.get('value'), (int, float)):
            vals[o['name']] = o['value']
        for v in o.values():
            walk(v)
    elif isinstance(o, list):
        [walk(v) for v in o]
walk(data)

keys = [k for k in rng if any(s in k for s in ['ovar', 'tilt', 'ocr', 'tvor'])]
for k in sorted(keys):
    v = vals.get(k)
    lo = rng[k].get('lo')
    hi = rng[k].get('hi')
    if v is None:
        continue
    bad = (lo is not None and v < lo) or (hi is not None and v > hi)
    flag = 'FLAG' if bad else 'ok  '
    print('  {} {:32s} = {:8.3f}   [{} .. {}]'.format(flag, k, v, lo, hi))
