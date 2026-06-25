import json
import sys

prefix = sys.argv[1] if len(sys.argv) > 1 else 'pursuit'
base = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax'
d = json.load(open(base + r'\web\benchmarks\benchmarks_data.json', encoding='utf-8'))
g = json.load(open(base + r'\src\oculomotor\benchmarks\golden_metrics.json', encoding='utf-8'))
rng = json.load(open(base + r'\web\benchmarks\metrics_ranges.json', encoding='utf-8'))
rows = {}
def w(o):
    if isinstance(o, dict):
        if 'name' in o and isinstance(o.get('value'), (int, float)):
            rows[o['name']] = o['value']
        for v in o.values():
            w(v)
    elif isinstance(o, list):
        [w(v) for v in o]
w(d)
for n in sorted(rows):
    if n.startswith(prefix):
        e = rng.get(n, {}); lo, hi, v = e.get('lo'), e.get('hi'), rows[n]
        bad = 'RED' if (lo is not None and v < lo) or (hi is not None and v > hi) else ''
        gold = g.get(n)
        gs = '{:8.3f}'.format(gold) if isinstance(gold, (int, float)) else '   --   '
        print('  {:30s} now={:8.3f}  golden={}  [{} .. {}] {}'.format(n, v, gs, lo, hi, bad))
