"""Show all band violations across all sections (data-only, no sims)."""
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

gate_fail, mon_flag = [], []
for k, v in sorted(vals.items()):
    e = rng.get(k, {})
    lo, hi, tier = e.get('lo'), e.get('hi'), e.get('tier', 'monitor')
    bad = (lo is not None and v < lo) or (hi is not None and v > hi)
    if bad:
        (gate_fail if tier == 'gate' else mon_flag).append((k, v, lo, hi))

print('total metrics:', len(vals))
print('\nGATE FAILURES:', len(gate_fail))
for k, v, lo, hi in gate_fail:
    print('  {:34s} = {:8.3f}  [{} .. {}]'.format(k, v, lo, hi))
print('\nMONITOR FLAGS:', len(mon_flag))
for k, v, lo, hi in mon_flag:
    print('  {:34s} = {:8.3f}  [{} .. {}]'.format(k, v, lo, hi))
