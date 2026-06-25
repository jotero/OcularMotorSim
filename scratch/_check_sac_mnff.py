import json
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
print('{:36s} {:>9} {:>11} {:>14}'.format('metric', 'now(1.0)', 'golden(1.5)', 'band'))
for n in sorted(rows):
    if n.startswith('sac_'):
        e = rng.get(n, {})
        band = '[{} .. {}]'.format(e.get('lo'), e.get('hi'))
        flag = ''
        v = rows[n]
        if (e.get('lo') is not None and v < e['lo']) or (e.get('hi') is not None and v > e['hi']):
            flag = ' RED'
        print('  {:34s} {:9.3f} {:11.3f}  {:>14}{}'.format(n, v, g.get(n, float('nan')), band, flag))
