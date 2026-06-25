"""Prune metrics_ranges.json to only the metric names currently emitted (in
benchmarks_data.json). Renames in this batch (tier removal, Bode unification,
saccade metric trims) orphan the old names; golden is auto-pruned by
`bench_metrics --update` (it rebuilds from data.json), but ranges are seed-only,
so stale band entries linger unless removed here. Prints the orphans first.
"""
import json

base = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\web\benchmarks'
data = json.load(open(base + r'\benchmarks_data.json', encoding='utf-8'))
rng = json.load(open(base + r'\metrics_ranges.json', encoding='utf-8'))

names = set()
for sec in data.get('sections', []):
    for fig in sec.get('figures', []):
        for m in fig.get('metrics', []):
            names.add(m['name'])

orphans = sorted(k for k in rng if k not in names)
print(f'data.json metric names: {len(names)}')
print(f'ranges.json entries:    {len(rng)}')
print(f'orphans to prune:       {len(orphans)}')
for k in orphans:
    print('   -', k)

for k in orphans:
    del rng[k]
json.dump(rng, open(base + r'\metrics_ranges.json', 'w', encoding='utf-8'),
          indent=2, ensure_ascii=False)
print(f'remaining ranges entries: {len(rng)}')
