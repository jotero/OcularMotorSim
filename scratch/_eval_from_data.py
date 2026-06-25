"""Evaluate metrics straight from benchmarks_data.json + metrics_ranges.json +
golden (NO bench re-run). Confirms the new PASS/FAIL/DRIFT/NEW relabel."""
import sys
from oculomotor.benchmarks import bench_metrics as bm

data   = bm.load_benchmarks_data()
ranges = bm.load_ranges()
golden = bm.load_golden()
want   = sys.argv[1] if len(sys.argv) > 1 else None   # optional section-id substring

rows = []
for sec in data.get('sections', []):
    if want and want not in sec.get('id', ''):
        continue
    for fig in sec.get('figures', []):
        for rec in fig.get('metrics', []):
            rows.append((sec.get('id', ''), bm.metric_from_record(rec, ranges)))

results = bm.evaluate([m for _, m in rows], golden)
tally = {}
for (sec, _), r in zip(rows, results):
    tally[r.status] = tally.get(r.status, 0) + 1
    if want or r.status in ('fail', 'drift', 'new'):
        g = '' if r.golden is None else f'{r.golden:.3f}'
        drift = '' if r.drift is None else f'{r.drift*100:+.1f}%'
        print(f'  {bm._MARK[r.status]:6} {r.metric.name:34} '
              f'val={r.metric.value:9.3f}  golden={g:>9}  drift={drift:>8}  [{sec}]')
print('\nsummary:', tally)
