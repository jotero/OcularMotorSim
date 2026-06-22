"""Compare freshly-computed benchmark metrics (benchmarks_data.json) vs the frozen
golden, to verify near-response improved and non-vergence sections are unchanged."""
import json, os
ROOT = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax'
data = json.load(open(os.path.join(ROOT, 'web', 'benchmarks', 'benchmarks_data.json'), encoding='utf-8'))
golden = json.load(open(os.path.join(ROOT, 'src', 'oculomotor', 'benchmarks', 'golden_metrics.json'), encoding='utf-8'))

# flatten metrics from data.json: structure is list of sections each with figures w/ metrics
cur = {}
def walk(o):
    if isinstance(o, dict):
        if 'name' in o and 'value' in o and isinstance(o.get('value'), (int, float)):
            cur[o['name']] = o['value']
        for v in o.values(): walk(v)
    elif isinstance(o, list):
        for v in o: walk(v)
walk(data)

rows = []
for name, gval in golden.items():
    cval = cur.get(name)
    if not isinstance(gval, (int, float)):
        continue
    if cval is None:
        rows.append((name, gval, None, None)); continue
    if gval == 0:
        pct = 0.0 if cval == 0 else 999.0
    else:
        pct = abs(cval - gval) / abs(gval) * 100
    rows.append((name, gval, cval, pct))

print("=== CHANGED metrics (|drift| > 2%) ===")
for name, g, c, pct in sorted([r for r in rows if r[3] is not None and r[3] > 2.0], key=lambda r:-r[3]):
    print(f"  {name:34s} {g:10.3f} -> {c:10.3f}   ({pct:6.1f}%)")
print("\n=== metrics in golden but MISSING from new data ===")
for name, g, c, pct in rows:
    if c is None: print(f"  {name}")
new_only = set(cur) - set(golden)
print("\n=== NEW metrics not in golden ===")
for name in sorted(new_only): print(f"  {name:34s} = {cur[name]:.3f}")
print(f"\nTotal golden metrics: {len(golden)}, current: {len(cur)}, unchanged(<2%): {sum(1 for r in rows if r[3] is not None and r[3] <= 2.0)}")
