"""Loosen the noisy post-saccadic peak bands to the noise floor (the noisy 1°
column is noise-dominated, not an EC signal) and report the seeded entries."""
import json

P = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\web\benchmarks\metrics_ranges.json'
r = json.load(open(P, encoding='utf-8'))

# Noisy peaks reflect the fixational-noise floor (~2.4 deg/s), not EC residual.
if 'sac_postsac_peak_long_noisy' in r:
    r['sac_postsac_peak_long_noisy']['hi'] = 3.0

json.dump(r, open(P, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)

for k in sorted(r):
    if 'postsac_peak' in k:
        e = r[k]
        print('  {:34s} [{} .. {}]  {}'.format(k, e.get('lo'), e.get('hi'), e.get('tier')))
