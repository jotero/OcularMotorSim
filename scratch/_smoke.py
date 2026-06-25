"""Smoke-test the batch of benchmark edits before the slow full run."""
import matplotlib; matplotlib.use('Agg')
import importlib

# 1. bode self-test (expected_bode + overlay must not break it)
import oculomotor.benchmarks.bode as bode
ef = bode.expected_bode('lowpass', fc=2.0, label='x'); assert set(ef) == {'f','gain','phase','label'}
ef = bode.expected_bode('highpass', fc=0.3); ef = bode.expected_bode('flat', g0=1.0)
print('bode.expected_bode OK')

# 2. VOR/OKR: Raphan returns (fm, sims); cascade consumes sims (3 cols, no re-run)
from oculomotor.benchmarks import bench_vor_okr as vo
fm_raphan, sims = vo._raphan(False)
print('raphan OK -> sims keys:', sorted(sims))
fm_casc = vo._cascade(sims, False)
print('cascade OK:', fm_casc['title'])
# Bode with expected overlay
fmv = vo._vor_bode(False); print('vor_bode OK,', len(fmv['metrics']), 'metrics')

# 3. gravity run includes tvor (merged section)
from oculomotor.benchmarks import bench_gravity as gr
figs = gr.run(show=False, only='ocr')   # quick: one gravity fig (no tvor when only=)
print('gravity only=ocr OK,', len(figs), 'fig')
from oculomotor.benchmarks import bench_tvor as tv
fmt = tv._bode(False); print('tvor _bode OK, metrics:', [m.name for m in fmt['metrics']][:3], '...')

# 4. a couple of the other Bode overlays
from oculomotor.benchmarks import bench_pursuit as pu
print('pursuit_bode OK,', len(pu._bode(False)['metrics']), 'metrics')
print('\nALL SMOKE TESTS PASSED')
