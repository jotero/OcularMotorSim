"""Diagnostic: realized inter-saccadic intervals in the double-step bench."""
import numpy as np
import jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, _saccade_onset_times, THETA, DT
from oculomotor.analysis import extract_burst

AMPS = [10, 20, 30, 40]
isis = [0.02, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50]
T_end = 1.2
t1 = 0.15
t_np = np.arange(0.0, T_end, DT)
T = len(t_np)

print(f'{"A":>3} {"cmdISI":>7} {"#sac":>5} {"onsets (ms after t1)":>26} {"realized ISI (ms)":>18}')
all_real = []
for ci, A in enumerate(AMPS):
    A1 = A / 2.0
    A2 = float(A)
    for ri, isi in enumerate(isis):
        t2 = t1 + isi
        tgt = np.where(t_np < t1, 0.0, np.where(t_np < t2, A1, A2)).astype(np.float32)
        pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
        pt3[:, 0] = np.tan(np.radians(tgt))
        st = _run(t_np, jnp.array(pt3), key=ri * len(AMPS) + ci)
        bst = extract_burst(st, THETA)[:, 0]
        ons = _saccade_onset_times(bst, t1)
        rel = list(np.diff(ons) * 1000.0) if len(ons) >= 2 else []
        all_real.extend(rel)
        ons_ms = [round((o - t1) * 1000) for o in ons]
        print(f'{A:>3} {isi*1000:>7.0f} {len(ons):>5} {str(ons_ms):>26} {str([round(x) for x in rel]):>18}')

all_real = np.array(all_real)
print(f'\nrealized ISI over {len(all_real)} gaps:  min={all_real.min():.1f}  '
      f'median={np.median(all_real):.1f}  max={all_real.max():.1f}')
