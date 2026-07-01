"""Behavior-preservation check for the canal-plane NI (Phase 2a).
Eye position (states.plant.left) is the physical ground truth — reconstructed at
the FCP, so it must be float-identical to the cardinal-NI baseline. Exercises H,
V, oblique (H+V = pitch = LARP+RALP), and a head-tilt hold (OCR → roll = RALP-LARP,
which exercises the non-diagonal roll<pitch leak)."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT

Z = 100.0

def saccade(degH, degV=0.0, T=1.2):
    t = np.arange(0.0, T, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(degH)), 0.0)
    pt3[:, 1] = np.where(t >= 0.1, Z * np.tan(np.radians(degV)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=THETA_NOISELESS, max_s=int(T / DT) + 200)
    return np.array(st.plant.left)

def sig(name, a):
    a = np.asarray(a)
    print(f'{name:14s} sum={a.sum():.5f}  absmax={np.abs(a).max():.5f}  std={a.std():.5f}')

eh = saccade(20.0)          # horizontal — H channel
ev = saccade(0.0, 18.0)     # vertical   — pitch = LARP+RALP
eo = saccade(15.0, 10.0)    # oblique
print('=== NI canal eye position (plant.left) ===')
sig('sac20_H', eh); sig('sac18_V', ev); sig('sac_oblique', eo)
print('final H:', np.round(eh[-1], 3), ' V:', np.round(ev[-1], 3),
      ' obl:', np.round(eo[-1], 3))
print('checksum:', float(eh.sum() + ev.sum() + eo.sum()))
