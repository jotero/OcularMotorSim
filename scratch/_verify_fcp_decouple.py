"""Verify the FCP read_activations decouple: muscle MNs >=0, AIN signed,
and (healthy) the MN firing equals the per-muscle nerve drive."""
import numpy as np, jax, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.analysis import extract_fcp_cascade

t = np.arange(0.0, 1.2, DT)
pt3 = np.zeros((len(t), 3)); pt3[:, 2] = 1.0
pt3[:, 0] = np.where(t >= 0.1, np.tan(np.radians(20.0)), 0.0)   # 20 deg rightward saccade
st = _run(t, jnp.array(pt3), key=0, params=THETA_NOISELESS)
c = extract_fcp_cascade(st, THETA_NOISELESS)
mn, nrv = c['mn'], c['nerves']

musc = mn[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]   # 12 muscle MNs
ain  = mn[:, [12, 13]]                                  # AIN (internuclear)
print(f'muscle MN firing  min={musc.min():+.2f}  max={musc.max():.1f}   (expect min ~>= 0)')
print(f'AIN firing        min={ain.min():+.2f}  max={ain.max():.1f}   (signed OK)')
print(f'nerve drive       min={nrv.min():+.2f}  max={nrv.max():.1f}   (>= 0)')

# Healthy: each muscle MN == its target nerve.  MN order -> nerve index:
pairs = {'LR_L': (0, 0), 'MR_L': (4, 1), 'SR_L': (6, 2), 'IR_L': (8, 3),
         'IO_L': (10, 5), 'CN4_L->SO_R': (2, 10), 'CN4_R->SO_L': (3, 4),
         'LR_R': (1, 6), 'MR_R': (5, 7)}
print('\nMN firing == nerve (healthy), max|diff|:')
for name, (im, inv) in pairs.items():
    print(f'  {name:14s} {np.max(np.abs(mn[:, im] - nrv[:, inv])):.3e}')

# eye behaviour sanity (should be unchanged vs pre-refactor)
print(f'\neye yaw final = {0.5*(np.array(st.plant.left)[-1,0]+np.array(st.plant.right)[-1,0]):.2f} deg (target 20)')
