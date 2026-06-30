"""Pair common-mode shift vs per-muscle fold.
Metrics: (a) INO spurious pulse (left adducting eye peak velocity, should be SLOW),
         (b) healthy conjugacy (transient disconjugacy = MLF lag, should be SMALL),
         (c) healthy main sequence (peak velocity, should be ~unchanged).
Far target (z=100) so version is clean (no vergence confound)."""
import sys
import numpy as np, jax, jax.numpy as jnp
import oculomotor.models.brain_models.final_common_pathway as fcp
fcp._FOLD = (len(sys.argv) > 1 and sys.argv[1] == 'fold')   # set BEFORE importing the sim graph
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

Z = 100.0

def saccade(theta, deg):
    t = np.arange(0.0, 1.0, DT)
    pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=theta)
    L = np.array(st.plant.left)[:, 0]; R = np.array(st.plant.right)[:, 0]
    return t, L, R

print(f'pull-only mode: {"FOLD" if fcp._FOLD else "PAIR-SHIFT"}')
print('--- HEALTHY conjugate ---')
for deg in (20.0, 40.0):
    t, L, R = saccade(THETA_NOISELESS, deg)
    lr = L - R
    td = np.abs(lr - lr[-1])                       # transient disconjugacy (MLF lag), vergence removed
    pv = np.abs(np.gradient(L, DT)).max()
    print(f'  {deg:4.0f}deg: L_end {L[-1]:6.2f} R_end {R[-1]:6.2f}  peakvel {pv:4.0f}  '
          f'max transient |d(L-R)| {td.max():.3f} (t={t[td.argmax()]:.2f})  steady |L-R| {abs(lr[-1]):.3f}')

print('--- INO g_mlf_L=0 (left eye adducting) ---')
for deg in (20.0, 40.0):
    t, L, R = saccade(with_brain(THETA_NOISELESS, g_mlf_L=0.0), deg)
    pv = np.abs(np.gradient(L, DT)).max()
    print(f'  {deg:4.0f}deg: L(adduct)_end {L[-1]:6.2f}  peak|vel| {pv:4.0f}  R(abduct)_end {R[-1]:6.2f}')
