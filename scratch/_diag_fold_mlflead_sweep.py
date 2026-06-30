"""Fold mode: can mlf_lead re-tune away the healthy transient disconjugacy
the fold introduces? (INO is g_mlf=0 so mlf_lead-independent there.)"""
import numpy as np, jax.numpy as jnp
import oculomotor.models.brain_models.final_common_pathway as fcp
fcp._FOLD = True
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

Z = 100.0
def saccade(theta, deg):
    t = np.arange(0.0, 1.0, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=theta)
    return t, np.array(st.plant.left)[:, 0], np.array(st.plant.right)[:, 0]

print('FOLD mode — healthy conjugate, sweep mlf_lead (default 0.5):')
for ml in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0):
    theta = with_brain(THETA_NOISELESS, mlf_lead=float(ml))
    row = []
    for deg in (20.0, 40.0):
        t, L, R = saccade(theta, deg)
        lr = L - R; td = np.abs(lr - lr[-1])
        row.append((np.abs(np.gradient(L, DT)).max(), td.max(), t[td.argmax()]))
    print(f'  mlf_lead={ml:.2f}: '
          f'20deg pv {row[0][0]:4.0f} disconj {row[0][1]:.3f}(t={row[0][2]:.2f}) | '
          f'40deg pv {row[1][0]:4.0f} disconj {row[1][1]:.3f}(t={row[1][2]:.2f})')
