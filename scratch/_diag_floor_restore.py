"""Can the MN rectification FLOOR be restored now that mlf_lead=0.5 carries the
post-sac fix? Monkeypatch _smooth_clip back to the two-sided (floored) form and
re-check post-sac ring + main sequence at the new defaults (mlf_lead=0.5, supp
off). Floor-OFF reference (current defaults): 2deg ring~0.98, 5deg~0.21,
peak_vel@20~642, #sac@40=1.

Floored clip = softplus(z) - softplus(z - g_max)  (smooth [0, g_max], floor at 0)
vs current   = z          - softplus(z - g_max)  (one-sided, passes negatives)."""
import numpy as np, jax, jax.numpy as jnp
import oculomotor.models.brain_models.final_common_pathway as fcp

def _floored(z, g_max):
    return jax.nn.softplus(z) - jax.nn.softplus(z - g_max)   # restore rectification floor
fcp._smooth_clip = _floored

from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT, _saccade_onset_times)
from oculomotor.analysis import ni_net, extract_burst

t = np.arange(0.0, 0.9, DT); i_j = int(0.1 / DT)


def sac(amp):
    st = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=THETA_NOISELESS)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    z = extract_z_opn(st); ub = np.array(extract_burst(st, THETA_NOISELESS))[:, 0]
    fast = np.abs(ub) > 20.0; be = int(np.where(fast)[0][-1]) if fast.any() else 0
    rw = np.zeros_like(t, bool); rw[be + 30:be + 230] = True; rw &= (z >= 50.0)
    ring = float(np.max(np.abs(evel[rw]))) if rw.any() else float('nan')
    pv = float(np.max(np.abs(evel[i_j:i_j + 400])))
    nsac = len(_saccade_onset_times(ub, 0.1))
    return ring, pv, nsac


print('FLOOR RESTORED (mlf_lead=0.5, suppression off):')
print(f'{"amp":>4} {"post-sac ring":>13} {"peak_vel":>9} {"#sac":>5}   (floor-off ref)')
ref = {2.0: '0.98', 5.0: '0.21', 20.0: '0.67', 40.0: '#sac=1'}
for amp in (2.0, 5.0, 20.0, 40.0):
    ring, pv, n = sac(amp)
    print(f'{amp:4.0f} {ring:13.2f} {pv:9.0f} {n:5d}   (~{ref[amp]})')
