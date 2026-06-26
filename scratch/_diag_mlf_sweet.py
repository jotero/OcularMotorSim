"""Verify the mlf_lead~0.5 sweet spot across saccade sizes, suppression OFF.
Post-sac ring = peak|eye_vel| in the settling window (z_opn>=50). If mlf_lead~0.5
minimises the ring for small AND large saccades with suppression off, it's a
real controller fix that removes the need for suppression."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT)
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

t = np.arange(0.0, 0.9, DT)


def ring(mlf, amp):
    P = with_brain(THETA_NOISELESS, mn_ff_yaw=1.0, mlf_lead=mlf,
                   saccadic_suppression_steepness=0.0)              # suppression OFF
    st = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=P)
    eye  = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    z    = extract_z_opn(st); ub = np.array(extract_burst(st, P))[:, 0]
    fast = np.abs(ub) > 20.0
    be   = int(np.where(fast)[0][-1]) if fast.any() else 0
    rw = np.zeros_like(t, bool); rw[be + 30:be + 230] = True; rw &= (z >= 50.0)
    return float(np.max(np.abs(evel[rw]))) if rw.any() else float('nan')


mlfs = [0.0, 0.25, 0.5, 0.75, 1.0]
print('post-saccadic ring (loops ON, suppression OFF) vs mlf_lead:')
print(f'{"amp":>4} ' + ' '.join(f'{"mlf="+format(m,".2f"):>9}' for m in mlfs))
for amp in (2.0, 5.0, 10.0, 20.0):
    rings = [ring(m, amp) for m in mlfs]
    best = mlfs[int(np.nanargmin(rings))]
    print(f'{amp:4.0f} ' + ' '.join(f'{r:9.2f}' for r in rings) + f'   best@{best:.2f}')
