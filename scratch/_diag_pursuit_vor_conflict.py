"""Does pursuit wind up against the VOR at low frequency? Trace, in the settled
0.05 Hz light+target run, the VS/VOR drive vs the pursuit memory vs the eye.

Eye-in-head velocity ≈ −w_est (VOR) + u_pursuit (pursuit). Compensatory = −head.
If pursuit (pu.R−pu.L) is IN PHASE with head (same sign), it SUBTRACTS from the
VOR's −w_est → gain < 1. Negative corr(VS_net, pursuit_net) = pursuit opposing VOR.
Also compares against target_present=0 (pursuit off) to confirm gain recovers."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import vs_net, extract_spv_states

AMP, SETTLE, f, T_end = 30.0, 2.0, 0.05, 250.0
w = 2 * np.pi * f
t = np.arange(0.0, T_end, DT); Tn = len(t)
on = t >= SETTLE
hv = np.zeros((Tn, 3)); hv[:, 0] = np.where(on, AMP * np.sin(w * (t - SETTLE)), 0.0)


def run(target_p):
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.full(Tn, 1.0),
                   target_present=jnp.full(Tn, target_p), key=0)
    return st


def amp(x):
    return (np.percentile(x, 97.5) - np.percentile(x, 2.5)) / 2.0


win = t >= (T_end - 80.0)              # settled window
head = hv[:, 0]
print(f'0.05 Hz, settled (last 80 s).  head amp = {amp(head[win]):.1f} deg/s\n')

for tp in (1.0, 0.0):
    st = run(tp)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    eye_vel = np.gradient(eye, DT)
    vsn = np.array(vs_net(st))[:, 0]                       # VS net (≈ w_est, VOR drive source)
    pun = np.array(st.brain.pu.R - st.brain.pu.L)[:, 0]    # pursuit memory net (= u_pursuit cmd)
    spv = -extract_spv_states(st, t, eye='version')[:, 0]  # compensatory slow-phase gain
    g = amp(spv[win]) / amp(head[win])
    cc = np.corrcoef(vsn[win], pun[win])[0, 1]
    lab = 'pursuit ON (target_present=1)' if tp else 'pursuit OFF (target_present=0)'
    print(f'{lab}')
    print(f'   slow-phase gain      = {g:.3f}')
    print(f'   VS_net amp           = {amp(vsn[win]):6.1f} deg/s')
    print(f'   pursuit_net amp      = {amp(pun[win]):6.1f} deg/s')
    print(f'   eye_vel amp          = {amp(eye_vel[win]):6.1f} deg/s')
    print(f'   corr(VS_net,pursuit) = {cc:+.2f}  (<0 = pursuit opposes VOR)\n')
