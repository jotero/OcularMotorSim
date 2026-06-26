"""Double-check before building on the hypothesis:
(1) Does mlf_lead=1 actually RING at the pure-controller level (loops+suppression
    OFF), or was the Decomp-#6 ring the suppression/loop? Compare the post-sac
    ring with: loops OFF | loops ON + supp ON | loops ON + supp OFF.
(2) Does mlf_lead=1 truly MINIMIZE the transient glissade (= optimally cancel the
    monocular MLF lag)? Measure the glissade vs mlf_lead, loops off.

glissade = peak|eye_vel - NI_vel| during the burst (transient controller lag)
ring     = peak|eye_vel| in the post-burst settling window (z_opn>=50, no fast phase)"""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT)
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

t = np.arange(0.0, 0.9, DT)
LOOPS_OFF = dict(K_pursuit=0.0, K_phasic_pursuit=0.0, K_pursuit_direct=0.0,
                 K_cereb_pu=0.0, K_vor_direct=0.0, K_cereb_okr=0.0)


def measure(mnff, mlf, mode, amp=5.0):
    kw = dict(mn_ff_yaw=mnff, mlf_lead=mlf)
    if mode == 'loops_off':
        kw.update(LOOPS_OFF)
    elif mode == 'supp_off':
        kw['saccadic_suppression_steepness'] = 0.0
    P = with_brain(THETA_NOISELESS, **kw)
    st = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=P)
    eye  = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    nivel = np.gradient(np.array(ni_net(st))[:, 0], DT)
    z  = extract_z_opn(st)
    ub = np.array(extract_burst(st, P))[:, 0]
    fast = np.abs(ub) > 20.0
    be = int(np.where(fast)[0][-1]) if fast.any() else 0
    gl = float(np.max(np.abs((evel - nivel)[fast]))) if fast.any() else float('nan')
    rw = np.zeros_like(t, bool); rw[be + 30:be + 230] = True
    rw &= (z >= 50.0)
    ring = float(np.max(np.abs(evel[rw]))) if rw.any() else float('nan')
    return gl, ring


print(f'{"mn_ff":>6} {"mlf":>4} {"loops_off":>20} {"loopsON+suppON":>18} {"loopsON+suppOFF":>18}')
print(f'{"":>6} {"":>4} {"glis   ring":>20} {"glis   ring":>18} {"glis   ring":>18}')
for mnff, mlf in [(1.0, 0.0), (1.0, 0.5), (1.0, 1.0), (1.5, 0.0)]:
    cells = []
    for mode in ('loops_off', 'default', 'supp_off'):
        gl, ring = measure(mnff, mlf, mode)
        cells.append(f'{gl:5.0f} {ring:5.2f}')
    print(f'{mnff:6.2f} {mlf:4.1f}   {cells[0]:>16}   {cells[1]:>14}   {cells[2]:>14}')
