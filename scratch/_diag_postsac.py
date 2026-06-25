"""Post-saccadic ring decomposition: version / monocular / vergence + vs NI_net.

Attributes the post-burst eye velocity to plant glissade (eye vs NI_net, lead B)
vs per-eye asymmetry (vergence L-R, lead E), for a small and a large saccade.
"""
import numpy as np
import jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, _pt3, extract_z_opn, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)


def peak(x, win):
    v = np.gradient(x, DT)
    return float(np.max(np.abs(v[win]))) if win.any() else float('nan')


for amp in [5.0, 40.0]:
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
    L = np.array(st.plant.left)
    R = np.array(st.plant.right)
    ver = (L + R) / 2.0
    vrg = L - R
    ni = np.array(ni_net(st))                      # (T,3) conjugate NI net, deg
    z = extract_z_opn(st)
    win = (t >= t_jump + 0.05) & (t <= 0.45) & (z >= 50.0)   # short post-burst window

    print(f'\n=== {amp:.0f} deg saccade (post-burst slow window) ===')
    print(f'  version eye vel (the ring)        = {peak(ver[:,0], win):6.2f} deg/s')
    print(f'  version eye - NI_net vel  (lead B)= {peak(ver[:,0]-ni[:,0], win):6.2f} deg/s   <- plant glissade vs NI')
    print(f'  NI_net vel (post-burst)           = {peak(ni[:,0], win):6.2f} deg/s   <- is the command still moving?')
    print(f'  vergence (L-R) vel        (lead E)= {peak(vrg[:,0], win):6.2f} deg/s   <- per-eye asymmetry')
    print(f'  monocular L vel = {peak(L[:,0], win):6.2f}   monocular R vel = {peak(R[:,0], win):6.2f} deg/s')
    # static landing gaps (deg) in the hold
    hold = (t >= 0.6) & (t <= 0.85)
    print(f'  hold: version eye-NI gap = {np.mean(ver[hold,0]-ni[hold,0]):+.3f} deg, '
          f'vergence = {np.mean(vrg[hold,0]):+.3f} deg')
