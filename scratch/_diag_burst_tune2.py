"""Re-sweep alpha_fac WITH the oblique curvature check (the horizontal-only sweep
missed that alpha_fac=0.5 wrecks obliques). Find a value that trims the overshoot
without breaking oblique straightness or the saccade count."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT, _saccade_onset_times
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)


def oblique_curvature(P, amp=12.0, d=45.0):
    pt3 = np.zeros((len(t), 3)); pt3[:, 2] = 1.0
    pt3[t >= t_jump, 0] = np.tan(np.radians(amp * np.cos(np.radians(d))))
    pt3[t >= t_jump, 1] = np.tan(np.radians(amp * np.sin(np.radians(d))))
    import jax.numpy as jnp
    st = _run(t, jnp.array(pt3), key=0, max_s=int(T_end / DT) + 200, params=P)
    eye = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
    h, v = eye[:, 0], eye[:, 1]
    i0 = int(t_jump / DT)
    h, v = h[i0:] - h[i0], v[i0:] - v[i0]
    speed = np.hypot(np.gradient(h, DT), np.gradient(v, DT))
    above = speed > 20.0
    if not above.any():
        return float('nan')
    on = int(np.argmax(above)); peak = on + int(np.argmax(speed[on:]))
    rest = np.where(~above[peak:])[0]
    off = peak + int(rest[0]) if len(rest) else len(h) - 1
    H1, V1 = h[off], v[off]; L = np.hypot(H1, V1)
    if L < 1.0:
        return float('nan')
    ux, uy = H1 / L, V1 / L
    hp, vp = h[on:off + 1], v[on:off + 1]
    return float(np.max(np.abs(hp * uy - vp * ux)) / L)


for af in [1.0, 0.85, 0.7, 0.6, 0.5]:
    P = with_brain(THETA_NOISELESS, alpha_fac=af)
    pt3 = _pt3(t, 40.0, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    ni = np.array(ni_net(st))[:, 0]
    ub = np.array(extract_burst(st, P))[:, 0]
    win = (t >= t_jump) & (t <= 0.6)
    over = ni[win].max() - 40.0
    nc = len(_saccade_onset_times(ub, t_jump))
    curv = oblique_curvature(P)
    print(f'alpha_fac={af}:  40deg cmd_overshoot={over:+.3f}  #sac={nc}  '
          f'oblique_curv={curv:.3f}  (band <0.08)')
