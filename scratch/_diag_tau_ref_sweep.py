"""Sweep the pulse-slide-step Ts (brain.tau_slide): post-saccadic ring vs peak
velocity vs landing. 20 deg horizontal saccade, noiseless. Ring measured RELATIVE
to the saccade's own end (velocity threshold) so the Ts delay doesn't contaminate it."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

Z = 100.0
def saccade(theta, deg=20.0, T=0.8):
    t = np.arange(0.0, T, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=theta, max_s=int(T / DT) + 200)
    return t, np.array(st.plant.left)[:, 0]

print(f'{"tau_slide(ms)":>13} {"peak_vel":>9} {"end(ms)":>8} {"ring_pkpk":>10} {"ring_RMS":>9} {"final":>7}')
for tr in [0.001, 0.003, 0.005, 0.008, 0.013, 0.020]:
    theta = with_brain(THETA_NOISELESS, tau_slide=float(tr))
    t, eye = saccade(theta)
    ev = np.gradient(eye, DT)
    pkv = np.abs(ev).max()
    peak_i = int(np.argmax(np.abs(ev)))
    # saccade end: first sample after the peak where |v| falls below 30 deg/s
    below = np.where(np.abs(ev[peak_i:]) < 30.0)[0]
    end_i = peak_i + (int(below[0]) if len(below) else 0)
    t_end = t[end_i]
    win = (t > t_end + 0.02) & (t < t_end + 0.18)     # settling window, saccade-relative
    pkpk = eye[win].max() - eye[win].min()
    rms = np.sqrt(np.mean(ev[win] ** 2))
    print(f'{tr*1000:>13.0f} {pkv:>9.1f} {t_end*1000:>8.0f} {pkpk:>10.4f} {rms:>9.3f} {eye[-1]:>7.3f}')
