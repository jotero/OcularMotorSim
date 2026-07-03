"""Quantify post-saccadic oscillation: eye velocity RMS in the settling window
after a 20 deg horizontal saccade. Run twice (current pulse-step vs muscle-pole
dropped from the inverse) to test whether the muscle-pole cancellation is the
ring source."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net

Z = 100.0
def saccade(deg=20.0, T=1.0):
    t = np.arange(0.0, T, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=THETA_NOISELESS, max_s=int(T / DT) + 200)
    return t, np.array(st.plant.left)[:, 0], ni_net(st)[:, 0]

t, eye, nin = saccade()
ev = np.gradient(eye, DT)
# saccade ends ~ when |ev| first drops back below 20 deg/s after the peak
peak_i = np.argmax(np.abs(ev))
end_i = peak_i + np.argmax(np.abs(ev[peak_i:]) < 20.0)
t_end = t[end_i]
# settling window: 20-200 ms after saccade end
win = (t > t_end + 0.02) & (t < t_end + 0.20)
rms = np.sqrt(np.mean(ev[win] ** 2))
pkpk = eye[win].max() - eye[win].min()
overshoot = eye[win].max() - eye[-1]
print(f'saccade end ~{t_end*1000:.0f} ms  final eye {eye[-1]:.3f}  final ni_net {nin[-1]:.3f}')
print(f'post-saccade (20-200ms):')
print(f'  EYE     vel RMS = {rms:.3f} deg/s   pos pk-pk = {pkpk:.4f} deg')
niv = np.gradient(nin, DT)
print(f'  NI_net  vel RMS = {np.sqrt(np.mean(niv[win]**2)):.3f} deg/s   '
      f'pos pk-pk = {nin[win].max()-nin[win].min():.4f} deg  <- SG loop (plant-independent)')
