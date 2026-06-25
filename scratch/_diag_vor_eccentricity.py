"""Is the low-frequency light+target gain collapse driven by POSITION amplitude
(earth-fixed target driven out of oculomotor range), not by frequency per se?

Hold f = 0.05 Hz fixed and vary the velocity amplitude V so the head/target
POSITION amplitude p = V/(2*pi*f) sweeps small->large. If the light+target
slow-phase gain is ~1 at small p and collapses only at large p (while the eye
position saturates), the driver is eccentricity/excursion, not a pursuit-vs-VOR
sign conflict (which would be amplitude-independent)."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import extract_spv_states

f, SETTLE, T_end = 0.05, 2.0, 150.0
w = 2 * np.pi * f
t = np.arange(0.0, T_end, DT); Tn = len(t)
on = t >= SETTLE
win = t >= (T_end - 60.0)


def amp(x):
    return (np.percentile(x, 97.5) - np.percentile(x, 2.5)) / 2.0


print(f'f = {f} Hz FIXED.  vary velocity amp V -> position amp p = V/(2*pi*f)')
print(f'orbital_limit param = {THETA_NOISELESS.brain.orbital_limit} deg\n')
print(f'{"pos_amp":>8} {"V(deg/s)":>9} {"L+T gain":>9} {"eye_pos_amp":>12}  reads as')
for p in (5.0, 15.0, 30.0, 60.0, 95.0):
    V = p * w
    hv = np.zeros((Tn, 3)); hv[:, 0] = np.where(on, V * np.sin(w * (t - SETTLE)), 0.0)
    head = hv[:, 0]
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.full(Tn, 1.0),
                   target_present=jnp.full(Tn, 1.0), key=0)
    spv = -np.array(extract_spv_states(st, t, eye='version'))[:, 0]
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    g = amp(spv[win]) / amp(head[win])
    epa = amp(eye[win])
    note = 'target in range' if p < 25 else 'target OUT of range'
    print(f'{p:8.0f} {V:9.2f} {g:9.3f} {epa:12.1f}  {note}')
