"""Does the DARK VOR actually roll off at very low frequency (as a high-pass must),
or does it stay flat to DC (which would be a bug — a non-leaky integrator / DC path)?

Sweep below the current Bode floor (0.005-0.05 Hz) in the DARK, using a SMALL,
constant POSITION amplitude (12 deg) so the eye never leaves range (no
out-of-range contamination, no nystagmus). Compare the measured gain/phase to a
clean first-order high-pass with the measured post-rotatory TC (34.6 s):
    |H| = wTC / sqrt(1 + (wTC)^2),  phase = atan(1/(wTC))  (lead)
If the model matches -> the rolloff is real, just below the plotted range.
If gain stays ~1 at 0.005 Hz -> DC pathway bug."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import extract_spv_states
from oculomotor.benchmarks import bode

POS_AMP, SETTLE, N_CYC, TC = 12.0, 150.0, 4.0, 34.6   # SETTLE >4*TC so transient is gone

print(f'DARK VOR, constant POSITION amplitude {POS_AMP} deg (in range, no nystagmus)')
print(f'compared to a clean high-pass, TC = {TC} s (corner {1/(2*np.pi*TC):.4f} Hz)\n')
print(f'{"freq(Hz)":>9} {"V(deg/s)":>9} {"pos_amp":>8} {"gain":>7} {"phase":>8}   '
      f'{"HP gain":>8} {"HP phase":>9}')
for f in (0.005, 0.01, 0.02, 0.05):
    w = 2 * np.pi * f
    V = POS_AMP * w
    T_end = SETTLE + N_CYC / f
    t = np.arange(0.0, T_end, DT); Tn = len(t)
    on = t >= SETTLE
    hv = np.zeros((Tn, 3)); hv[:, 0] = np.where(on, V * np.sin(w * (t - SETTLE)), 0.0)
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.zeros(Tn), target_present=jnp.zeros(Tn), key=0)
    spv = -extract_spv_states(st, t, eye='version')[:, 0]
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    g, p = bode.bode_point(t, hv[:, 0], spv, f, settle_frac=0.5)
    pos_amp = (np.percentile(eye, 97.5) - np.percentile(eye, 2.5)) / 2.0
    wTC = w * TC
    hp_g = wTC / np.sqrt(1 + wTC ** 2)
    hp_p = np.degrees(np.arctan(1.0 / wTC))
    print(f'{f:9.3f} {V:9.3f} {pos_amp:8.1f} {g:7.3f} {p:+8.1f}   {hp_g:8.3f} {hp_p:+9.1f}')
