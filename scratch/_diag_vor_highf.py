"""Where does the VOR high-frequency gain drop enter? Trace per-stage gain vs
frequency for the dark VOR: head -> canal afferent -> VS command (vs_net) ->
raw eye velocity (d/dt plant) -> SPV (extract_spv_states). Whichever stage starts
falling with frequency is the culprit (sensory cascade / VS / motor pulse-step /
or the SPV-extraction smoothing)."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import extract_spv_states, vs_net, extract_canal
from oculomotor.benchmarks import bode

POS_MAX = 20.0


def amp(sig, t, f, frac=0.5):
    m = t >= frac * t[-1]
    a, _, _ = bode.fit_sinusoid(t[m], np.asarray(sig)[m], f)
    return a


print(f'{"f(Hz)":>6} {"canal":>8} {"vs_net":>8} {"eye_vel":>8} {"spv":>8}   (each ÷ head, normalized to f=0.5)')
rows = []
for f in (0.5, 1.0, 2.0, 5.0, 10.0, 20.0):
    w = 2 * np.pi * f
    V = bode.capped_velocity_amp(f, 30.0, POS_MAX)
    T_end = max(10.0, 10.0 / f)
    t = np.arange(0.0, T_end, DT); Tn = len(t)
    hv = np.zeros((Tn, 3)); hv[:, 0] = V * np.sin(w * t)
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.zeros(Tn), target_present=jnp.zeros(Tn), key=0)
    h = amp(hv[:, 0], t, f)
    canal = amp(np.array(extract_canal(st)), t, f) / h
    vsn = amp(np.array(vs_net(st))[:, 0], t, f) / h
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = amp(np.gradient(eye, DT), t, f) / h
    spv = amp(np.array(extract_spv_states(st, t, eye='version'))[:, 0], t, f) / h
    rows.append((f, canal, vsn, evel, spv))

# normalize each column to its f=0.5 value so we see the relative rolloff
base = rows[0]
for f, c, vn, ev, sp in rows:
    print(f'{f:6.1f} {c/base[1]:8.3f} {vn/base[2]:8.3f} {ev/base[3]:8.3f} {sp/base[4]:8.3f}')
print('\n(raw ÷head gains at f=0.5:  canal=%.3f vs_net=%.3f eye_vel=%.3f spv=%.3f)'
      % (base[1], base[2], base[3], base[4]))
