"""Inspect the pursuit Bode at one frequency: raw eye velocity vs slow-phase
(extract_spv_states) vs target. Is the slow-phase clean, and does the gain come
from it or from saccade contamination?"""
import numpy as np
import jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_pursuit import _run, THETA_NOISELESS, DT
from oculomotor.analysis import extract_spv_states, extract_z_opn
import oculomotor.benchmarks.bode as bode

AMP, N_CYC, SETTLE = 10.0, 5, 1.5
for f in [0.5, 1.5]:
    T_end = min(SETTLE + N_CYC / f, 45.0)
    t = np.arange(0.0, T_end, DT); Tn = len(t); w = 2 * np.pi * f
    on = t >= SETTLE
    vel = np.where(on, AMP * np.sin(w * (t - SETTLE)), 0.0)
    pos = np.where(on, -(AMP / w) * (np.cos(w * (t - SETTLE)) - 1.0), 0.0)
    pt3 = np.zeros((Tn, 3)); pt3[:, 2] = 1.0; pt3[:, 0] = np.tan(np.radians(pos))
    vt3 = np.zeros((Tn, 3)); vt3[:, 0] = vel.astype(np.float32)
    st = _run(THETA_NOISELESS, t, jnp.array(pt3), jnp.array(vt3), key=0)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    raw = np.gradient(eye, DT)
    spv = extract_spv_states(st, t)[:, 0]
    z = extract_z_opn(st)
    g_raw, _ = bode.bode_point(t, vel, raw, f, settle_frac=0.45)
    g_spv, _ = bode.bode_point(t, vel, spv, f, settle_frac=0.45)
    frac_masked = float(np.mean((z < 50)[on]))
    print(f'f={f} Hz: gain(raw)={g_raw:.3f}  gain(spv)={g_spv:.3f}  '
          f'pos sweep amp={AMP/w:.1f} deg  frac fast-phase={frac_masked:.2f}')
    if f == 0.5:
        fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(t, vel, 'k', label='target vel')
        ax[0].plot(t, raw, color='orange', alpha=0.6, label='raw eye vel (w/ saccades)')
        ax[0].plot(t, spv, color='steelblue', lw=2, label='slow-phase vel (masked)')
        ax[0].legend(fontsize=8); ax[0].set_ylabel('vel (deg/s)'); ax[0].set_ylim(-25, 25)
        ax[0].grid(alpha=0.25)
        ax[1].plot(t, z, label='z_opn'); ax[1].axhline(50, color='r', ls='--', label='fast<50')
        ax[1].legend(fontsize=8); ax[1].set_xlabel('t (s)'); ax[1].set_xlim(SETTLE, SETTLE + 4)
        ax[1].grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_pursuit_spv.png', dpi=110)
        print('saved _pursuit_spv.png')
