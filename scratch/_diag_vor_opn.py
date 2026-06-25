"""During the high-frequency dark VOR, is the saccade generator firing discrete
small saccades occasionally, or is the burst continuously active (locked to the
oscillation)? Trace z_opn (OPN gate) + u_burst + eye velocity, count discrete
z_opn<50 epochs, and report burst amplitude."""
import numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import extract_z_opn, extract_burst
from oculomotor.benchmarks import bode

for f in (5.0, 10.0):
    w = 2 * np.pi * f
    V = bode.capped_velocity_amp(f, 30.0, 20.0)
    T_end = 6.0
    t = np.arange(0.0, T_end, DT); Tn = len(t)
    hv = np.zeros((Tn, 3)); hv[:, 0] = V * np.sin(w * t)
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.zeros(Tn), target_present=jnp.zeros(Tn), key=0)
    z = np.array(extract_z_opn(st))
    ub = np.array(extract_burst(st, THETA_NOISELESS))[:, 0]
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    is_fast = z < 50.0
    edges = np.diff(np.concatenate([[0], is_fast.astype(int), [0]]))
    n_ev = int((edges == 1).sum())
    cyc = f * T_end
    print(f'\n=== {f:.0f} Hz dark VOR ({T_end:.0f}s, {cyc:.0f} cycles) ===')
    print(f'  z_opn<50: {is_fast.mean():.0%} of time, {n_ev} discrete epochs '
          f'({n_ev/T_end:.0f}/s, {n_ev/cyc:.2f}/cycle)')
    print(f'  u_burst:  peak={np.abs(ub).max():6.0f} deg/s   '
          f'median|burst| when z_opn<50 = {np.median(np.abs(ub[is_fast])):.0f} deg/s')
    print(f'  eye pos excursion: +/-{(np.percentile(eye, 97.5)-np.percentile(eye, 2.5))/2:.2f} deg')
    print(f'  eye vel peak: {np.abs(evel).max():.0f} deg/s  (head vel peak {V:.0f})')
    if f == 10.0:
        win = (t >= 4.0) & (t <= 4.5)
        fig, ax = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
        ax[0].plot(t[win], hv[win, 0], color='#888', label='head vel')
        ax[0].plot(t[win], evel[win], color='#c0392b', label='eye vel'); ax[0].legend(fontsize=8)
        ax[0].set_ylabel('deg/s')
        ax[1].plot(t[win], z[win], color='purple'); ax[1].axhline(50, color='r', ls='--', lw=0.8)
        ax[1].set_ylabel('z_opn'); ax[1].fill_between(t[win], 0, 200, where=is_fast[win], color='red', alpha=0.1)
        ax[2].plot(t[win], ub[win], color='darkorange'); ax[2].set_ylabel('u_burst (deg/s)')
        ax[2].set_xlabel('t (s)')
        for a in ax:
            a.grid(alpha=0.2)
        fig.suptitle('SG activity during 10 Hz dark VOR (settled window, red = z_opn<50)')
        fig.tight_layout()
        fig.savefig(r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_vor_opn.png', dpi=110)
        print('  saved scratch/_vor_opn.png')
