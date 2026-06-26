"""Pure controller fidelity: pursuit + OKN/OKR loops OPEN (no visual feedback),
so eye-vs-NI-command is the plant/pulse-step behavior alone and the glissade
reversal is uncontaminated by post-saccadic drift feeding back.

Ideal pulse-step: eye_vel == NI command vel (plant inverted). The glissade
(eye - command) is the uncancelled plant/FCP dynamics. A single neg->pos
reversal = a residual 1st-order lag (delay); a richer shape = higher order."""
import numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net
from oculomotor.sim.simulator import with_brain

t = np.arange(0.0, 0.9, DT)
P = with_brain(THETA_NOISELESS, K_pursuit=0.0, K_phasic_pursuit=0.0, K_pursuit_direct=0.0,
               K_cereb_pu=0.0, K_vor_direct=0.0, K_cereb_okr=0.0)   # all visual loops OFF

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for col, amp in enumerate((2.0, 10.0)):
    st   = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=P)
    eye  = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    ni   = np.array(ni_net(st))[:, 0]; nivel = np.gradient(ni, DT)
    gl   = evel - nivel
    win  = (t >= 0.28) & (t <= 0.52)
    ax = axes[col]
    ax.plot(t[win], nivel[win], color='#1b7837', lw=1.5, label='NI command vel (target)')
    ax.plot(t[win], evel[win],  color='#c0392b', lw=1.5, label='eye vel (actual)')
    ax.plot(t[win], gl[win],    color='gray',    lw=1.3, label='glissade (eye - command)')
    ax.axhline(0, color='k', lw=0.3); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title(f'{amp:.0f} deg saccade — controller, loops OFF')
    ax.set_xlabel('t (s)'); ax.set_ylabel('deg/s')
    # timing of the command vs eye peak, and the glissade reversal
    ic = int(np.argmax(np.abs(nivel * win))); ie = int(np.argmax(np.abs(evel * win)))
    print(f'{amp:4.0f} deg: cmd_peak={nivel[ic]:.0f}@{t[ic]:.3f}s  eye_peak={evel[ie]:.0f}@{t[ie]:.3f}s  '
          f'(eye lags {1000*(t[ie]-t[ic]):.0f} ms, peak ratio {evel[ie]/nivel[ic]:.2f})  '
          f'glissade neg={np.min(gl[win]):.1f} pos={np.max(gl[win]):.1f}')

fig.suptitle('Pure controller fidelity (pursuit + OKN loops OFF): eye vs NI command')
fig.tight_layout()
fig.savefig(r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_controller.png', dpi=110)
print('saved scratch/_controller.png')
