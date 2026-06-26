"""Suppression OFF (saccadic_suppression_steepness=0 -> gate==1). Characterize the
~6 deg/s EC residual: is it the CONTROLLER (eye lags the NI command through the
saccade) or the EC PREDICTION/cascade (sensed slip != predicted slip even when
eye==NI)?

Top row:    eye velocity vs NI_net velocity   (controller fidelity)
Bottom row: sensed scene slip vs -EC prediction + their sum (the residual)
If eye~=NI but sensed != -EC -> it's the EC prediction/cascade, not the controller."""
import numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT, read_brain_acts
from oculomotor.analysis import ni_net
from oculomotor.sim.simulator import with_brain

t = np.arange(0.0, 0.9, DT)
P = with_brain(THETA_NOISELESS, saccadic_suppression_steepness=0.0)   # suppression OFF

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for col, amp in enumerate((2.0, 10.0)):
    st   = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=P)
    acts = read_brain_acts(st, P)
    eye  = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    ni   = np.array(ni_net(st))[:, 0]; nivel = np.gradient(ni, DT)
    sslip = np.array(acts.pc.scene_angular_vel)[:, 0]     # sensed scene slip (delayed)
    ecp   = np.array(acts.cb.fl_okr_drive)[:, 0]          # EC prediction (un-gated, supp off)
    resid = sslip + ecp                                   # EC residual (= scene_mm at sat=1)
    win = (t >= 0.25) & (t <= 0.62)

    ax = axes[0, col]
    ax.plot(t[win], evel[win], color='#c0392b', lw=1.4, label='eye vel')
    ax.plot(t[win], nivel[win], color='#1b7837', lw=1.0, ls='--', label='NI_net vel (command)')
    ax.plot(t[win], (evel - nivel)[win], color='gray', lw=1.0, label='eye - NI (glissade)')
    ax.axhline(0, color='k', lw=0.3); ax.legend(fontsize=8); ax.grid(alpha=0.2)
    ax.set_title(f'{amp:.0f} deg — controller: eye vs NI command'); ax.set_ylabel('deg/s')

    ax2 = axes[1, col]
    ax2.plot(t[win], sslip[win], color='darkorange', lw=1.4, label='sensed scene slip')
    ax2.plot(t[win], -ecp[win], color='#2166ac', lw=1.0, ls='--', label='-EC prediction')
    ax2.plot(t[win], resid[win], color='k', lw=1.6, label='residual (sensed + EC)')
    ax2.axhline(0, color='k', lw=0.3); ax2.legend(fontsize=8); ax2.grid(alpha=0.2)
    ax2.set_title(f'{amp:.0f} deg — EC mismatch (suppression OFF)')
    ax2.set_xlabel('t (s)'); ax2.set_ylabel('deg/s')

    g = lambda x: float(np.max(np.abs(x[win])))
    print(f'{amp:4.0f} deg:  peak |eye-NI|={g(evel-nivel):5.2f}   '
          f'peak sensed_slip={g(sslip):6.2f}  peak EC_pred={g(ecp):6.2f}  peak residual={g(resid):5.2f}')

fig.suptitle('EC residual with suppression OFF — controller (eye vs NI) vs EC prediction (sensed vs predicted slip)')
fig.tight_layout()
fig.savefig(r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_ec_residual.png', dpi=110)
print('saved scratch/_ec_residual.png')
