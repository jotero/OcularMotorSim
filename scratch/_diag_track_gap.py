"""Controller tracking gap: does the actual eye track NI_net, or glissade? And is
the gap a clean linear lag (recoverable by feedforward) or a hard FCP floor clip
(antagonist pinned at 0)? 40deg saccade, near target."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
pt3 = _pt3(t, 40.0, t_jump=t_jump)
st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)

ver = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
ni = np.array(ni_net(st))[:, 0]
eye_vel = np.gradient(ver, DT)
ni_vel = np.gradient(ni, DT)
mn = np.array(st.brain.fcp.mn)   # (T, 14): [LR_L,LR_R,CN4_L,CN4_R,MR_L,MR_R,...,AIN_L,AIN_R]
LR_L, LR_R, MR_L, MR_R = 0, 1, 4, 5   # rightward yaw: agonists LR_R/MR_L, antagonists LR_L/MR_R

w = (t >= 0.1) & (t <= 0.45)
print('peak |eye-desired| velocity (burst+settle) =', round(float(np.max(np.abs((eye_vel - ni_vel)[w]))), 2), 'deg/s')
print('position glissade max |eye-NI|            =', round(float(np.max(np.abs((ver - ni)[w]))), 3), 'deg')
print('antagonist MN minima:  MR_R =', round(float(mn[:, MR_R].min()), 1),
      ' LR_L =', round(float(mn[:, LR_L].min()), 1), '  (<0 => symmetric/no floor; ~0 => floored)')
print('agonist  MN maxima:    LR_R =', round(float(mn[:, LR_R].max()), 1),
      ' MR_L =', round(float(mn[:, MR_L].max()), 1))

fig, ax = plt.subplots(4, 1, figsize=(10, 11), sharex=True)
ax[0].plot(t, ni, label='NI_net (desired pos)'); ax[0].plot(t, ver, label='eye pos', ls='--')
ax[0].set_title('position tracking'); ax[0].legend(fontsize=8)
ax[1].plot(t, ni_vel, label='d(NI_net)/dt (desired vel)'); ax[1].plot(t, eye_vel, label='eye vel', ls='--')
ax[1].set_title('velocity tracking'); ax[1].legend(fontsize=8)
ax[2].plot(t, eye_vel - ni_vel, color='crimson', label='velocity gap (eye − desired) = glissade')
ax[2].axhline(0, color='k', lw=0.4); ax[2].set_title('tracking gap'); ax[2].legend(fontsize=8)
ax[3].plot(t, mn[:, LR_R], label='LR_R (agonist)')
ax[3].plot(t, mn[:, MR_R], label='MR_R (antagonist)')
ax[3].plot(t, mn[:, LR_L], label='LR_L (antagonist)')
ax[3].axhline(0, color='k', lw=0.6); ax[3].set_title('MN firing rates — floor check')
ax[3].legend(fontsize=8); ax[3].set_xlabel('t (s)')
for a in ax:
    a.set_xlim(0.08, 0.45); a.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_track_gap.png', dpi=110)
print('saved _track_gap.png')
