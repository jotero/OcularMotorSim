"""Side-1 EC mismatch: timing vs magnitude. Overlay sensed slip vs EC prediction
for a 5deg saccade; test whether the residual correlates with the slip (magnitude
/ gain error) or its time-derivative (timing / delay-mismatch error)."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT, read_brain_acts

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
pt3 = _pt3(t, 5.0, t_jump=t_jump)
st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
acts = read_brain_acts(st, THETA_NOISELESS)

ver = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
eye_vel = np.gradient(ver[:, 0], DT)

s_sc  = np.array(acts.pc.scene_angular_vel)[:, 0]
vis_sc = np.array(acts.pc.scene_visible)
ec_sc = np.array(acts.cb.ec_scene)[:, 0]
pred_sc = -vis_sc * ec_sc                                   # EC prediction (should match slip)
r_sc = np.array(acts.cb.saccadic_suppression_scene) * s_sc + np.array(acts.cb.fl_okr_drive)[:, 0]

s_tg = np.array(acts.pc.target_vel)[:, 0]
r_tg = np.array(acts.cb.pred_err)[:, 0]
pred_tg = s_tg - r_tg

w = (t >= 0.12) & (t <= 0.5)


def analyze(s, r, name):
    s_, r_ = s[w], r[w]
    ds = np.gradient(s_, DT)

    def fit(x):
        x = x - x.mean(); y = r_ - r_.mean()
        sl = np.dot(x, y) / np.dot(x, x) if np.dot(x, x) > 1e-9 else 0.0
        cc = np.corrcoef(x, y)[0, 1] if x.std() > 1e-9 and y.std() > 1e-9 else 0.0
        return sl, cc
    gs, gc = fit(s_)            # residual ~ slip  => gain error (slope = 1-g)
    ts, tc = fit(ds)           # residual ~ dslip => timing error (slope = delay, s)
    print(f'{name}: peak|resid|={np.max(np.abs(r_)):.2f} deg/s | '
          f'MAGNITUDE r~s: slope={gs:+.3f} corr={gc:+.2f} | '
          f'TIMING r~ds/dt: slope={ts*1000:+.1f} ms corr={tc:+.2f}')


analyze(s_sc, r_sc, 'scene ')
analyze(s_tg, r_tg, 'target')

fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
ax[0].plot(t, s_sc, label='scene slip (sensed)')
ax[0].plot(t, pred_sc, label='EC prediction', ls='--')
ax[0].plot(t, r_sc, label='residual (mismatch)', lw=2, color='crimson')
ax[0].set_title('scene channel'); ax[0].legend(fontsize=8); ax[0].axhline(0, color='k', lw=0.4)
ax[1].plot(t, s_tg, label='target slip (sensed)')
ax[1].plot(t, pred_tg, label='EC prediction', ls='--')
ax[1].plot(t, r_tg, label='residual (pred_err)', lw=2, color='crimson')
ax[1].set_title('target channel'); ax[1].legend(fontsize=8); ax[1].axhline(0, color='k', lw=0.4)
ax[2].plot(t, eye_vel, color='k', label='version eye vel')
ax[2].set_title('eye velocity'); ax[2].legend(fontsize=8); ax[2].set_xlabel('t (s)')
for a in ax:
    a.set_xlim(0.1, 0.5); a.grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_ec_waveform.png', dpi=110)
print('saved _ec_waveform.png')
