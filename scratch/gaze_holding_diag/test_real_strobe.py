"""Does a REAL strobed target (pulsed target_present) hold eccentric gaze?

40 deg eccentric, dark (scene off), target flashed 10/20 ms every 1 s, with the
target_strobed FLAG OFF -- so the only mechanism is genuine intermittent
presentation. Compared against the flag-based 'strobe' and a continuous target.
"""
import numpy as np, jax, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_spv_states, ni_net

DT, DEG, TEND = 0.001, 40.0, 12.0
T_STEP, T_ACQ = 0.1, 1.0      # target steps at 0.1 s; continuous until 1.0 s to acquire
PERIOD = 1.0

THETA = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0,
                                sigma_vel=0.0, sigma_slip=0.0), sigma_acc=0.0)
t = np.arange(0.0, TEND, DT); T = len(t)


def pulses(on_ms):
    """1 during acquisition, then `on_ms` ms on at each 1 s boundary."""
    p = np.zeros(T, np.float32)
    p[(t >= T_STEP) & (t < T_ACQ)] = 1.0
    if on_ms:
        phase = np.mod(t - T_ACQ, PERIOD)
        p[(t >= T_ACQ) & (phase < on_ms * 1e-3)] = 1.0
    return p


def run(present, strobe_flag):
    pt = np.zeros((T, 3), np.float32)
    pt[:, 0] = np.where(t < T_STEP, 0.0, np.tan(np.radians(DEG))).astype(np.float32)
    pt[:, 2] = 1.0
    st = simulate(THETA, t,
                  target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32)),
                  target_strobed_array=np.full(T, float(strobe_flag), np.float32),
                  scene_present_array=np.zeros(T, np.float32),          # DARK
                  target_present_array=present,
                  max_steps=int(T * 1.1) + 1000,
                  sim_config=SimConfig(warmup_s=0.0),
                  return_states=True, key=jax.random.PRNGKey(0))
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    spv = extract_spv_states(st, t, eye='version')[:, 0]
    return eye, spv, ni_net(st)[:, 0]


CONDS = [
    ('continuous target (dark)',   pulses(0) * 0 + np.where(t >= T_STEP, 1, 0).astype(np.float32), 0),
    ('FLAG strobe (present=1)',    np.where(t >= T_STEP, 1, 0).astype(np.float32),                 1),
    ('REAL pulse 20 ms / 1 s',     pulses(20),                                                     0),
    ('REAL pulse 10 ms / 1 s',     pulses(10),                                                     0),
]

fig, axes = plt.subplots(len(CONDS), 1, figsize=(14, 3.0 * len(CONDS)), sharex=True)
print(f'{"condition":26} {"eye@2s":>8} {"eye@end":>8} {"net drift":>10} {"SPV 2-12s":>10}  '
      f'{"interflash drift":>16}')
for ax, (label, present, flag) in zip(axes, CONDS):
    eye, spv, ni = run(present, flag)
    w = t >= 2.0
    net = (eye[-1] - eye[2000]) / (t[-1] - 2.0)
    s = spv[w][np.isfinite(spv[w])]
    # Open-loop drift: last 700 ms of each inter-flash interval (well clear of the
    # flash-triggered corrective saccade), averaged over intervals.
    seg = []
    for k in range(int(T_ACQ) + 1, int(TEND) - 1):
        a, b = int((k + 0.25) / DT), int((k + 0.95) / DT)
        seg.append((eye[b] - eye[a]) / (t[b] - t[a]))
    print(f'{label:26} {eye[2000]:8.3f} {eye[-1]:8.3f} {net:+10.3f} {np.mean(s):+10.3f}  '
          f'{np.mean(seg):+16.3f}')
    ax.axhline(DEG, color='k', lw=0.6, ls=':')
    ax.plot(t, eye, color='#2166ac', lw=1.2, label='eye (version)')
    ax.plot(t, ni, color='#4dac26', lw=0.9, ls='--', alpha=0.8, label='NI')
    ax.fill_between(t, DEG - 3, DEG + 3, where=present > 0.5, color='#d6604d',
                    alpha=0.35, step='mid', label='target visible')
    ax.set_title(f'{label}   (net {net:+.3f} deg/s, interflash {np.mean(seg):+.3f} deg/s)',
                 fontsize=10)
    ax.set_ylabel('Eye yaw (deg)'); ax.grid(True, alpha=0.15); ax.legend(fontsize=7, loc='lower left')
    ax.set_ylim(DEG - 4, DEG + 4)
axes[-1].set_xlabel('Time (s)')
fig.suptitle(f'Real strobe vs strobe-flag — {DEG:.0f}° eccentric gaze holding in the DARK',
             fontsize=12, fontweight='bold')
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = __file__.replace('test_real_strobe.py', 'real_strobe.png')
fig.savefig(out, dpi=110)
print('\nfigure ->', out)
