"""Trace the tau_i=2.5 / K_cereb_fl=0.88 instability across visual conditions.

Question: is the oscillation intrinsic to the integrator, or does it need the
visual position-feedback loop? Same two parameter sets under four conditions
that differ ONLY in what visual information reaches the eye.
"""
import numpy as np, jax, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_spv_states, ni_net, ni_null

DT, DEG, TEND = 0.001, 20.0, 8.0
T_STEP, ACQ, PERIOD, ON_MS = 0.1, 1.0, 1.0, 20.0

BASE = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0, sigma_pos=0,
                               sigma_vel=0, sigma_slip=0), sigma_acc=0)
NEW  = with_brain(BASE, tau_i=2.5,  K_cereb_fl=0.88)   # the unstable candidate
OLD  = with_brain(BASE, tau_i=25.0, K_cereb_fl=1.0)    # current default (reverted to)

t = np.arange(0.0, TEND, DT); T = len(t)
pt = np.zeros((T, 3), np.float32)
pt[:, 0] = np.where(t < T_STEP, 0.0, np.tan(np.radians(DEG))).astype(np.float32)
pt[:, 2] = 1.0

on_always = np.where(t >= T_STEP, 1.0, 0.0).astype(np.float32)
acq_only  = np.where((t >= T_STEP) & (t < ACQ), 1.0, 0.0).astype(np.float32)
pulsed    = acq_only.copy()
pulsed[(t >= ACQ) & (np.mod(t - ACQ, PERIOD) < ON_MS * 1e-3)] = 1.0

CONDS = [
    ('Continuous target visible (flag off)',      on_always, 0.0),
    ('FLAG strobe (target on, velocity gated)',   on_always, 1.0),
    ('REAL strobe — 20 ms pulse every 1 s',       pulsed,    0.0),
    ('Total darkness after acquisition',          acq_only,  0.0),
]


def run(theta, present, flag):
    st = simulate(theta, t,
                  target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32)),
                  target_strobed_array=np.full(T, flag, np.float32),
                  scene_present_array=np.zeros(T, np.float32),          # DARK throughout
                  target_present_array=present,
                  max_steps=int(T * 1.1) + 1000, sim_config=SimConfig(warmup_s=0.0),
                  return_states=True, key=jax.random.PRNGKey(0))
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    spv = extract_spv_states(st, t, eye='version')[:, 0]
    w = t >= 2.0
    v = spv[w][np.isfinite(spv[w])]
    return eye, np.array(ni_net(st))[:, 0], (float(np.mean(v)) if len(v) else float('nan'))


fig, axes = plt.subplots(len(CONDS), 1, figsize=(14, 2.9 * len(CONDS)), sharex=True)
fig.suptitle(
    f'Gaze holding at {DEG:.0f}°, DARK — where does the tau_i=2.5 / K_fl=0.88 instability appear?\n'
    'red = candidate (tau_i 2.5 s, K_fl 0.88)   ·   blue = current default (tau_i 25 s, K_fl 1.0)',
    fontsize=12, fontweight='bold')

print(f'{"condition":42} {"SPV new":>9} {"SPV old":>9}')
for ax, (label, present, flag) in zip(axes, CONDS):
    eye_n, ni_n, spv_n = run(NEW, present, flag)
    eye_o, ni_o, spv_o = run(OLD, present, flag)
    print(f'{label:42} {spv_n:+9.3f} {spv_o:+9.3f}')
    ax.axhline(DEG, color='k', lw=0.6, ls=':')
    ax.fill_between(t, DEG - 12, DEG + 12, where=present > 0.5, color='#d6604d',
                    alpha=0.13, step='mid', label='target present')
    ax.plot(t, eye_o, color='#2166ac', lw=1.4, label=f'default  (SPV {spv_o:+.2f} deg/s)')
    ax.plot(t, eye_n, color='#d6604d', lw=1.4, label=f'candidate (SPV {spv_n:+.2f} deg/s)')
    ax.set_title(label, fontsize=10)
    ax.set_ylabel('Eye yaw (deg)'); ax.grid(True, alpha=0.15)
    ax.set_ylim(DEG - 12, DEG + 12); ax.legend(fontsize=8, loc='lower left', ncol=3)
axes[-1].set_xlabel('Time (s)')
fig.tight_layout(rect=[0, 0, 1, 0.955])
out = __file__.replace('test_instability_trace.py', 'instability_trace.png')
fig.savefig(out, dpi=110)
print('\nfigure ->', out)
