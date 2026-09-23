"""40 deg eccentric fixation: target CONTINUOUSLY ON vs REAL STROBE.

Current model defaults (tau_i=25, K_cereb_fl=1.0), dark, noiseless — the only
difference between conditions is whether the target stays lit or is flashed.
Motion panels are auto-zoomed to the post-acquisition data so residual activity
is visible rather than being flattened by the acquisition transient.
"""
import numpy as np, jax, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_spv_states, read_brain_decoded
from oculomotor.models.brain_models import perception_cyclopean as pc_mod

DT, DEG, TEND = 0.001, 40.0, 10.0
T_STEP, ACQ, PERIOD, ON_MS = 0.1, 1.0, 1.0, 20.0
ZOOM_FROM = 2.0                      # ignore the acquisition transient when auto-scaling

import sys
# Optional CLI overrides:  compare_on_vs_strobe.py [tau_i] [K_cereb_fl]
TAU_I = float(sys.argv[1]) if len(sys.argv) > 1 else float(PARAMS_DEFAULT.brain.tau_i)
K_FL  = float(sys.argv[2]) if len(sys.argv) > 2 else float(PARAMS_DEFAULT.brain.K_cereb_fl)
TAG   = '' if len(sys.argv) <= 1 else f'_tau{TAU_I:g}_K{K_FL:g}'
THETA = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0, sigma_pos=0,
                                sigma_vel=0, sigma_slip=0), sigma_acc=0,
                   tau_i=TAU_I, K_cereb_fl=K_FL)
t = np.arange(0.0, TEND, DT); T = len(t)
pt = np.zeros((T, 3), np.float32)
pt[:, 0] = np.where(t < T_STEP, 0.0, np.tan(np.radians(DEG))).astype(np.float32)
pt[:, 2] = 1.0

on_always = np.where(t >= T_STEP, 1.0, 0.0).astype(np.float32)
pulsed    = np.where((t >= T_STEP) & (t < ACQ), 1.0, 0.0).astype(np.float32)
pulsed[(t >= ACQ) & (np.mod(t - ACQ, PERIOD) < ON_MS * 1e-3)] = 1.0


def run(present):
    st = simulate(THETA, t,
                  target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32)),
                  target_strobed_array=np.zeros(T, np.float32),   # no flag: real presentation only
                  scene_present_array=np.zeros(T, np.float32),    # DARK
                  target_present_array=present,
                  max_steps=int(T * 1.1) + 1000, sim_config=SimConfig(warmup_s=0.0),
                  return_states=True, key=jax.random.PRNGKey(0))
    ret   = st.sensory.retina_L
    x_cyc = np.array(jax.vmap(pc_mod.to_array)(st.brain.pc))
    eye   = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    return dict(
        eye      = eye,
        eye_vel  = np.gradient(eye, DT),
        spv      = extract_spv_states(st, t, eye='version')[:, 0],
        tpos_ret = np.array(ret.target_pos)[:, -3],
        tvel_ret = np.array(ret.target_vel)[:, -3],
        tvel_cyc = (x_cyc @ np.array(pc_mod.C_vel).T)[:, 0],
        tvis_ret = np.array(ret.target_visible)[:, -1],
        pu_net   = np.array(read_brain_decoded(st, THETA).pu.net)[:, 0],
    )


A = run(on_always)   # target continuously on
B = run(pulsed)      # real strobe
w = t >= ZOOM_FROM


def zoom(*series, pad=0.15, floor=1e-4):
    """Symmetric-ish limits from the post-acquisition data only."""
    lo = min(float(np.nanmin(s[w])) for s in series)
    hi = max(float(np.nanmax(s[w])) for s in series)
    if hi - lo < floor:
        mid = 0.5 * (hi + lo); lo, hi = mid - floor, mid + floor
    m = (hi - lo) * pad
    return lo - m, hi + m


ROWS = [
    ('Eye position (deg)',              'eye',      None,                       False),
    ('Target visible [0,1]',            'tvis_ret', (-0.05, 1.05),              False),
    ('Retinal position error (deg)',    'tpos_ret', None,                       True),
    ('Retinal target MOTION (deg/s)',   'tvel_ret', None,                       True),
    ('Cyclopean target_vel (deg/s)',    'tvel_cyc', None,                       True),
    ('Pursuit net drive (deg/s)',       'pu_net',   None,                       True),
    ('Slow-phase velocity (deg/s)',     'spv',      None,                       True),
]

fig, axes = plt.subplots(len(ROWS), 1, figsize=(15, 2.25 * len(ROWS)), sharex=True)
fig.suptitle(
    f'{DEG:.0f}° eccentric fixation in the DARK — target CONTINUOUSLY ON vs REAL STROBE '
    f'({ON_MS:.0f} ms every {PERIOD:.0f} s)\n'
    f'tau_i={THETA.brain.tau_i:g} s, K_cereb_fl={THETA.brain.K_cereb_fl:g}  '
    f'(tau_eff = tau_i/(1-K) = {TAU_I/max(1-K_FL,1e-9):.0f} s), noiseless.  '
    f'Motion panels auto-zoomed to t > {ZOOM_FROM:g} s so residuals are visible.',
    fontsize=12, fontweight='bold')

for ax, (label, key, ylim, autoz) in zip(axes, ROWS):
    for k in np.where(np.diff(pulsed) > 0)[0]:
        ax.axvline(t[k], color='#d6604d', lw=0.6, alpha=0.30)
    ax.plot(t, A[key], color='#2166ac', lw=1.3, label='target ON (continuous)')
    ax.plot(t, B[key], color='#d6604d', lw=1.3, label='target STROBING (20 ms/1 s)')
    if ylim:
        ax.set_ylim(*ylim)
    elif autoz:
        ax.set_ylim(*zoom(A[key], B[key]))
    ax.set_ylabel(label, fontsize=8); ax.grid(True, alpha=0.15)
    ax.axhline(0, color='k', lw=0.4)
    rng_a = float(np.nanmax(A[key][w]) - np.nanmin(A[key][w]))
    rng_b = float(np.nanmax(B[key][w]) - np.nanmin(B[key][w]))
    ax.set_title(f'{label}   —   post-{ZOOM_FROM:g}s peak-to-peak:  ON {rng_a:.4g}  |  STROBE {rng_b:.4g}',
                 fontsize=9)
axes[0].legend(fontsize=8, loc='upper left')
axes[-1].set_xlabel('Time (s)')
fig.tight_layout(rect=[0, 0, 1, 0.965])
out = __file__.replace('compare_on_vs_strobe.py', f'on_vs_strobe{TAG}.png')
fig.savefig(out, dpi=110)

print(f'{"signal":30} {"ON p2p":>12} {"STROBE p2p":>12}   (t > {ZOOM_FROM:g}s)')
for label, key, _, _ in ROWS:
    print(f'{label:30} {np.nanmax(A[key][w]) - np.nanmin(A[key][w]):12.5f} '
          f'{np.nanmax(B[key][w]) - np.nanmin(B[key][w]):12.5f}')
print('\nfigure ->', out)
