"""
Epley maneuver benchmark — right posterior-canal BPPV.

Head movement sequence based on:
  Arán-Tapia et al. (2024). Numerical Simulations of the Epley Maneuver
  With Clinical Implications. Ear & Hearing, 45, 1033–1044.

BPPV pathophysiology:
  Displaced otoconia in the right posterior (RPC) semicircular canal create
  gravity-driven ampullopetal endolymph flow during the Dix-Hallpike position.
  This is approximated by injecting a decaying excitatory signal along the
  RPC's sensitive axis [yaw=0, pitch=−S, roll=+S] during hold phases.

Expected nystagmus:
  DH hold:      upbeat + right-torsional (VOR compensates RPC excitation)
  L-turn hold:  nystagmus converts as debris transits the long arm
  Face-down:    brief reversal as debris exits the canal
  After sit-up: nystagmus extinguished (otoconia in utricle)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from oculomotor.sim.simulator import simulate, default_params, with_sensory, with_brain
from oculomotor.sim.kinematics import build_kinematics
import oculomotor.analysis as an

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)
DT   = 0.001   # s — solver step

# ── Canal geometry (from canal.py) ───────────────────────────────────────────
# RPC = canal index 5,  LARP plane, orientation [0, −S, +S] in [yaw, pitch, roll]
# Excitatory for RPC → VOR response [0, +S, −S] → upbeat + right torsion
_S       = 2**-0.5           # sin(45°) ≈ 0.7071
RPC_AXIS = np.array([0.0, -_S, +_S])   # unit vector; excitatory direction for RPC

CANAL_NAMES  = ['LHC', 'LAC', 'LPC', 'RHC', 'RAC', 'RPC']
CANAL_COLORS = ['#4a90e2', '#5bab5b', '#e2724a',
                '#a0c4f0', '#90cc90', '#e05050']

# ── Epley phase sequence (right posterior-canal BPPV) ─────────────────────────
# (label, duration_s, yaw_vel °/s, pitch_vel °/s, roll_vel °/s)
# Sign convention: yaw+ = rightward, pitch+ = nose up, roll+ = top-of-head left
PHASES = [
    ('Upright rest',         3.0,    0,      0,      0),
    ('Turn head\nright 45°', 2.0, +22.5,     0,      0),  # yaw 45° right
    ('Lie back\n(DH)',        3.0,    0,   +30.0,     0),  # pitch 90°
    ('Head\nextension',       1.0,    0,   +20.0,     0),  # pitch 20° past supine
    ('Hold DH\n(nystagmus)', 30.0,    0,      0,      0),  # BPPV nystagmus phase
    ('Turn head\nleft 90°',  3.0, -30.0,     0,      0),  # yaw 90° left
    ('Hold L-turn',         30.0,    0,      0,      0),  # debris transit
    ('Roll body\nleft 90°',   3.0,    0,      0,   +30.0), # roll 90° face-down
    ('Hold\nface-down',      15.0,    0,      0,      0),  # debris exits canal
    ('Sit up',                3.0,    0,   -30.0,     0),  # pitch −90°
    ('Nose down\n20°',        1.0,    0,   -20.0,     0),  # un-extend
    ('Hold\nseated',         10.0,    0,      0,      0),  # brief rest
    ('Roll back',             3.0,    0,      0,   -30.0), # roll back upright
    ('Centre\nhead',          2.0, +22.5,     0,      0),  # yaw back to 0°
    ('Final rest',            5.0,    0,      0,      0),
]

# Phase index → (BPPV amplitude °/s, decay TC s, onset delay s within phase)
# Model: otoconia-driven ampullopetal endolymph flow along RPC axis
BPPV_EVENTS = {
    4: (55.0, 14.0, 2.5),   # DH hold — strong; onset latency 2–5 s is typical
    6: (25.0,  7.0, 1.0),   # L-turn hold — moderate; debris transiting
    8: (18.0,  5.0, 0.5),   # Face-down — weaker; debris exiting
   11: ( 8.0,  3.0, 0.5),   # Post-Epley seat hold — mild residual
}

# ── Build head velocity array ─────────────────────────────────────────────────

def _phase_edges():
    edges = [0.0]
    for _, dur, *_ in PHASES:
        edges.append(edges[-1] + dur)
    return edges


def build_head_vel(add_bppv: bool = True):
    """Return (t, hv) where hv is (T, 3) [yaw, pitch, roll] deg/s."""
    edges = _phase_edges()
    segments = []
    for _, dur, yv, pv, rv in PHASES:
        n   = max(1, round(dur / DT))
        seg = np.zeros((n, 3))
        seg[:, 0] = yv
        seg[:, 1] = pv
        seg[:, 2] = rv
        segments.append(seg)

    hv = np.concatenate(segments, axis=0)
    t  = np.arange(len(hv)) * DT

    if add_bppv:
        for ph_idx, (amp, tau, onset) in BPPV_EVENTS.items():
            t0    = edges[ph_idx] + onset
            t1    = edges[ph_idx + 1]
            mask  = (t >= t0) & (t < t1)
            tloc  = t[mask] - t0
            decay = amp * np.exp(-tloc / tau)
            hv[mask] += decay[:, None] * RPC_AXIS[None, :]

    return t, hv


# ── Canal afferents from x2 state ─────────────────────────────────────────────

def _canal_afferents(states):
    """Return (T, 6) canal afferent signals (0 at rest, ± deg/s)."""
    from oculomotor.models.sensory_models.canal import FLOOR as _F, _SOFTNESS as _K
    try:
        from scipy.special import softplus as _sp
    except ImportError:
        def _sp(x):
            return np.where(x > 30, x, np.log1p(np.exp(np.clip(x, -500, 30))))
    x2 = np.array(states.sensory.canal.x2)   # (T, 6)
    k  = float(_K)
    f  = float(_F)
    # formula gives ~0 at x2=0 (deviation from resting discharge f)
    return -f + _sp(k * (x2 + f)) / k + _sp(k * (x2 - f)) / k


# ── Parameters — noiseless so phase structure is clear ────────────────────────

PARAMS = with_sensory(default_params(), sigma_canal=0, sigma_pos=0, sigma_vel=0)
PARAMS = with_brain(PARAMS, sigma_acc=0)

# ── Run simulations ───────────────────────────────────────────────────────────

print('Building head velocity arrays …')
t_h, hv_h = build_head_vel(add_bppv=False)
t_b, hv_b = build_head_vel(add_bppv=True)

km_h = build_kinematics(t_h, rot_vel=hv_h)
km_b = build_kinematics(t_b, rot_vel=hv_b)

ones_h = np.ones(len(t_h))
ones_b = np.ones(len(t_b))

print('Running healthy VOR (no BPPV) …')
states_h = simulate(PARAMS, t_h,
                    head=km_h,
                    scene_present_array=ones_h,
                    target_present_array=ones_h,
                    return_states=True, key=None)

print('Running BPPV simulation …')
states_b = simulate(PARAMS, t_b,
                    head=km_b,
                    scene_present_array=ones_b,
                    target_present_array=ones_b,
                    return_states=True, key=None)

print('Extracting signals …')
# eye position (cyclopean) → velocity
ep_h = (np.array(states_h.plant.left) + np.array(states_h.plant.right)) / 2.0
ep_b = (np.array(states_b.plant.left) + np.array(states_b.plant.right)) / 2.0
ev_h = np.gradient(ep_h, DT, axis=0)
ev_b = np.gradient(ep_b, DT, axis=0)

vs_h = an.vs_net(states_h)   # (T, 3) velocity storage
vs_b = an.vs_net(states_b)

ca_h = _canal_afferents(states_h)   # (T, 6)
ca_b = _canal_afferents(states_b)

# ── Cumulative head position (for reference) ──────────────────────────────────
hp_b = np.cumsum(hv_b * DT, axis=0)   # [yaw, pitch, roll] deg

# ── Figure ────────────────────────────────────────────────────────────────────
EDGES = _phase_edges()
T = t_b

SHADE  = ['#e8f0ff', '#fff0e8']
C_YAW  = '#4a90e2'
C_PTC  = '#e25e4a'
C_ROL  = '#4ab55e'
C_TOR  = '#9b59b6'

fig, axes = plt.subplots(5, 1, figsize=(16, 14), sharex=True,
                          gridspec_kw={'hspace': 0.55, 'top': 0.88})
fig.suptitle(
    'Epley Maneuver  —  Right Posterior-Canal BPPV\n'
    'Head movements: Arán-Tapia et al. (2024)  •  '
    'BPPV modelled as decaying excitatory RPC signal simulating otoconia-driven endolymph flow\n'
    'Solid = BPPV patient    Dashed = healthy VOR',
    fontsize=10
)

# Phase shading + labels (top panel only for labels)
for ax in axes:
    for i, (_, dur, *_) in enumerate(PHASES):
        ax.axvspan(EDGES[i], EDGES[i+1], alpha=0.30,
                   color=SHADE[i % 2], zorder=0)

# Phase labels on top panel
for i, (lbl, dur, *_) in enumerate(PHASES):
    if dur < 1.5:
        continue
    mid = (EDGES[i] + EDGES[i+1]) / 2.0
    axes[0].text(mid, 1.01, lbl, transform=axes[0].get_xaxis_transform(),
                 ha='center', va='bottom', fontsize=6.5, alpha=0.8,
                 linespacing=1.2)

def _fmt(ax, ylabel, title, lw_min=None):
    ax.axhline(0, color='k', lw=0.4)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=9, pad=3)
    ax.tick_params(labelsize=8)
    if lw_min is not None:
        lo, hi = ax.get_ylim()
        if hi - lo < lw_min:
            mid = (lo + hi) / 2
            ax.set_ylim(mid - lw_min/2, mid + lw_min/2)

# ── 1. Head angular velocity ──────────────────────────────────────────────────
ax = axes[0]
ax.plot(T, hv_b[:, 0], color=C_YAW, lw=0.9, label='Yaw')
ax.plot(T, hv_b[:, 1], color=C_PTC, lw=0.9, label='Pitch')
ax.plot(T, hv_b[:, 2], color=C_ROL, lw=0.9, label='Roll')
ax.legend(fontsize=8, loc='upper right', ncol=3)
_fmt(ax, 'Head vel\n(deg/s)', 'Head angular velocity  (maneuver steps + BPPV phantom canal input)', 10)

# ── 2. Canal afferents — all 6 channels ───────────────────────────────────────
ax = axes[1]
for i, (name, col) in enumerate(zip(CANAL_NAMES, CANAL_COLORS)):
    is_larp = name in ('RPC', 'LAC')   # LARP coplanar pair — bold
    lw = 1.6 if is_larp else 0.7
    ls = '-'  if is_larp else '-'
    ax.plot(T, ca_b[:, i], color=col, lw=lw, ls=ls, label=name, alpha=0.95)
ax.legend(fontsize=8, loc='upper right', ncol=6)
_fmt(ax, 'Canal afferent\n(dev. from rest,\ndeg/s)',
     'Canal afferents  (RPC bold red = right posterior; LAC bold green = coplanar LARP pair)', 10)

# ── 3. Velocity storage ───────────────────────────────────────────────────────
ax = axes[2]
kw_h = dict(lw=0.7, alpha=0.55, ls='--')
kw_b = dict(lw=1.2)
ax.plot(t_h, vs_h[:, 0], color=C_YAW, **kw_h)
ax.plot(t_h, vs_h[:, 1], color=C_PTC, **kw_h)
ax.plot(t_h, vs_h[:, 2], color=C_ROL, **kw_h)
ax.plot(T,   vs_b[:, 0], color=C_YAW, **kw_b, label='Yaw')
ax.plot(T,   vs_b[:, 1], color=C_PTC, **kw_b, label='Pitch')
ax.plot(T,   vs_b[:, 2], color=C_ROL, **kw_b, label='Roll')
ax.legend(fontsize=8, loc='upper right', ncol=3)
_fmt(ax, 'VS net\n(deg/s)', 'Velocity storage  (left − right net signal)', 10)

# ── 4. Eye velocity — horizontal + vertical ───────────────────────────────────
ax = axes[3]
ax.plot(t_h, ev_h[:, 0], color=C_YAW, **kw_h)
ax.plot(t_h, ev_h[:, 1], color=C_PTC, **kw_h)
ax.plot(T,   ev_b[:, 0], color=C_YAW, **kw_b, label='Horiz (yaw)')
ax.plot(T,   ev_b[:, 1], color=C_PTC, **kw_b, label='Vert  (pitch)')
ax.legend(fontsize=8, loc='upper right', ncol=2)
_fmt(ax, 'Eye vel\n(deg/s)',
     'Eye velocity — horizontal & vertical  (fast phases = VOR corrective saccades)', 20)

# ── 5. Torsional eye velocity — most pathognomonic for posterior-canal BPPV ───
ax = axes[4]
ax.plot(t_h, ev_h[:, 2], color=C_TOR, **kw_h, label='Torsion healthy')
ax.plot(T,   ev_b[:, 2], color=C_TOR, **kw_b, label='Torsion BPPV')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlabel('Time (s)', fontsize=9)
_fmt(ax, 'Tors. eye vel\n(deg/s)',
     'Torsional eye velocity  (most pathognomonic — upbeat+right-torsional in DH hold)', 15)

for ax in axes:
    ax.set_xlim(T[0], T[-1])

out_path = OUT / 'epley_maneuver.png'
fig.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved → {out_path}')
plt.close(fig)
