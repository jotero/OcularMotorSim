"""Eyelid benchmarks — spontaneous blinks and lid-follows-vertical-gaze.

The eyelid is a per-eye lid plant driven by a levator/Müller resting posture,
an orbicularis blink command, and a downgaze lid-follow (see eyelid.py /
eyelid_plant.py). Lesions (ptosis via CN III levator nuclear/nerve + Horner
Müller; lagophthalmos via CN VII orbicularis) are modelled but not shown here.

Panels
------
1. Spontaneous blinks: the model's blink train during steady fixation.
2. Vertical lid-follow: the upper-lid closure tracks downgaze depth.

Usage:
    python -X utf8 -m oculomotor.benchmarks.bench_eyelid
    python -X utf8 -m oculomotor.benchmarks.bench_eyelid --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oculomotor.benchmarks import bench_utils as utils

import numpy as np
import jax
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
from oculomotor.analysis import eyelid, ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
KEY  = jax.random.PRNGKey(0)

# Noiseless params for clean traces (this is a debug/figure bench; see CLAUDE.md).
PARAMS = with_brain(
    with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0, sigma_slip=0.0),
    sigma_acc=0.0,
)

SECTION = dict(
    id='eyelid',
    title='10. Eyelid',
    description=(
        'Per-eye upper-lid plant: spontaneous blinks (conjugate, ~15/min) + an '
        'upper lid that follows vertical gaze (downgaze lowers the lid). Fast '
        'closing / slower opening (rate-asymmetric). Ptosis (CN III levator, '
        'nuclear vs nerve; Horner Müller) and lagophthalmos (CN VII orbicularis) '
        'lesions are wired but not shown here.'
    ),
)


# ── Panel 1: spontaneous blinks ───────────────────────────────────────────────

def _blinks(show):
    """Spontaneous random-blink train during steady fixation (model output)."""
    t = np.arange(0.0, 30.0, DT); T = len(t)
    st = simulate(PARAMS, t, scene_present_array=np.ones(T), return_states=True, key=KEY)
    eyelid_L = eyelid(st)[:, 0]

    above    = eyelid_L > 0.5
    n_blinks = int(np.sum(np.diff(above.astype(int)) == 1))
    rate     = n_blinks / (t[-1] - t[0]) * 60.0

    fig, ax = plt.subplots(figsize=(10, 4.0))
    fig.suptitle('Spontaneous blinks (steady fixation)', fontsize=11, fontweight='bold')
    ax.plot(t, eyelid_L, color='#1a1a2e', lw=1.5, label='Eyelid closure (both eyes)')
    ax.axhline(1.0, color='#bbbbbb', lw=0.7, ls='--')
    ax_fmt(ax, ylabel='Eyelid closure (0=open, 1=closed)', xlabel='Time (s)', ylim=(-0.05, 1.1))
    ax.text(0.01, 0.97, f'{n_blinks} blinks in {t[-1]-t[0]:.0f} s  ≈ {rate:.0f}/min',
            transform=ax.transAxes, fontsize=8, va='top',
            bbox=dict(boxstyle='round', fc='#f0f0f5', ec='#ccc'))
    ax.legend(fontsize=7, loc='upper right')

    path, rp = utils.save_fig(fig, 'eyelid_blinks', show=show, params=PARAMS,
                              conditions='Steady fixation; spontaneous blink train (eyelid model)')
    return utils.fig_meta(path, rp,
        title='Spontaneous blinks',
        description='Model eyelid output during steady fixation — the spontaneous blink '
                    'train (conjugate): a smooth 0→1→0 pulse, rate ~15/min.',
        expected='~12–20 blinks/min, each ~150 ms; conjugate across both eyes.',
        citation='eyelid model (eyelid.py → eyelid_plant.py)',
    )


# ── Panel 2: eyelid follows vertical eye movement ─────────────────────────────

def _vertical_follow(show):
    """Eyelid closure follows vertical gaze — graded downgaze steps lower the lid."""
    T_A, T_B, T_C, TOTAL = 2.0, 5.0, 8.0, 11.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    y = np.zeros(T)
    y[(t >= T_A) & (t < T_B)] = np.tan(np.radians(-15.0))   # look down ~15°
    y[(t >= T_B) & (t < T_C)] = np.tan(np.radians(-30.0))   # look down ~30°
    pt = np.stack([np.zeros(T), y, np.ones(T)], axis=1)     # (T, 3) m

    # Blinks off (eyelid_blink_rate=0) to isolate the lid-follow coupling.
    p_noblink = with_brain(PARAMS, eyelid_blink_rate=0.0)
    st = simulate(p_noblink, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), return_states=True, key=KEY)
    eye_pitch = np.array(st.plant.left[:, 1])               # left eye pitch (deg, + = up)
    eyelid_L  = eyelid(st)[:, 0]                            # model lid closure (left)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Eyelid follows vertical gaze (downgaze lowers the lid)',
                 fontsize=11, fontweight='bold')
    ax.plot(t, eyelid_L, color='#1a1a2e', lw=2.0, label='Eyelid closure')
    ax_fmt(ax, ylabel='Eyelid closure (0=open, 1=closed)', xlabel='Time (s)', ylim=(-0.05, 0.6))

    axr = ax.twinx()
    axr.plot(t, eye_pitch, color=utils.C['eye'], lw=1.4, ls='--', label='Eye pitch')
    axr.axhline(0.0, color='#ddd', lw=0.7)
    axr.set_ylabel('Eye pitch (deg, + up / − down)', fontsize=8, color=utils.C['eye'])
    axr.set_ylim(-35, 10); axr.tick_params(labelsize=7, colors=utils.C['eye'])

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = axr.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=7, loc='upper right')

    path, rp = utils.save_fig(fig, 'eyelid_vertical_follow', show=show, params=PARAMS,
                              conditions='Midline target stepping down ~15° then ~30°, then centre')
    return utils.fig_meta(path, rp,
        title='Eyelid follows vertical eye movement',
        description='The eye makes graded downward movements (~15° then ~30°); the model '
                    'upper-lid closure tracks downgaze depth (lid lowers as the eye looks '
                    'further down). Blinks disabled here to isolate the gaze-follow.',
        expected='Closure ~0 at centre / upgaze, rising with downgaze depth (graded).',
        citation='eyelid model (eyelid.py → eyelid_plant.py)',
    )


def run(show=SHOW):
    print('\n=== Eyelid (spontaneous blinks + vertical lid-follow) ===')
    print('  1/2  spontaneous blinks …');   f1 = _blinks(show)
    print('  2/2  vertical lid-follow …');   f2 = _vertical_follow(show)
    return [f1, f2]


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f['title'])
    if SHOW:
        plt.show()
