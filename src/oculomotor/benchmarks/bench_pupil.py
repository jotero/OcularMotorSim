"""Pupil benchmarks — HEALTHY light reflex + near response.

The two pupils (L, R) are independent iris plants with rate-asymmetric dynamics
(fast constriction, slow re-dilation). The light reflex is consensual (both
pupils driven by summed retinal input).

Panels (healthy function only — pupil LESIONS live in bench_clinical_pupil):
1. Light reflex: dark → light → dark scene step. Fast constriction (sphincter)
   on light onset, slow re-dilation in the dark — the hallmark asymmetry.
2. Near response: target far → near under a dim room; accommodation rises and
   the pupil constricts with it (near-triad miosis).

Usage:
    python -X utf8 -m oculomotor.benchmarks.bench_pupil
    python -X utf8 -m oculomotor.benchmarks.bench_pupil --show
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

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, simulate, with_brain, with_sensory,
)
from oculomotor.sim import kinematics as km
from oculomotor.analysis import pupil_size, luminance, ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
KEY  = jax.random.PRNGKey(0)

# Noiseless params for clean pupil traces (canal/pos/vel noise + SG accumulator
# diffusion all off) — this is a debug/figure bench (see CLAUDE.md).
PARAMS = with_brain(
    with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0, sigma_slip=0.0),
    sigma_acc=0.0,
)

_C_L, _C_R = '#8B4513', '#d98b5f'   # left / right pupil colours

SECTION = dict(
    id='pupil',
    title='9. Pupil',
    description=(
        'Two independent iris plants (per eye) with rate-asymmetric dynamics '
        '(fast constriction ~0.3 s, slow re-dilation ~1 s). Consensual light '
        'reflex (afferent luminance → pretectum → both Edinger-Westphal nuclei) '
        'plus near-triad constriction (accommodation-linked). '
        '(Pupil lesions — CN III / RAPD / Argyll Robertson / Horner — are in the '
        'clinical benchmarks, section F.)'
    ),
)


# ── Panel 1: pupillary light reflex (fast constriction, slow dilation) ─────────

def _light_reflex(show):
    T_ON, T_OFF, TOTAL = 3.0, 9.0, 14.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    scene = np.zeros(T, dtype=np.float32)
    scene[(t >= T_ON) & (t < T_OFF)] = 1.0

    st = simulate(PARAMS, t, scene_present_array=scene, return_states=True, key=KEY)
    pupil = pupil_size(st)[:, 0]          # both eyes equal (healthy) → left
    lum   = luminance(st).mean(axis=1)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Pupillary light reflex: fast constriction, slow re-dilation',
                 fontsize=11, fontweight='bold')
    ax.axvspan(T_ON, T_OFF, color='#fff3b0', alpha=0.5, lw=0, label='lights on')
    ax.plot(t, pupil, color=_C_L, lw=2.0, label='Pupil diameter')
    ax_fmt(ax, ylabel='Pupil diameter (mm)', xlabel='Time (s)', ylim=(1.5, 9.0))

    axr = ax.twinx()
    axr.plot(t, lum, color='#c0a000', lw=1.2, ls='--', label='Afferent luminance')
    axr.set_ylabel('Afferent luminance', fontsize=8, color='#8a7400')
    axr.set_ylim(-0.05, 1.2); axr.tick_params(labelsize=7, colors='#8a7400')

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = axr.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=7, loc='center right')

    path, rp = utils.save_fig(fig, 'pupil_light_reflex', show=show, params=PARAMS,
                              conditions='Dark → lit → dark scene step (target present throughout)')
    return utils.fig_meta(path, rp,
        title='Pupillary light reflex (constriction/dilation asymmetry)',
        description='Scene steps dark→light→dark. The pupil constricts quickly on light '
                    'onset (sphincter) and re-dilates slowly in the dark.',
        expected='Fast miosis to light (τ ≈ tau_pupil_constrict ~0.3 s); slow mydriasis '
                 '(τ ≈ tau_pupil_dilate ~1 s) — visibly asymmetric.',
        citation='Loewenfeld (1993); Ellis (1981)',
    )


# ── Panel 2: near-response pupil constriction ─────────────────────────────────

def _near_response(show):
    T_NEAR, TOTAL = 3.0, 10.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    p_far  = np.array([0.0, 0.0, 3.0]);  p_near = np.array([0.0, 0.0, 0.33])
    pt = np.tile(p_far, (T, 1)); pt[t >= T_NEAR] = p_near
    scene = np.full(T, 0.4, dtype=np.float32)   # dim room so near miosis clears the floor

    st = simulate(PARAMS, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=scene, return_states=True, key=KEY)
    pupil = pupil_size(st)[:, 0]
    acc   = np.array(st.acc_plant[:, 0])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Near-response pupil constriction (dim room, target 3 m → 33 cm)',
                 fontsize=11, fontweight='bold')
    ax.axvline(T_NEAR, color='gray', lw=0.8, ls=':')
    ax.plot(t, pupil, color=_C_L, lw=2.0, label='Pupil diameter')
    ax_fmt(ax, ylabel='Pupil diameter (mm)', xlabel='Time (s)', ylim=(2.0, 7.0))

    axr = ax.twinx()
    axr.plot(t, acc, color=utils.C['eye'], lw=1.4, ls='--', label='Accommodation')
    axr.set_ylabel('Accommodation (D)', fontsize=8, color=utils.C['eye'])
    axr.set_ylim(-0.2, 4.0); axr.tick_params(labelsize=7, colors=utils.C['eye'])

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = axr.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=7, loc='center right')

    path, rp = utils.save_fig(fig, 'pupil_near_response', show=show, params=PARAMS,
                              conditions='Dim lit room (scene_present=0.4), midline target 3 m → 0.33 m')
    return utils.fig_meta(path, rp,
        title='Near-response pupil constriction',
        description='Target far→near. Accommodation rises (right axis) and the pupil '
                    'constricts with it — the near-triad pupil miosis.',
        expected='Pupil constricts ~1 mm as accommodation rises to ~3 D.',
        citation='Myers & Stark (1990)',
    )


def run(show=SHOW):
    print('\n=== Pupil (light reflex, near response) ===')
    print('  1/2  pupillary light reflex …');   f1 = _light_reflex(show)
    print('  2/2  near-response constriction …'); f2 = _near_response(show)
    return [f1, f2]


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f['title'])
    if SHOW:
        plt.show()
