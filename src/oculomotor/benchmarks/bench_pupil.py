"""Pupil benchmarks — pupillary light reflex, near response, and lesions.

Panels
------
1. Light reflex: dark → light → dark scene step. The pupil constricts when the
   room lights come on (miosis, τ ≈ tau_pupil) and re-dilates in the dark
   (mydriasis). Afferent luminance is overlaid.
2. Near response: target steps far → near under a dimly lit room. Accommodation
   rises and the pupil constricts together (the near-triad pupil miosis).
3. Lesions: a light step compared across the four wired pupil pathologies —
   CN III palsy (blown pupil), RAPD (weak afferent light reflex), Argyll
   Robertson (light-near dissociation), and Horner (sympathetic miosis).

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
    PARAMS_DEFAULT, simulate, with_brain, with_sensory, with_cn3_palsy, with_horner,
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

SECTION = dict(
    id='pupil',
    title='Pupil — light reflex + near response',
    description=(
        'Pupillary light reflex (afferent luminance → Edinger-Westphal → iris '
        'sphincter) and the near-triad pupil constriction (accommodation-linked). '
        'Pupil diameter (mm) is a first-order iris plant (tau_pupil ≈ 0.4 s) driven '
        'by a dark baseline minus light- and near-reflex constriction. Lesion panel '
        'exercises the CN III (blown pupil), afferent (RAPD), pretectal '
        '(light-near dissociation), and sympathetic (Horner) pathways.'
    ),
)


# ── Panel 1: pupillary light reflex ───────────────────────────────────────────

def _light_reflex(show):
    """Dark → light → dark scene step; pupil constricts to light, dilates in dark."""
    T_ON, T_OFF, TOTAL = 3.0, 9.0, 14.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    scene = np.zeros(T, dtype=np.float32)
    scene[(t >= T_ON) & (t < T_OFF)] = 1.0

    st = simulate(PARAMS, t, scene_present_array=scene, return_states=True, key=KEY)
    pupil = pupil_size(st)
    lum   = luminance(st)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Pupillary light reflex: dark → light → dark',
                 fontsize=11, fontweight='bold')
    ax.axvspan(T_ON, T_OFF, color='#fff3b0', alpha=0.5, lw=0, label='lights on')
    ax.plot(t, pupil, color='#8B4513', lw=2.0, label='Pupil diameter')
    ax_fmt(ax, ylabel='Pupil diameter (mm)', xlabel='Time (s)', ylim=(1.5, 8.0))

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
        title='Pupillary light reflex',
        description='Scene steps dark→light→dark. Pupil constricts on light onset and '
                    're-dilates in the dark; afferent luminance (right axis) drives it.',
        expected='Miosis to light (constriction ~3–4 mm) with τ ≈ tau_pupil (~0.4 s); '
                 'mydriasis back to baseline in the dark.',
        citation='Loewenfeld (1993); McDougal & Gamlin (2015)',
    )


# ── Panel 2: near-response pupil constriction ─────────────────────────────────

def _near_response(show):
    """Target far → near under a dim lit room; accommodation + pupil constrict."""
    T_NEAR, TOTAL = 3.0, 10.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    p_far  = np.array([0.0, 0.0, 3.0])   # 3 m  (0.33 D)
    p_near = np.array([0.0, 0.0, 0.33])  # 33 cm (3 D)
    pt = np.tile(p_far, (T, 1))
    pt[t >= T_NEAR] = p_near
    # Dim room (scene_present = 0.4) so the near miosis is visible above the light-
    # reflex floor (full light saturates the pupil to pupil_min).
    scene = np.full(T, 0.4, dtype=np.float32)

    st = simulate(PARAMS, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=scene, return_states=True, key=KEY)
    pupil = pupil_size(st)
    acc   = np.array(st.acc_plant[:, 0])

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.suptitle('Near-response pupil constriction (dim room, target 3 m → 33 cm)',
                 fontsize=11, fontweight='bold')
    ax.axvline(T_NEAR, color='gray', lw=0.8, ls=':')
    ax.plot(t, pupil, color='#8B4513', lw=2.0, label='Pupil diameter')
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
        description='Target steps far→near. Accommodation rises (right axis) and the '
                    'pupil constricts with it — the near-triad pupil miosis.',
        expected='Pupil constricts ~1 mm as accommodation rises to ~3 D; both track the '
                 'near step with the accommodation/iris plant TCs.',
        citation='Myers & Stark (1990)',
    )


# ── Panel 3: pupil lesions ────────────────────────────────────────────────────

def _lesions(show):
    """Light step compared across the four wired pupil lesion pathways."""
    T_ON, TOTAL = 3.0, 10.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    scene = np.zeros(T, dtype=np.float32)
    scene[t >= T_ON] = 1.0

    conditions = [
        ('Healthy',                    PARAMS,                                        '#333333', '-'),
        ('CN III palsy (blown)',       with_cn3_palsy(PARAMS, side='right'),          '#b2182b', '-'),
        ('RAPD (afferent 0.3)',        with_brain(PARAMS, g_pupil_afferent=0.3),      '#ef8a62', '--'),
        ('Argyll Robertson (light 0)', with_brain(PARAMS, g_pupil_light_reflex=0.0),  '#2166ac', '-.'),
        ('Horner (baseline −2.5 mm)',  with_horner(PARAMS, miosis_mm=2.5),            '#1a9850', ':'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    fig.suptitle('Pupil lesions: response to a light step (dark → light at 3 s)',
                 fontsize=11, fontweight='bold')
    ax.axvspan(T_ON, TOTAL, color='#fff3b0', alpha=0.4, lw=0)
    for label, params, color, ls in conditions:
        st = simulate(params, t, scene_present_array=scene, return_states=True, key=KEY)
        ax.plot(t, pupil_size(st), color=color, ls=ls, lw=1.8, label=label)
    ax_fmt(ax, ylabel='Pupil diameter (mm)', xlabel='Time (s)', ylim=(1.5, 8.0))
    ax.legend(fontsize=7, loc='center right', ncol=1)

    path, rp = utils.save_fig(fig, 'pupil_lesions', show=show, params=PARAMS,
                              conditions='Dark → lit step; five lesion conditions overlaid')
    return utils.fig_meta(path, rp,
        title='Pupil lesions',
        description='Light step across healthy, CN III palsy (blown pupil, shared with the '
                    'oculomotor eye-muscle lesion), RAPD, Argyll Robertson (light-near '
                    'dissociation), and Horner (sympathetic miosis).',
        expected='Healthy constricts to light; CN III blown pupil + Argyll Robertson stay '
                 'dilated (no light reflex); RAPD constricts weakly; Horner sits at a small '
                 'baseline but still reacts.',
        citation='Loewenfeld (1993); McDougal & Gamlin (2015)',
    )


def run(show=SHOW):
    print('\n=== Pupil (light reflex + near response) ===')
    print('  1/3  pupillary light reflex …')
    f1 = _light_reflex(show)
    print('  2/3  near-response constriction …')
    f2 = _near_response(show)
    print('  3/3  lesions …')
    f3 = _lesions(show)
    return [f1, f2, f3]


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f['title'])
    if SHOW:
        plt.show()
