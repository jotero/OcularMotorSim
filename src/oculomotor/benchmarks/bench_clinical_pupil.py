"""Clinical pupillary lesion benchmarks — anisocoria + lesion survey.

Light-reflex signatures of pupil lesions (moved here from the healthy `bench_pupil`
section, which now shows only intact light-reflex + near-response function):

1. Anisocoria — LEFT CN III palsy: the left pupil is fixed and dilated (blown)
   while the right pupil still reacts to light. Healthy (equal pupils) overlaid.
2. Lesion survey (left pupil): healthy vs CN III blown, RAPD, Argyll Robertson
   (light-near dissociation), and Horner (sympathetic miosis).

Usage:
    python -X utf8 -m oculomotor.benchmarks.bench_clinical_pupil
    python -X utf8 -m oculomotor.benchmarks.bench_clinical_pupil --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oculomotor.benchmarks import bench_clinical_utils as utils

import numpy as np
import jax
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, simulate, with_brain, with_sensory, with_cn3_palsy, with_horner,
)
from oculomotor.analysis import pupil_size, ax_fmt

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
    id='clin_pupil',
    title='F. Pupillary Lesions',
    description=(
        'Light-reflex signatures of pupil lesions: efferent parasympathetic (CN III '
        'palsy → fixed dilated pupil, anisocoria), afferent (relative afferent '
        'pupillary defect, RAPD), central pretectal (Argyll Robertson light-near '
        'dissociation), and sympathetic (Horner miosis). Healthy overlaid for reference. '
        '(Intact light reflex + near response are in the healthy "Pupil" section.)'
    ),
)


# ── Anisocoria — LEFT CN III palsy (per-eye) ──────────────────────────────────

def _anisocoria(show):
    T_ON, TOTAL = 3.0, 10.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    scene = np.zeros(T, dtype=np.float32); scene[t >= T_ON] = 1.0

    healthy = pupil_size(simulate(PARAMS, t, scene_present_array=scene,
                                  return_states=True, key=KEY))
    cn3 = pupil_size(simulate(with_cn3_palsy(PARAMS, side='left'), t,
                              scene_present_array=scene, return_states=True, key=KEY))

    fig, ax = plt.subplots(figsize=(10, 4.8))
    fig.suptitle('Anisocoria — left CN III palsy (pupil-involving), light step at 3 s',
                 fontsize=11, fontweight='bold')
    ax.axvspan(T_ON, TOTAL, color='#fff3b0', alpha=0.4, lw=0)
    ax.plot(t, healthy[:, 0], color='#999999', lw=1.2, ls=':', label='Healthy (both eyes)')
    ax.plot(t, cn3[:, 0], color=_C_L, lw=2.2, label='Left pupil (CN III palsy — blown)')
    ax.plot(t, cn3[:, 1], color=_C_R, lw=2.0, ls='--', label='Right pupil (reactive)')
    ax_fmt(ax, ylabel='Pupil diameter (mm)', xlabel='Time (s)', ylim=(1.5, 9.0))
    ax.legend(fontsize=7, loc='center right')

    path, rp = utils.save_fig(fig, 'clin_pupil_anisocoria_cn3', show=show,
                              params=with_cn3_palsy(PARAMS, side='left'),
                              conditions='Dark → lit step; left CN III palsy (pupil-involving)')
    return utils.fig_meta(path, rp,
        title='Anisocoria — left CN III palsy',
        description='A pupil-involving left oculomotor palsy: the left pupil is fixed and '
                    'dilated (parasympathetic sphincter fibres travel with CN III) while the '
                    'right pupil constricts normally to light — anisocoria.',
        expected='Left pupil stays ~8.5 mm (blown, unreactive); right pupil constricts to '
                 '~3 mm on light. Pupil-sparing palsy would keep both reactive.',
        citation='Loewenfeld (1993); Kerr & Hollowell (1964)',
    )


# ── Pupil lesion survey (left pupil across conditions) ────────────────────────

def _lesions(show):
    T_ON, TOTAL = 3.0, 10.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    scene = np.zeros(T, dtype=np.float32); scene[t >= T_ON] = 1.0

    conditions = [
        ('Healthy',                    PARAMS,                                        '#333333', '-'),
        ('CN III palsy (blown)',       with_cn3_palsy(PARAMS, side='left'),           '#b2182b', '-'),
        ('RAPD (afferent 0.2)',        with_brain(PARAMS, g_pupil_afferent_L=0.2),    '#ef8a62', '--'),
        ('Argyll Robertson (light 0)', with_brain(PARAMS, g_pupil_light_reflex=0.0),  '#2166ac', '-.'),
        ('Horner (left)',              with_horner(PARAMS, side='left', miosis_mm=2.5),'#1a9850', ':'),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.0))
    fig.suptitle('Pupil lesion survey — LEFT pupil, response to a light step (at 3 s)',
                 fontsize=11, fontweight='bold')
    ax.axvspan(T_ON, TOTAL, color='#fff3b0', alpha=0.4, lw=0)
    for label, params, color, ls in conditions:
        st = simulate(params, t, scene_present_array=scene, return_states=True, key=KEY)
        ax.plot(t, pupil_size(st)[:, 0], color=color, ls=ls, lw=1.8, label=label)
    ax_fmt(ax, ylabel='Left pupil diameter (mm)', xlabel='Time (s)', ylim=(1.5, 9.0))
    ax.legend(fontsize=7, loc='center right', ncol=1)

    path, rp = utils.save_fig(fig, 'clin_pupil_lesions', show=show, params=PARAMS,
                              conditions='Dark → lit step; left-sided lesions, left pupil shown')
    return utils.fig_meta(path, rp,
        title='Pupil lesion survey',
        description='Left pupil across healthy, CN III palsy (blown), RAPD, Argyll Robertson '
                    '(light-near dissociation), and Horner (sympathetic miosis).',
        expected='Healthy constricts; CN III blown + Argyll Robertson stay dilated (no light '
                 'reflex); RAPD constricts weakly; Horner sits at a small baseline but reacts.',
        citation='Loewenfeld (1993); McDougal & Gamlin (2015)',
    )


def run(show=SHOW):
    print('\n=== Clinical Pupil (anisocoria, lesion survey) ===')
    print('  1/2  anisocoria (left CN III) …'); f1 = _anisocoria(show)
    print('  2/2  lesion survey …');            f2 = _lesions(show)
    return [f1, f2]


if __name__ == '__main__':
    figs = run(show=SHOW)
    for f in figs:
        print(f['title'])
    if SHOW:
        plt.show()
