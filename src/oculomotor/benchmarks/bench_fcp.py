"""Final common pathway benchmark — cascade over a 9-position gaze grid + near/far vergence.

Stacked timecourse of the whole final common pathway, decomposed Robinson pulse/step:
  eye(version + ocular vergence)
    → version  STEP = NI bilateral pops      |  PULSE = τp·burst (feed-through)
    → vergence STEP = verg_fast + verg_tonic  |  PULSE = direct path  | AC/A cross-link
    → MN(14, signed, incl. MLF internuclear AIN)  → nerves/muscles(12, pull-only), split by eye.

Makes the [H,V,T] command ↔ per-muscle recruitment mapping legible (substrate for the
canal/muscle-aligned coordinate work) and shows the vergence pulse/step alongside it.

Usage:
    python -X utf8 -m oculomotor.benchmarks.bench_fcp
    python -X utf8 -m oculomotor.benchmarks.bench_fcp --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oculomotor.benchmarks import bench_utils as utils

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_fcp_cascade
from oculomotor.benchmarks.bench_metrics import Metric

SHOW = '--show' in sys.argv
DT   = 0.001
# Noiseless (clean cascade traces): saccades on (g_burst=700), sensory + accumulator noise off.
THETA = with_brain(with_sensory(with_brain(PARAMS_DEFAULT, g_burst=700.0),
                                sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

AXES3      = ['horizontal', 'vertical', 'torsion']
MN_LABELS  = ['LR_L', 'LR_R', 'CN4_L', 'CN4_R', 'MR_L', 'MR_R', 'SR_L', 'SR_R',
              'IR_L', 'IR_R', 'IO_L', 'IO_R', 'AIN_L', 'AIN_R']
NRV_LABELS = ['LR_L', 'MR_L', 'SR_L', 'IR_L', 'SO_L', 'IO_L',
              'LR_R', 'MR_R', 'SR_R', 'IR_R', 'SO_R', 'IO_R']
c3    = ['#c0392b', '#2980b9', '#27ae60']     # horizontal / vertical / torsion
c3_dk = ['#922b21', '#1a5276', '#1d8348']     # NI left pop  (dark)
c3_lt = ['#e6b0aa', '#a9cce3', '#a9dfbf']     # NI right pop (light)
PAIR  = {'LR': '#1f4e79', 'MR': '#7fb3e6', 'SR': '#2e7d32', 'IR': '#9ccc65',
         'SO': '#6a1b9a', 'IO': '#ce93d8', 'CN4': '#6a1b9a', 'AIN': '#9e9e9e'}
def _mcol(lbl): return PAIR.get(lbl.split('_')[0], '#333')

# MN legend: full motor pathway nucleus·side → nerve·side → muscle·side, so the
# decussations are explicit.  CN4 (trochlear) is the one cranial nerve that
# decussates — its axons cross between nucleus and nerve (nucleus L → nerve R →
# SO R).  The abducens internuclear (AIN) is not a cranial nerve: it routes
# through the MLF to the contralateral MR.  (nuc = nerve roman numeral; LR↔VI,
# MR/SR/IR/IO↔III, SO↔IV.)
_PATH = {'LR': ('VI', False, 'LR'), 'MR': ('III', False, 'MR'),
         'SR': ('III', False, 'SR'), 'IR': ('III', False, 'IR'),
         'IO': ('III', False, 'IO'), 'CN4': ('IV', True, 'SO')}
def _opp(s): return 'R' if s == 'L' else 'L'
def _mn_label(lbl):
    stem, side = lbl.split('_')
    if stem == 'AIN':                          # abducens internuclear → MLF → contralateral MR
        return f'VI·{side} →MLF→ MR·{_opp(side)}'
    cn, cross, musc = _PATH[stem]
    ns = _opp(side) if cross else side         # nerve + muscle side (trochlear decussates)
    return f'{cn}·{side} → n{cn}·{ns} → {musc}·{ns}'


def _sequence():
    """Build the 9-position gaze + near/far paradigm → (t, pt3, trans, out_lab).

    Center-out-center: settle, then L/R/U/D + four obliques (±30° at 1 m), always
    returning to center, 1 s holds.  Then two depth shifts on the midline — near
    (0.25 m) and far (4 m) — with 2 s holds (vergence is slower than saccades).
    Target world position = depth · [tan(h), tan(v), 1].
    """
    GAZE = [('L', -30, 0), ('R', 30, 0), ('U', 0, 30), ('D', 0, -30),
            ('UR', 30, 30), ('UL', -30, 30), ('DL', -30, -30), ('DR', 30, -30)]
    seq = [('c', 0, 0, 1.0, 1.0)]                              # settle at 1 m
    for name, h, v in GAZE:
        seq += [(name, h, v, 1.0, 1.0), ('c', 0, 0, 1.0, 1.0)]
    seq += [('near', 0, 0, 0.25, 2.0), ('c', 0, 0, 1.0, 2.0),  # depth shifts (2 s holds)
            ('far',  0, 0, 4.0, 2.0),  ('c', 0, 0, 1.0, 2.0)]

    holds  = np.array([s[4] for s in seq])
    starts = np.concatenate([[0.0], np.cumsum(holds)])
    t      = np.arange(0.0, starts[-1] + 0.5, DT)
    seg    = np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(seq) - 1)
    hh = np.array([s[1] for s in seq])[seg]; vv = np.array([s[2] for s in seq])[seg]
    zz = np.array([s[3] for s in seq])[seg]
    pt3 = np.zeros((len(t), 3))
    pt3[:, 0] = zz * np.tan(np.radians(hh)); pt3[:, 1] = zz * np.tan(np.radians(vv)); pt3[:, 2] = zz
    trans   = starts[1:len(seq)]
    out_lab = [(seq[i][0], starts[i]) for i in range(1, len(seq)) if seq[i][0] != 'c']
    return t, pt3, trans, out_lab


def _cascade(show):
    t, pt3, trans, out_lab = _sequence()
    T = len(t)
    st = simulate(THETA, jnp.array(t), target=km.build_target(t, lin_pos=pt3),
                  scene_present_array=jnp.ones(T), max_steps=T + 300,
                  return_states=True, key=jax.random.PRNGKey(0))
    c = extract_fcp_cascade(st, THETA)

    fig, ax = plt.subplots(9, 1, figsize=(14.5, 20.5), sharex=True)
    for a in ax:
        for tt in trans:
            a.axvline(tt, color='#ececec', lw=0.5, zorder=0)
        a.axhline(0, color='k', lw=0.3); a.grid(True, alpha=0.12)

    for k in range(3):
        ax[0].plot(t, c['eye'][:, k],        color=c3[k],    lw=1.2, label=AXES3[k])
        ax[1].plot(t, c['ni_L'][:, k],       color=c3_dk[k], lw=1.1, label=f'{AXES3[k]} L')
        ax[1].plot(t, c['ni_R'][:, k],       color=c3_lt[k], lw=1.1, label=f'{AXES3[k]} R')
        ax[2].plot(t, c['v_pulse'][:, k],    color=c3[k],    lw=1.0, label=AXES3[k])
        ax[3].plot(t, c['verg_step'][:, k],  color=c3[k],    lw=1.2, label=AXES3[k])
        ax[4].plot(t, c['verg_pulse'][:, k], color=c3[k],    lw=1.0, label=AXES3[k])
    ax[0].plot(t, c['eye_verg'], color='#8e44ad', lw=1.3, label='vergence (L−R)')
    ax[3].plot(t, c['aca'], color='#7f8c8d', lw=1.2, ls='--', label='AC/A (H)')

    mn_L, mn_R = [0, 2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11, 13]
    for j in mn_L: ax[5].plot(t, c['mn'][:, j], color=_mcol(MN_LABELS[j]), lw=1.0, label=_mn_label(MN_LABELS[j]))
    for j in mn_R: ax[6].plot(t, c['mn'][:, j], color=_mcol(MN_LABELS[j]), lw=1.0, label=_mn_label(MN_LABELS[j]))
    for j in range(6):     ax[7].plot(t, c['nerves'][:, j], color=_mcol(NRV_LABELS[j]), lw=1.1, label=NRV_LABELS[j].split('_')[0])
    for j in range(6, 12): ax[8].plot(t, c['nerves'][:, j], color=_mcol(NRV_LABELS[j]), lw=1.1, label=NRV_LABELS[j].split('_')[0])

    titles = ['eye position (version) + ocular vergence  [deg]',
              'version STEP — NI bilateral pops, L/R push-pull  [deg]',
              'version PULSE — τp·burst (feed-through)  [deg]',
              'vergence STEP — verg_fast+tonic, H/V/T  (+ AC/A, H)  [deg]',
              'vergence PULSE — direct path, H/V/T  [deg]',
              'MN firing — LEFT nuclei  (≥0 nucleus firing; AIN signed; nucleus→muscle in legend)  [spk/s]',
              'MN firing — RIGHT nuclei  (≥0 nucleus firing; AIN signed; nucleus→muscle in legend)  [spk/s]',
              'nerves/muscles — LEFT eye  (pull-only)  [spk/s]',
              'nerves/muscles — RIGHT eye  (pull-only)  [spk/s]']
    for a, ti in zip(ax, titles):
        a.set_title(ti, fontsize=9, loc='left')
        for name, tt in out_lab:
            a.text(tt, 0.90, name, transform=a.get_xaxis_transform(), fontsize=7,
                   color='#333', ha='center', va='top',
                   bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.65))
    for i in (0, 1, 2, 3, 4, 5, 6, 7):
        ax[i].legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), fontsize=7.5, frameon=False)
    ax[-1].set_xlabel('time (s)')
    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'fcp_cascade', show=show, params=THETA,
        conditions='Noiseless; 9-position gaze ±30° at 1 m (center-out-center) then near 0.25 m / far 4 m')

    # ── metrics: pull-only invariant + vergence accuracy + gaze accuracy ──────────
    ipd = float(THETA.sensory.ipd)
    def _geo_verg(z): return float(np.degrees(2 * np.arctan(ipd / 2 / z)))
    def _verg_at(tt): return float(c['eye_verg'][int(tt / DT) - 5])     # just before next transition
    near_t, _, far_t, _ = trans[-4:]
    verg_near = _verg_at(near_t + 2.0)        # end of near hold
    verg_far  = _verg_at(far_t + 2.0)         # end of far hold
    pullonly_frac = float(np.mean(c['nerves'] >= -1e-6))
    gaze_amp = float(np.max(np.abs(c['eye'][:, 0])))   # rightward gaze targets at ±30°
    metrics = [
        Metric('fcp_nerve_pullonly_frac', pullonly_frac, lo=0.999, hi=None, golden_tol=0.001,
               units='frac', cite='—',
               desc='Fraction of per-muscle nerve samples ≥ 0 — pull-only co-contraction invariant'),
        Metric('fcp_gaze_amp', gaze_amp, lo=27.0, hi=33.0, golden_tol=0.1, units='deg',
               cite='—', desc='Peak horizontal eye position at the 30° gaze targets'),
        Metric('fcp_verg_near', verg_near, lo=_geo_verg(0.25) - 2.5, hi=_geo_verg(0.25) + 2.5,
               golden_tol=0.2, units='deg', cite='—',
               desc=f'Ocular vergence at near target (0.25 m; geometric {_geo_verg(0.25):.1f}°)'),
        Metric('fcp_verg_far', verg_far, lo=0.3, hi=2.5, golden_tol=0.3, units='deg', cite='—',
               desc=f'Ocular vergence at far target (4 m; geometric {_geo_verg(4.0):.1f}°)'),
    ]

    fm = utils.fig_meta(path, rp,
        title='Final Common Pathway — cascade over 9-position gaze + near/far vergence',
        description='Stacked timecourse of the whole final common pathway for a center-out-center '
                    '9-position gaze sequence (±30° at 1 m) followed by near (0.25 m) and far (4 m) '
                    'depth shifts. Version drive is decomposed Robinson pulse/step (NI bilateral pops '
                    '+ τp·burst feed-through); vergence drive is decomposed step (verg_fast+tonic), '
                    'pulse (direct path) and AC/A cross-link, all in H/V/T. The command then flows '
                    'through the 14 signed motoneurons (incl. CN4 and the MLF internuclear AIN) to the '
                    '12 pull-only per-muscle nerve drives, split by eye.',
        expected='Eye reaches each gaze target; ocular vergence rides ~3.7° at 1 m, ~14.6° at near and '
                 '~0.9° at far (V/T vergence ≈ 0 in this symmetric paradigm). Each saccade is an NI step '
                 '+ burst pulse; the conjugate [H,V,T] command spreads across the recti/obliques per eye. '
                 'Nerve drive stays pull-only (≥0) via reciprocal co-contraction.',
        citation='Robinson (1975); Sparks (2002)',
        fig_type='behavior')
    fm['metrics'] = metrics
    return fm


SECTION = dict(
    id='fcp', title='8. Final Common Pathway',
    description='Motoneuron + cranial-nerve output stage. Traces the full command cascade — '
                'Robinson pulse/step decomposition of version and vergence drive through the 14 '
                'motoneurons (incl. the MLF internuclear AIN) to the 12 pull-only per-muscle nerves '
                '— over a 9-position gaze grid plus near/far vergence shifts. Exposes the '
                '[H,V,T]-command ↔ per-muscle recruitment mapping.',
)


def run(show=False):
    print('\n=== Final Common Pathway ===')
    figs = []
    print('  1/1  cascade (9-position gaze + near/far) …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
