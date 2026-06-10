"""Alternating cover test — simulated esotropia and exotropia.

Alternates the occluder between L and R eyes to break binocular fusion and reveal
the full manifest strabismic deviation.  Refixation saccades when the cover
switches eyes are the key diagnostic sign.

Protocol (per simulation):
    0 – T_BIN  : binocular fixation (both eyes see target)
    T_BIN – T2 : cover R eye (target_L=1, target_R=0)
    T2 – T3    : cover L eye (target_L=0, target_R=1)
    T3 – T4    : cover R eye
    T4 – T5    : cover L eye
    T5 – T_END : binocular (fusion recovery)

Conditions (three columns):
    Exotropia  : tonic_verg = 0°   → manifest exo at far (eyes too diverged)
    Normal     : tonic_verg ≈ 3.7° → small eso phoria at far (tonic > geo)
    Esotropia  : tonic_verg = 9°   → manifest eso at far (eyes too converged)

Rows per panel:
    0 : L (blue) and R (red) eye yaw positions — refixation saccades visible
    1 : Vergence (L−R) — drifts to tonic when one eye is covered
    2 : Version (L+R)/2 — saccade-driven version shifts visible

Usage:
    python -X utf8 scripts/bench_alternating_cover_test.py
    python -X utf8 scripts/bench_alternating_cover_test.py --show
"""

import sys
import os
import datetime

# Ensure src/ is on path whether run from repo root or scripts/
_SCRIPTS = os.path.dirname(os.path.abspath(__file__))
_ROOT    = os.path.normpath(os.path.join(_SCRIPTS, '..'))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

import numpy as np
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import oculomotor
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ax_fmt

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064   # m, default inter-pupillary distance

# Noiseless params: suppress microsaccades and canal noise so the cover-induced
# drifts and refixation saccades are the dominant signal in the traces.
_PARAMS_NOISELESS = with_sensory(
    PARAMS_DEFAULT,
    sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0,
)
_PARAMS_NOISELESS = with_brain(_PARAMS_NOISELESS, sigma_acc=0.0)

TONIC_DEFAULT = PARAMS_DEFAULT.brain.tonic_verg

CONDITIONS = [
    # (column title, tonic_verg_deg, line_color)
    ('Exotropia\n(tonic = 0°)',                          0.0,           '#d62728'),
    (f'Normal\n(tonic = {TONIC_DEFAULT:.1f}°)',          TONIC_DEFAULT, '#2ca02c'),
    ('Esotropia\n(tonic = 9°)',                          9.0,           '#1f77b4'),
]

# ── Alternating cover protocol ────────────────────────────────────────────────

T_BIN = 1.0    # s — initial binocular fixation
T2    = 3.0    # s — switch: cover L eye
T3    = 5.0    # s — switch: cover R eye
T4    = 7.0    # s — switch: cover L eye
T5    = 9.0    # s — uncover: binocular fusion recovery
T_END = 11.5   # s

# (t_start, t_end, which_eye_is_covered)  'both' = both open
PHASES = [
    (0.0,   T_BIN, 'both'),
    (T_BIN, T2,    'R'),
    (T2,    T3,    'L'),
    (T3,    T4,    'R'),
    (T4,    T5,    'L'),
    (T5,    T_END, 'both'),
]

_COLOR_R_COVERED = '#f4a582'   # light orange — R eye covered
_COLOR_L_COVERED = '#92c5de'   # light blue   — L eye covered


def _geo_verg(d_m):
    """Geometric vergence for a midline target at depth d_m (degrees)."""
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / d_m))


D_FAR   = 3.0
GEO_FAR = _geo_verg(D_FAR)    # ≈ 1.22°


# ── Simulation ────────────────────────────────────────────────────────────────

def _run(t, tonic_verg):
    """Simulate alternating cover test at far distance.  Returns (eL_yaw, eR_yaw)."""
    T     = len(t)
    p_tgt = np.tile([0.0, 0.0, float(D_FAR)], (T, 1))

    tg_L = np.ones(T, dtype=np.float32)
    tg_R = np.ones(T, dtype=np.float32)
    for t_s, t_e, eye in PHASES:
        mask = (t >= t_s) & (t < t_e)
        if eye == 'L':
            tg_L[mask] = 0.0
        elif eye == 'R':
            tg_R[mask] = 0.0

    params = with_brain(_PARAMS_NOISELESS, tonic_verg=float(tonic_verg))
    st = simulate(
        params, t,
        target=km.build_target(t, lin_pos=p_tgt),
        target_present_L_array=tg_L,
        target_present_R_array=tg_R,
        scene_present_array=np.ones(T, dtype=np.float32),
        return_states=True,
    )
    return np.array(st.plant.left[:, 0]), np.array(st.plant.right[:, 0])


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _shade_phases(ax, ylo, yhi):
    """Background shading for cover phases; also draws phase-transition markers."""
    for t_s, t_e, eye in PHASES:
        if eye == 'R':
            ax.axvspan(t_s, t_e, alpha=0.18, color=_COLOR_R_COVERED, zorder=0, lw=0)
        elif eye == 'L':
            ax.axvspan(t_s, t_e, alpha=0.18, color=_COLOR_L_COVERED, zorder=0, lw=0)
    for t_s, _, eye in PHASES:
        if eye != 'both':
            ax.axvline(t_s, color='#888888', lw=0.6, ls=':', zorder=1)


def _annotate_phases(ax, y_top):
    """Label each cover phase at the top of the axis (only once, middle column)."""
    labels = {
        'R': 'R covered',
        'L': 'L covered',
        'both_start': 'binocular',
        'both_end': 'fusion\nrecovery',
    }
    for i, (t_s, t_e, eye) in enumerate(PHASES):
        xm = (t_s + t_e) / 2.0
        if eye == 'R':
            ax.text(xm, y_top, 'R covered', ha='center', va='top',
                    fontsize=7, color='#b22222', style='italic')
        elif eye == 'L':
            ax.text(xm, y_top, 'L covered', ha='center', va='top',
                    fontsize=7, color='#1a6095', style='italic')
        elif i == 0:
            ax.text(xm, y_top, 'binocular', ha='center', va='top',
                    fontsize=7, color='#555555', style='italic')
        else:
            ax.text(xm, y_top, 'fusion\nrecovery', ha='center', va='top',
                    fontsize=7, color='#555555', style='italic')


# ── Main figure ───────────────────────────────────────────────────────────────

def main(show=False):
    t = np.arange(0.0, T_END, DT)

    # ── Run simulations ───────────────────────────────────────────────────────
    sims = {}
    for label, tonic, _ in CONDITIONS:
        print(f'  Simulating {label.split(chr(10))[0]} …')
        eL, eR = _run(t, tonic)
        sims[label] = dict(eL=eL, eR=eR, verg=eL - eR, vers=(eL + eR) / 2)

    # ── Figure ────────────────────────────────────────────────────────────────
    NCOL = len(CONDITIONS)
    fig, axes = plt.subplots(3, NCOL, figsize=(15, 10), sharex=True)
    fig.suptitle(
        f'Alternating Cover Test  —  far target ({D_FAR} m)\n'
        f'Geometric vergence demand = {GEO_FAR:.1f}°  |  '
        f'Orange shading = R eye covered,  Blue shading = L eye covered',
        fontsize=11, fontweight='bold',
    )

    CL = '#2166ac'   # L eye colour
    CR = '#d6604d'   # R eye colour

    for col, (label, tonic, color) in enumerate(CONDITIONS):
        axes[0, col].set_title(label, fontsize=10, pad=6)
        d = sims[label]
        eL, eR, verg, vers = d['eL'], d['eR'], d['verg'], d['vers']

        phoria = tonic - GEO_FAR
        pdir   = 'eso' if phoria > 0.15 else ('exo' if phoria < -0.15 else 'ortho')

        # ── Row 0: per-eye yaw ────────────────────────────────────────────────
        ax = axes[0, col]
        y_lo = min(eL.min(), eR.min(), -abs(tonic) / 2 - 0.5) - 1.0
        y_hi = max(eL.max(), eR.max(),  abs(tonic) / 2 + 0.5) + 1.5
        _shade_phases(ax, y_lo, y_hi)

        ax.plot(t, eL, color=CL, lw=1.5, label='L eye')
        ax.plot(t, eR, color=CR, lw=1.5, ls='--', label='R eye')
        ax.axhline(+GEO_FAR / 2, color=CL, lw=0.8, ls='-.', alpha=0.55,
                   label=f'L geo +{GEO_FAR/2:.2f}°')
        ax.axhline(-GEO_FAR / 2, color=CR, lw=0.8, ls='-.', alpha=0.55,
                   label=f'R geo −{GEO_FAR/2:.2f}°')
        ax.set_ylim(y_lo, y_hi)
        ax_fmt(ax, ylabel='Eye yaw (deg)' if col == 0 else '')
        if col == 0:
            ax.legend(fontsize=7.5, loc='lower left', ncol=2)
        if col == 1:
            _annotate_phases(ax, y_hi * 0.97)

        # ── Row 1: vergence ───────────────────────────────────────────────────
        ax = axes[1, col]
        v_lo = min(verg.min(), GEO_FAR, tonic) - 1.2
        v_hi = max(verg.max(), GEO_FAR, tonic) + 1.2
        _shade_phases(ax, v_lo, v_hi)

        ax.plot(t, verg, color=color, lw=1.5, label='Vergence L−R')
        ax.axhline(GEO_FAR, color='tomato', lw=1.0, ls='--',
                   label=f'Geo demand {GEO_FAR:.1f}°')
        ax.axhline(tonic, color='#888', lw=0.9, ls=':',
                   label=f'Tonic {tonic:.1f}°')
        ax.set_ylim(v_lo, v_hi)

        psign = '+' if phoria >= 0 else ''
        ax.text(0.97, 0.05,
                f'Manifest deviation: {psign}{phoria:.1f}° ({pdir})',
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=8, color='#222222')
        ax_fmt(ax, ylabel='Vergence L−R (deg)' if col == 0 else '')
        if col == 0:
            ax.legend(fontsize=7.5, loc='upper left')

        # ── Row 2: version ────────────────────────────────────────────────────
        ax = axes[2, col]
        vhi = max(abs(vers).max() + 0.5, 1.5)
        _shade_phases(ax, -vhi, vhi)

        ax.plot(t, vers, color='#4dac26', lw=1.4, label='Version (L+R)/2')
        ax.axhline(0, color='gray', lw=0.8, ls='--', alpha=0.6, label='Midline')
        ax.set_ylim(-vhi, vhi)
        ax_fmt(ax, ylabel='Version (L+R)/2 (deg)' if col == 0 else '',
               xlabel='Time (s)')
        if col == 0:
            ax.legend(fontsize=7.5)

    fig.tight_layout(rect=[0, 0.03, 1, 0.94])

    # ── Watermark ─────────────────────────────────────────────────────────────
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = oculomotor.__version__
    fig.text(0.998, 0.003, f'{ts}  |  {ver}',
             ha='right', va='bottom', fontsize=6, color='#888888',
             transform=fig.transFigure)
    fig.text(0.005, 0.003,
             'Params: noiseless (sigma_canal=0, sigma_pos=0, sigma_vel=0, sigma_acc=0)  |  '
             'tonic_verg varies per column',
             ha='left', va='bottom', fontsize=6, color='#888888',
             transform=fig.transFigure)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_dir  = os.path.join(_ROOT, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'alternating_cover_test.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'Saved → {out_path}')

    if show:
        plt.show()
    else:
        plt.close(fig)

    return out_path


if __name__ == '__main__':
    main(show=SHOW)
