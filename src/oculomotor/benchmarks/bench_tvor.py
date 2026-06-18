"""Translational VOR (T-VOR) cascade benchmark.

Single cascade panel: trapezoid head-velocity pulses along sway / heave / surge
in two conditions:
    LIT  — scene visible, NO fixation target (so the eye is driven by T-VOR +
           OKR / heading visual fusion, but NOT by saccadic catch-up to a target)
    DARK — scene off, target off → pure vestibular T-VOR

Rows (top to bottom):
    0. head linear velocity (stimulus)
    1. g_est (gravity estimator)
    2. v_lin (heading estimator — vest+visual fusion)
    3. eye position (yaw / pitch / vergence)
    4. eye SPV — slow-phase velocity (saccades masked via OPN latch)

Usage:
    python -X utf8 scripts/bench_tvor.py
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

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain
from oculomotor.sim import kinematics as km
from oculomotor.analysis import (extract_spv_states, extract_spv, extract_z_opn,
                                 ax_fmt)
from oculomotor.benchmarks.bench_metrics import Metric
from oculomotor.benchmarks import bode

SHOW = '--show' in sys.argv
DT   = 0.001
KEY  = jax.random.PRNGKey(0)


SECTION = dict(
    id='tvor', title='8. Translational VOR (T-VOR)',
    description='Vest + visual fusion of head translation, scaled by vergence-derived '
                'distance to give compensatory eye velocity. Trapezoid pulses along '
                'sway / heave / surge in lit (scene only) and dark conditions.',
)


def _cascade(show):
    """T-VOR cascade: short velocity pulse, 3 directions × {LIT scene-only, DARK} conditions.

    Rows: stimulus, g_est, v_lin, eye position, eye SPV.
    LIT shown solid, DARK shown dashed.
    LIT = scene visible, target NOT visible — so heading-visual fusion + OKR are on,
          but the brain has no foveation target to make saccades to.  This isolates
          the visual contribution to T-VOR slow phase from any saccadic catch-up.
    DARK = scene off, target off — pure vestibular T-VOR.
    """

    AXES   = [(0, +1, 'Sway',  'rightward'),
              (1, +1, 'Heave', 'upward'),
              (2, -1, 'Surge', 'backward (away from target)')]
    PEAK   = 0.20    # 20 cm/s peak velocity (plateau)
    RAMP   = 0.2     # 200 ms ramp up + ramp down (smooth — avoids huge accel transients)
    HOLD   = 1.6     # 1.6 s plateau between ramps  → total motion ≈ 2 s
    T_PULSE= 0.5
    T_TOTAL = 8.0
    DEPTH  = 0.4     # m, near target — for the brain this is invisible (target_present=0),
                     # so the brain uses tonic_verg for T-VOR distance regardless of DEPTH.
    # Cascade trace — disable all sensory + accumulator noise so the curves are clean.
    from oculomotor.sim.simulator import with_sensory
    params = with_brain(
        with_sensory(PARAMS_DEFAULT,
                     sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
        sigma_acc=0.0,
    )
    t = np.arange(0.0, T_TOTAL, DT)
    T = len(t)

    # Smooth trapezoid velocity envelope (in [0, 1])
    t_rel = t - T_PULSE
    env   = np.zeros(T)
    ramp_up   = (t_rel >= 0)        & (t_rel < RAMP)
    plateau   = (t_rel >= RAMP)     & (t_rel < RAMP + HOLD)
    ramp_dn   = (t_rel >= RAMP + HOLD) & (t_rel < 2*RAMP + HOLD)
    env[ramp_up] = t_rel[ramp_up] / RAMP
    env[plateau] = 1.0
    env[ramp_dn] = 1.0 - (t_rel[ramp_dn] - RAMP - HOLD) / RAMP

    fig, axes = plt.subplots(8, 3, figsize=(15, 19), sharex=True)
    fig.suptitle(f'T-VOR cascade — LIT scene-only (solid) vs DARK (dashed); '
                 f'L-eye blue, R-eye orange; peak {PEAK*100:.0f} cm/s, '
                 f'{RAMP*1000:.0f} ms ramps, {HOLD:.1f} s plateau',
                 fontsize=12, fontweight='bold')

    CONDITIONS = [('LIT',  True,  '-'),
                  ('DARK', False, '--')]

    for col, (axis, sign, name, dir_descr) in enumerate(AXES):
        # Stimulus is identical for both conditions
        head_vel = np.zeros((T, 3), dtype=np.float32)
        head_vel[:, axis] = (sign * PEAK * env).astype(np.float32)
        head = km.build_kinematics(t, lin_vel=head_vel)
        pt = np.tile([0.0, 0.0, DEPTH], (T, 1)).astype(np.float32)
        target = km.build_target(t, lin_pos=pt)

        # Plot stimulus once
        ax = axes[0, col]
        ax.plot(t, head_vel[:, axis] * 100, color='gray', lw=1.4)
        ax.axhline(0, color='k', lw=0.4)
        ax.set_title(f'{name} ({dir_descr})', fontsize=10, fontweight='bold')
        if col == 0: ax.set_ylabel('Head vel\n(cm/s)', fontsize=9)
        ax.grid(True, alpha=0.2)

        for cond_name, scene_on, ls in CONDITIONS:
            st = simulate(
                params, t, head=head, target=target,
                # LIT = scene visible, target invisible (no foveation target);
                # DARK = nothing visible (pure vestibular T-VOR).
                scene_present_array  = np.ones(T)  if scene_on else np.zeros(T),
                target_present_array = np.zeros(T),
                return_states=True, key=KEY)

            g_est = np.array(st.brain.sm.g_est)
            v_lin = np.array(st.brain.sm.v_lin)
            eye_L = np.array(st.plant.left)
            eye_R = np.array(st.plant.right)
            eye_version = (eye_L + eye_R) / 2.0
            eye_verg    = eye_L[:, 0] - eye_R[:, 0]

            # Smoothed eye velocity per axis.  We don't use the OPN saccade mask here
            # because (a) any saccade spikes are the actual eye velocity that needs
            # to be visible, and (b) interpolation across masked epochs would smear
            # the T-VOR slow-phase magnitude downward.  A 50 ms moving average on the
            # eye position is enough to suppress per-sample gradient noise without
            # blurring saccades or the slow-phase envelope.
            smooth_n   = 50   # 50 samples = 50 ms at DT=1ms
            kernel     = np.ones(smooth_n) / smooth_n
            def _smooth_grad(y):
                return np.gradient(np.convolve(y, kernel, mode='same'), DT)
            spv_L = np.stack([_smooth_grad(eye_L[:, i]) for i in range(3)], axis=1)
            spv_R = np.stack([_smooth_grad(eye_R[:, i]) for i in range(3)], axis=1)

            # ── Row 1: g_est ───────────────────────────────────────────────
            ax = axes[1, col]
            ax.plot(t, g_est[:, 0],         color='#c0392b', lw=1.0, ls=ls,
                    label=f'{cond_name} g[x]'   if col == 0 else None)
            ax.plot(t, g_est[:, 1] - 9.81,  color='#27ae60', lw=1.0, ls=ls,
                    label=f'{cond_name} g[y]−G' if col == 0 else None)
            ax.plot(t, g_est[:, 2],         color='#2980b9', lw=1.0, ls=ls,
                    label=f'{cond_name} g[z]'   if col == 0 else None)

            # ── Row 2: v_lin (HE) ──────────────────────────────────────────
            ax = axes[2, col]
            ax.plot(t, v_lin[:, 0]*100, color='#c0392b', lw=1.0, ls=ls,
                    label=f'{cond_name} v_lin[x]' if col == 0 else None)
            ax.plot(t, v_lin[:, 1]*100, color='#27ae60', lw=1.0, ls=ls,
                    label=f'{cond_name} v_lin[y]' if col == 0 else None)
            ax.plot(t, v_lin[:, 2]*100, color='#2980b9', lw=1.0, ls=ls,
                    label=f'{cond_name} v_lin[z]' if col == 0 else None)

            # ── Row 3: eye position ────────────────────────────────────────
            ax = axes[3, col]
            ax.plot(t, eye_version[:, 0], color='#1f77b4', lw=1.1, ls=ls,
                    label=f'{cond_name} Yaw'   if col == 0 else None)
            ax.plot(t, eye_version[:, 1], color='#2ca02c', lw=1.1, ls=ls,
                    label=f'{cond_name} Pitch' if col == 0 else None)
            ax.plot(t, eye_verg,          color='#d62728', lw=1.1, ls=ls,
                    label=f'{cond_name} Verg'  if col == 0 else None)

            # ── Rows 4/5/6: per-axis eye velocity, both eyes (L blue, R orange) ─
            for r, axis_idx, axis_name in [(4, 0, 'Yaw'),
                                           (5, 1, 'Pitch'),
                                           (6, 2, 'Roll')]:
                ax = axes[r, col]
                ax.plot(t, spv_L[:, axis_idx], color='#1f77b4', lw=1.0, ls=ls,
                        label=f'{cond_name} L' if col == 0 else None)
                ax.plot(t, spv_R[:, axis_idx], color='#ff7f0e', lw=1.0, ls=ls,
                        label=f'{cond_name} R' if col == 0 else None)

            # ── Row 7: vergence velocity (L − R, all 3 axes overlaid) ──────
            ax = axes[7, col]
            verg_vel = spv_L - spv_R   # (T, 3) per-axis vergence velocity
            ax.plot(t, verg_vel[:, 0], color='#1f77b4', lw=1.0, ls=ls,
                    label=f'{cond_name} H-verg' if col == 0 else None)
            ax.plot(t, verg_vel[:, 1], color='#2ca02c', lw=1.0, ls=ls,
                    label=f'{cond_name} V-verg' if col == 0 else None)
            ax.plot(t, verg_vel[:, 2], color='#d62728', lw=1.0, ls=ls,
                    label=f'{cond_name} Cyclo'  if col == 0 else None)

        # Common formatting per row in each column
        for r, ylab in [(1, 'g_est\n(m/s², − G₀ on y)'),
                        (2, 'v_lin (HE)\n(cm/s)'),
                        (3, 'Eye position\n(deg)'),
                        (4, 'Eye vel Yaw\n(deg/s)'),
                        (5, 'Eye vel Pitch\n(deg/s)'),
                        (6, 'Eye vel Roll\n(deg/s)'),
                        (7, 'Vergence vel\nL−R (deg/s)')]:
            ax = axes[r, col]
            ax.axhline(0, color='k', lw=0.4)
            if col == 0: ax.set_ylabel(ylab, fontsize=9)
            ax.grid(True, alpha=0.2)
            if r in (4, 5):
                ax.set_ylim(-15, 15)   # yaw/pitch eye velocity — wider range
            if r in (6, 7):
                ax.set_ylim(-2, 2)     # roll + vergence — tight zoom for slow-phase detail
            if r == 7:
                ax.set_xlabel('Time (s)', fontsize=9)
            if col == 0:
                ax.legend(fontsize=6.5, loc='upper right', ncol=2)

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    path, rp = utils.save_fig(fig, 'tvor_cascade', show=show, params=params,
                              conditions='Lit, head linear acceleration + near/far targets (translational VOR cascade) — noiseless')
    return utils.fig_meta(path, rp,
        title='T-VOR cascade — sway / heave / surge × LIT (scene only) / DARK',
        description=f'Trapezoid head velocity (peak {PEAK*100:.0f} cm/s, {RAMP*1000:.0f} ms ramps, '
                    f'{HOLD:.1f} s plateau) along x (sway), y (heave), z (surge); target at {DEPTH*100:.0f} cm. '
                    'Rows: stimulus, g_est, v_lin, eye position, eye SPV.',
        expected='Sway → eye yaw counter-rotates (cross product); '
                 'Heave → eye pitch counter-rotates; '
                 'Surge → vergence rate (dot product · IPD/D²) drives vergence change. '
                 'g_est should remain ≈ [0, 9.81, 0]. SPV in DARK measures pure vestibular T-VOR; '
                 'LIT scene-only adds OKR / visual heading contribution.',
        citation='Paige & Tomko (1991); Angelaki & Hess (2001)',
    )


def _bode(show):
    """T-VOR lateral (sway) frequency response — sinusoidal interaural translation,
    NOISELESS. Primary channel: horizontal eye velocity (SPV) ÷ the GEOMETRIC IDEAL
    (V/d, so gain ≈ 1 = perfect compensation); high-pass. Conditions: dark, scene,
    scene+target. Bottom panel: the cross-axis effect on torsion (somatogravic/OCR).
    """
    from oculomotor.sim.simulator import with_sensory
    params = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                     sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)
    V_AMP, D = 0.20, 0.4                       # 20 cm/s peak, target at 40 cm
    FREQS = np.array([0.1, 0.2, 0.5, 1.0, 2.0, 4.0])
    N_CYC, SETTLE = 6, 3.0
    CONDS = [('dark', 0.0, 0.0, '#444444'),
             ('scene', 1.0, 0.0, '#1b7837'),
             ('scene+target', 1.0, 1.0, '#c0392b')]

    def measure(scene_p, target_p, f):
        T_end = min(SETTLE + N_CYC / f, 45.0)
        t = np.arange(0.0, T_end, DT); Tn = len(t); w = 2 * np.pi * f
        on = t >= SETTLE
        hv = np.zeros((Tn, 3), np.float32)
        hv[:, 0] = np.where(on, V_AMP * np.sin(w * (t - SETTLE)), 0.0).astype(np.float32)
        head   = km.build_kinematics(t, lin_vel=hv)
        target = km.build_target(t, lin_pos=np.tile([0.0, 0.0, D], (Tn, 1)).astype(np.float32))
        st = simulate(params, t, head=head, target=target,
                      scene_present_array=np.full(Tn, scene_p),
                      target_present_array=np.full(Tn, target_p),
                      return_states=True, key=KEY)
        eye = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
        yaw_spv = extract_spv_states(st, t, eye='version')[:, 0]
        ideal   = -(hv[:, 0] / D) * (180.0 / np.pi)        # geometric-ideal eye yaw vel (deg/s)
        g, ph   = bode.bode_point(t, ideal, yaw_spv, f, settle_frac=0.4)
        m = t >= (t[0] + 0.4 * (t[-1] - t[0]))
        tor_amp, _, _ = bode.fit_sinusoid(t[m], eye[m, 2], f)   # torsion (roll) amplitude
        return g, ph, float(tor_amp)

    res = {lab: {'g': [], 'ph': [], 'tor': []} for lab, *_ in CONDS}
    for lab, sp, tp, col in CONDS:
        for f in FREQS:
            g, ph, tor = measure(sp, tp, f)
            res[lab]['g'].append(g); res[lab]['ph'].append(ph); res[lab]['tor'].append(tor)

    fig, (axg, axp, axt) = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
    fig.suptitle('T-VOR lateral (sway) — Frequency Response (noiseless)\n'
                 'primary gain vs geometric ideal (V/d)  ·  cross-effect on torsion',
                 fontsize=11, fontweight='bold')
    for lab, sp, tp, col in CONDS:
        axg.loglog(FREQS, res[lab]['g'],   'o-', color=col, lw=1.5, ms=5, label=lab)
        axp.semilogx(FREQS, res[lab]['ph'], 's-', color=col, lw=1.5, ms=5, label=lab)
        axt.semilogx(FREQS, res[lab]['tor'], 'd-', color=col, lw=1.5, ms=5, label=lab)
    axg.axhline(1.0, color='gray', lw=0.7, ls=':', alpha=0.6)
    axg.set_ylabel('Primary gain\n(eye yaw ÷ ideal)'); axg.legend(fontsize=8); axg.grid(True, which='both', alpha=0.2)
    axp.axhline(0, color='k', lw=0.4); axp.set_ylabel('Phase (deg)'); axp.grid(True, which='both', alpha=0.2)
    axt.set_ylabel('Cross torsion\namplitude (deg)'); axt.set_xlabel('Frequency (Hz)')
    axt.grid(True, which='both', alpha=0.2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path, rp = utils.save_fig(fig, 'tvor_bode', show=show, params=params,
        conditions=f'NOISELESS — sinusoidal lateral (sway) translation 0.1–4 Hz '
                   f'({V_AMP*100:.0f} cm/s), target {D*100:.0f} cm')

    metrics = []
    i1 = int(np.argmin(np.abs(FREQS - 1.0)))
    for lab, sp, tp, col in CONDS:
        k = lab.replace('+', '_').replace(' ', '')
        metrics.append(Metric(f'tvor_bode_gain_high_{k}', float(res[lab]['g'][-1]), tier='monitor',
            lo=0.2, hi=1.4, golden_tol=0.15, units='',
            cite='Paige & Tomko (1991); Angelaki & Hess (2001)',
            desc=f'tVOR high-frequency gain vs geometric ideal (4 Hz) — {lab}'))
        metrics.append(Metric(f'tvor_torsion_cross_{k}', float(res[lab]['tor'][i1]), tier='monitor',
            lo=None, hi=None, golden_tol=0.3, units='deg',
            cite='cross-axis somatogravic / OCR coupling',
            desc=f'Cross-axis torsion amplitude at 1 Hz during lateral translation — {lab}'))
    fm = utils.fig_meta(path, rp,
        title='T-VOR — Lateral Bode (primary gain + torsion cross-effect)',
        description=f'Sinusoidal interaural translation ({V_AMP*100:.0f} cm/s, 0.1–4 Hz), NOISELESS, '
                    f'target at {D*100:.0f} cm. Top: horizontal eye-velocity gain vs the geometric ideal '
                    f'(V/d); middle: phase; bottom: cross-axis torsion (somatogravic / OCR coupling).',
        expected='tVOR is high-pass — gain rises toward 1 at high frequency (otolith tilt/translation '
                 'disambiguation). Cross torsion rolls off with frequency (somatogravic low-pass).',
        citation='Paige & Tomko (1991); Angelaki & Hess (2001); Laurens & Angelaki (2011)',
        fig_type='behavior')
    fm['metrics'] = metrics
    return fm


def run(show=False):
    print('\n=== T-VOR ===')
    figs = []
    print('  1/2  cascade …')
    figs.append(_cascade(show))
    print('  2/2  lateral Bode …')
    figs.append(_bode(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
    if SHOW:
        # save_fig uses non-blocking show; final blocking show keeps windows open.
        plt.show()
