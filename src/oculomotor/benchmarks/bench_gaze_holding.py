"""Gaze-holding benchmarks — eccentric fixation, slow-phase drift vs eye position.

The neural integrator's job is to hold the eye still off-centre. Its leak shows up
as *centripetal* drift whose speed grows with eccentricity, so the diagnostic
signal is the SLOPE of slow-phase velocity against eye position — that slope is
1/τ_eff, the effective gaze-holding time constant. A single-position drift number
cannot separate a leaky integrator (drift ∝ position) from a spontaneous
nystagmus (drift constant across positions); the slope can.

Two lighting conditions, because they probe different machinery:

* **Light** — full-field scene visible. Retinal slip from any drift drives OKR/VS,
  which corrects it smoothly. Gaze holding here is vision-assisted, so drift is
  near zero even with an imperfect integrator.
* **Dark, strobed target** — no scene, so no OKR. The target is stroboscopic:
  it still defines *where* to look (position error → corrective quick phases) but
  carries no motion signal, so the pursuit channel cannot smooth the drift out.
  What remains is the integrator holding on its own — the clinical dark-room test.

Noiseless by design: the NI leak is deterministic, and sensory noise would add
scatter to a slope fit for no scientific gain.

Usage:
    python -X utf8 -m oculomotor.benchmarks.bench_gaze_holding
    python -X utf8 -m oculomotor.benchmarks.bench_gaze_holding --show
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

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig,
)
from oculomotor.sim import kinematics as km
from oculomotor.sim.stimuli import strobe_train, STROBE_HZ_DEFAULT, STROBE_MS_DEFAULT
from oculomotor.models.sensory_models import retina as retina_mod
from oculomotor.analysis import (ax_fmt, ni_net, extract_spv_states, extract_sg,
                                 read_brain_decoded)
from oculomotor.benchmarks.bench_metrics import Metric

SHOW = '--show' in sys.argv
DT   = 0.001

# Eccentricities held (deg, yaw). Both signs so the fit reports a symmetric slope
# and any left/right asymmetry is visible in the scatter rather than averaged away.
ECCENTRICITIES = [-40.0, -30.0, -20.0, -10.0, 0.0, 10.0, 20.0, 30.0, 40.0]

T_STEP  = 0.1    # s — target jumps to the eccentric position
T_HOLD  = 6.0    # s — total record
T_MEAS  = 2.0    # s — start of the measurement window (after the saccade settles)

# The strobed target (both conditions): the standard strobe from sim.stimuli.
FLASH_HZ  = STROBE_HZ_DEFAULT    # 1 Hz
FLASH_MS  = STROBE_MS_DEFAULT    # 20 ms
FLASH_PERIOD = 1.0 / FLASH_HZ   # s
FLASH_ACQ = 1.0                  # s — target continuously on until here (acquisition)

THETA = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_pos=0.0,
                                sigma_vel=0.0, sigma_slip=0.0), sigma_acc=0.0)

C_LIGHT = '#2166ac'
C_DARK  = '#762a83'


def _target_at(t_np, deg):
    """Stationary target at `deg` yaw, stepping there at T_STEP."""
    T  = len(t_np)
    pt = np.zeros((T, 3), np.float32)
    pt[:, 0] = np.where(t_np < T_STEP, 0.0, np.tan(np.radians(deg))).astype(np.float32)
    pt[:, 2] = 1.0
    return pt


def _flash_train(t_np):
    """Strobed target: continuous until FLASH_ACQ (so the eye acquires it), then a
    genuine flash train (stimuli.strobe_train defaults) — absent between flashes."""
    p = strobe_train(t_np, FLASH_HZ, FLASH_MS, t0=FLASH_ACQ)
    p[t_np < FLASH_ACQ] = 1.0
    return p


def _simulate_hold(deg, lit, t_np, present=None, key=0):
    """Raw simulation of an eccentric hold — returns the full SimState.

    present: (T,) target_present schedule. None → the strobed target (_flash_train).
    """
    T = len(t_np)
    scene = np.ones(T, np.float32) if lit else np.zeros(T, np.float32)
    if present is None:
        present = _flash_train(t_np)
    return simulate(THETA, t_np,
                    target=km.build_target(t_np, lin_pos=_target_at(t_np, deg),
                                           lin_vel=np.zeros((T, 3), np.float32)),
                    scene_present_array=scene,
                    target_present_array=np.asarray(present, np.float32),
                    max_steps=int(T * 1.1) + 1000,
                    sim_config=SimConfig(warmup_s=0.0),
                    return_states=True,
                    key=jax.random.PRNGKey(key))


def _run_hold(deg, lit, key=0):
    """Hold gaze at `deg`; return (t, eye_yaw, spv_yaw, ni_yaw).

    The target is strobed (a flash train) in BOTH conditions so a stationary dot
    carries no usable motion signal into the pursuit channel; the light/dark
    difference is then purely the full-field scene.
    """
    t_np = np.arange(0.0, T_HOLD, DT)
    st = _simulate_hold(deg, lit, t_np, key=key)
    # eye='version' (not the 'left' default): gaze holding is a CONJUGATE function,
    # and the SPV must describe the same signal as the plotted trace. A per-eye SPV
    # would fold any disconjugate drift into a "gaze holding" number.
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    spv = extract_spv_states(st, t_np, eye='version')[:, 0]
    return t_np, eye, spv, ni_net(st)[:, 0]


def _drift_fit(ecc, spv):
    """Least-squares fit SPV = -k·position + b over eccentricity.

    Returns (k, b) with k POSITIVE for centripetal drift (the leaky-integrator
    sign): at +30° a leak drives the eye back toward centre, i.e. negative SPV.
    k = 1/τ_eff, so τ_eff = 1/k is the effective gaze-holding time constant.
    """
    ecc = np.asarray(ecc, float)
    spv = np.asarray(spv, float)
    ok  = np.isfinite(spv)
    if ok.sum() < 2:
        return float('nan'), float('nan')
    slope, intercept = np.polyfit(ecc[ok], spv[ok], 1)
    return float(-slope), float(intercept)


# ── Figure 1: slow-phase drift vs eye position, light vs dark ────────────────

def _drift_vs_position(show):
    meas = {}          # (lit, deg) -> mean SPV over the hold window
    net  = {}          # (lit, deg) -> net gaze displacement rate incl. quick phases
    traces = {}        # (lit, deg) -> (t, eye, spv, ni)
    for lit in (True, False):
        for deg in ECCENTRICITIES:
            t_np, eye, spv, ni = _run_hold(deg, lit)
            win = t_np >= T_MEAS
            # Mean over VALID slow-phase samples only — extract_spv_states masks
            # quick phases, and averaging a fast phase in would swamp the drift.
            vals = spv[win][np.isfinite(spv[win])]
            meas[(lit, deg)] = float(np.mean(vals)) if len(vals) else float('nan')
            # Net rate INCLUDING quick phases. SPV alone cannot distinguish "gaze
            # wanders off" from "gaze is held by quick phases resetting a drift" —
            # the two have very different functional consequences, and here they
            # differ by more than 10x, so both are worth recording.
            net[(lit, deg)] = float((eye[-1] - eye[int(T_MEAS / DT)])
                                    / (t_np[-1] - T_MEAS))
            traces[(lit, deg)] = (t_np, eye, spv, ni)

    spv_light = [meas[(True,  d)] for d in ECCENTRICITIES]
    spv_dark  = [meas[(False, d)] for d in ECCENTRICITIES]
    k_light, b_light = _drift_fit(ECCENTRICITIES, spv_light)
    k_dark,  b_dark  = _drift_fit(ECCENTRICITIES, spv_dark)

    fig = plt.figure(figsize=(15, 10))
    gs  = fig.add_gridspec(2, 2, hspace=0.32, wspace=0.22, height_ratios=[1, 1.15])
    ax_dk = fig.add_subplot(gs[0, 0])
    ax_lt = fig.add_subplot(gs[0, 1], sharey=ax_dk)
    ax_sc = fig.add_subplot(gs[1, :])

    def _tc_label(k):
        """τ_eff = 1/k, but only a leak has one. A negative k means the drift runs
        AWAY from the null, where '1/k seconds' would be a meaningless number."""
        if k > 1e-6:
            return f'τ_eff = {1.0 / k:.0f} s (centripetal)'
        if k < -1e-6:
            return f'CENTRIFUGAL k = {k:.4f}/s'
        return 'no drift'

    fig.suptitle(
        'Gaze Holding — Slow-Phase Drift vs Eye Position\n'
        f'Eccentric fixation {min(ECCENTRICITIES):.0f}…{max(ECCENTRICITIES):.0f}°, '
        f'noiseless.  Dark: {_tc_label(k_dark)}  |  Light: {_tc_label(k_light)}',
        fontsize=12, fontweight='bold')

    cmap = plt.get_cmap('coolwarm')
    cols = {d: cmap(i / (len(ECCENTRICITIES) - 1)) for i, d in enumerate(ECCENTRICITIES)}

    for ax, lit, title in ((ax_dk, False, 'DARK + strobed target — integrator alone'),
                           (ax_lt, True,  'LIGHT — vision (OKR) assists holding')):
        for d in ECCENTRICITIES:
            t_np, eye, _, _ = traces[(lit, d)]
            ax.plot(t_np, eye, color=cols[d], lw=1.2)
            ax.axhline(d, color=cols[d], lw=0.6, ls=':', alpha=0.5)
        ax.axvline(T_MEAS, color='k', lw=0.8, ls='--', alpha=0.4)
        ax.text(T_MEAS, ax.get_ylim()[1], ' measurement window', fontsize=7,
                va='top', ha='left', color='#555555')
        ax.set_title(title, fontsize=10)
        ax_fmt(ax, ylabel='Eye yaw (deg)', xlabel='Time (s)')
        ax.grid(True, alpha=0.15)

    ax_sc.axhline(0, color='k', lw=0.5)
    ax_sc.axvline(0, color='k', lw=0.5)
    ecc_fine = np.linspace(min(ECCENTRICITIES), max(ECCENTRICITIES), 100)
    for spvs, k, b, col, lbl in ((spv_dark,  k_dark,  b_dark,  C_DARK,
                                  f'dark + strobed target  (k={k_dark:+.4f}/s — {_tc_label(k_dark)})'),
                                 (spv_light, k_light, b_light, C_LIGHT,
                                  f'light  (k={k_light:+.4f}/s — {_tc_label(k_light)})')):
        ax_sc.plot(ECCENTRICITIES, spvs, 'o', color=col, ms=7, label=lbl)
        ax_sc.plot(ecc_fine, -k * ecc_fine + b, color=col, lw=1.2, ls='--', alpha=0.7)
    ax_sc.set_title('Slow-phase velocity vs eye position.  Negative SPV at positive gaze = '
                    'centripetal (normal, leaky integrator, k > 0);  positive SPV at positive '
                    'gaze = centrifugal (over-compensating, k < 0)',
                    fontsize=10)
    ax_fmt(ax_sc, ylabel='Slow-phase velocity (deg/s)', xlabel='Eye position (deg)')
    ax_sc.grid(True, alpha=0.15)
    ax_sc.legend(fontsize=9)

    # gridspec already carries the spacing; tight_layout would fight the twin-free
    # 2x2-with-spanning-row layout and warns.
    path, rp = utils.save_fig(fig, 'gaze_holding_drift', show=show, params=THETA,
                              conditions='Eccentric fixation, light vs dark (strobed target), noiseless')

    spv_ecc_dark  = max(abs(meas[(False, ECCENTRICITIES[0])]),
                        abs(meas[(False, ECCENTRICITIES[-1])]))
    spv_ecc_light = max(abs(meas[(True,  ECCENTRICITIES[0])]),
                        abs(meas[(True,  ECCENTRICITIES[-1])]))
    net_ecc_dark  = max(abs(net[(False, ECCENTRICITIES[0])]),
                        abs(net[(False, ECCENTRICITIES[-1])]))
    print(f'      dark : k={k_dark:+.5f}/s ({_tc_label(k_dark)})  '
          f'|SPV|@{max(ECCENTRICITIES):.0f}deg={spv_ecc_dark:.3f}  net={net_ecc_dark:.3f} deg/s')
    print(f'      light: k={k_light:+.5f}/s ({_tc_label(k_light)})  '
          f'|SPV|@{max(ECCENTRICITIES):.0f}deg={spv_ecc_light:.3f} deg/s')
    print('      position -> SPV dark, SPV light, net dark (deg/s):')
    for d in ECCENTRICITIES:
        print(f'        {d:+6.0f}  {meas[(False, d)]:+8.3f}  {meas[(True, d)]:+8.3f}  '
              f'{net[(False, d)]:+8.3f}')
    ratio = (spv_ecc_light / spv_ecc_dark) if spv_ecc_dark > 1e-9 else float('nan')
    metrics = [
        # SIGNED slope, positive = centripetal, which is the only physiological
        # direction for a normal integrator: it leaks toward the null, it does not
        # push away from it. Banding |k| instead would let a centrifugal drift —
        # an over-compensating integrator, a cerebellar sign — pass silently.
        # hi=0.05 is the leak limit (tau_eff >= 20 s); lo=-0.01 tolerates numerical
        # slop around zero but not a real centrifugal drift.
        Metric('gaze_hold_drift_slope_dark', k_dark,
               lo=-0.01, hi=0.05, golden_tol=0.25, units='1/s',
               cite='Cannon & Robinson (1987); Becker & Klein (1973)',
               desc='Slow-phase drift per degree of eccentricity in the dark, SIGNED '
                    '(+ = centripetal/leaky = normal, − = centrifugal). |k| = 1/τ_eff, '
                    'so the hi bound is τ_eff ≥ 20 s'),
        Metric('gaze_hold_spv_ecc_dark', spv_ecc_dark,
               lo=None, hi=4.0, golden_tol=0.3, units='deg/s',
               cite='Becker & Klein (1973)',
               desc=f'|slow-phase velocity| at {max(ECCENTRICITIES):.0f}° eccentric gaze in the '
                    f'dark — normals drift only a few deg/s'),
        Metric('gaze_hold_net_drift_dark', net_ecc_dark,
               lo=None, hi=1.0, golden_tol=0.3, units='deg/s',
               cite='Becker & Klein (1973)',
               desc=f'NET gaze displacement rate at {max(ECCENTRICITIES):.0f}° in the dark, quick '
                    f'phases included — whether gaze is actually held, not just how fast it drifts '
                    f'between resets'),
        Metric('gaze_hold_spv_ecc_light', spv_ecc_light,
               lo=None, hi=2.0, golden_tol=0.3, units='deg/s',
               cite='Becker & Klein (1973)',
               desc=f'|slow-phase velocity| at {max(ECCENTRICITIES):.0f}° eccentric gaze in the light'),
        # Vision must not make gaze holding WORSE. With a stationary full-field
        # scene, drift produces retinal slip that OKR nulls, so light ≤ dark.
        Metric('gaze_hold_light_dark_ratio', ratio,
               lo=None, hi=1.0, golden_tol=0.25, units='',
               cite='Becker & Klein (1973)',
               desc='Eccentric |SPV| in light ÷ in dark — must be ≤ 1: a visible scene gives OKR '
                    'a slip signal to null, so vision can only help holding'),
        Metric('gaze_hold_center_spv_dark', abs(meas[(False, 0.0)]),
               lo=None, hi=1.0, golden_tol=0.3, units='deg/s',
               cite='Becker & Klein (1973)',
               desc='|SPV| at straight-ahead gaze in the dark — must be ~0: drift at the null is '
                    'a spontaneous nystagmus, not an integrator leak'),
    ]

    fm = utils.fig_meta(path, rp,
        title='Gaze Holding — Drift vs Eye Position (light vs dark)',
        description='Eccentric fixation held at −40…+40°, in the light and in the dark with a '
                    'strobed target. Top: eye position traces per eccentricity. Bottom: mean '
                    'slow-phase velocity vs eye position, with the fitted drift slope k = 1/τ_eff.',
        expected='Centripetal drift growing linearly with eccentricity, steeper in the dark than '
                 'in the light (OKR corrects retinal slip when the scene is visible). '
                 'No drift at straight-ahead. Dark τ_eff > ~20 s in normals; a leaky integrator '
                 'shortens it and produces gaze-evoked nystagmus.',
        citation='Becker & Klein (1973) Vision Res 13:1021; Cannon & Robinson (1987) J Neurophysiol 57:1383',
        fig_type='behavior')
    fm['metrics'] = metrics
    return fm


# ── Figure 2: flashed-target cascade (debug) ─────────────────────────────────
# A genuinely flashed target is the clean way to probe gaze holding: between
# flashes there is no visual feedback at all, so the integrator runs open-loop.
# This panel traces one 40° hold through the whole chain to show where the flash
# survives, where it is attenuated, and what holds the position between flashes.

FLASH_DEG    = 40.0
FLASH_TEND   = 10.0


def _flash_cascade(show):
    from oculomotor.models.brain_models import perception_cyclopean as pc_mod

    t_np   = np.arange(0.0, FLASH_TEND, DT)
    flash  = _flash_train(t_np)
    st     = _simulate_hold(FLASH_DEG, lit=False, t_np=t_np, present=flash)

    eye  = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    ret  = st.sensory.retina_L
    tv_ret  = np.array(ret.target_visible)[:, -1]      # last cascade stage = delayed
    tp_ret  = np.array(jax.vmap(retina_mod.read_outputs)(ret).target_pos)[:, 0]   # delayed retinal target yaw
    x_cyc   = np.array(jax.vmap(pc_mod.to_array)(st.brain.pc))
    tv_cyc  = (x_cyc @ np.array(pc_mod.C_target_visible).T)[:, 0]
    e_pd    = np.array(jax.vmap(pc_mod.read_activations)(st.brain.pc).target_pos)[:, 0]
    mem     = np.array(st.brain.pt.mem_pos)[:, 0]
    trust   = np.array(st.brain.pt.mem_age)
    sg      = extract_sg(st, THETA)
    # Motion channels — the check that the strobe actually suppresses MOTION and
    # not merely position. eye_vel is the cause of retinal slip, so plotting it
    # alongside shows the retina reporting ~no motion even while the eye moves.
    tvel_ret  = np.array(ret.target_vel)[:, -3]
    tvel_cyc  = (x_cyc @ np.array(pc_mod.C_vel).T)[:, 0]
    slip_cyc  = (x_cyc @ np.array(pc_mod.C_slip).T)[:, 0]
    pu_net    = np.array(read_brain_decoded(st, THETA).pu.net)[:, 0]
    eye_vel   = np.gradient(eye, DT)

    rows = 7
    fig, axes = plt.subplots(rows, 1, figsize=(14, 2.3 * rows), sharex=True)
    fig.suptitle(
        f'Gaze-Holding Cascade — {FLASH_DEG:.0f}° eccentric, DARK, target flashed '
        f'{FLASH_MS:.0f} ms every {FLASH_PERIOD:.0f} s\n'
        f'(continuous until {FLASH_ACQ:.0f} s to acquire, then genuinely absent between flashes)',
        fontsize=12, fontweight='bold')

    for ax in axes:                       # flash times on every row
        for k in np.where(np.diff(flash) > 0)[0]:
            ax.axvline(t_np[k], color=utils.C['target'], lw=0.7, alpha=0.35)
        ax.grid(True, alpha=0.15)

    axes[0].plot(t_np, flash,  color=utils.C['target'], lw=1.2, label='target_present (input)')
    axes[0].plot(t_np, tv_ret, color=utils.C['eye'],    lw=1.5, label='retina delayed target_visible')
    axes[0].plot(t_np, tv_cyc, color=utils.C['vs'],     lw=1.2, ls='--', label='cyclopean target_visible')
    axes[0].set_title(f'Visibility — a {FLASH_MS:.0f} ms flash reaches only '
                      f'{tv_ret[t_np > FLASH_ACQ + FLASH_PERIOD].max():.2f} of full visibility through the '
                      f'6-stage cascade', fontsize=9)
    ax_fmt(axes[0], ylabel='visible [0,1]'); axes[0].legend(fontsize=7)

    axes[1].axhline(FLASH_DEG, color='k', lw=0.6, ls=':', label='true target (40°)')
    axes[1].plot(t_np, tp_ret, color=utils.C['eye'], lw=1.4, label='retina delayed target_pos (yaw)')
    axes[1].plot(t_np, e_pd,   color=utils.C['vs'],  lw=1.2, ls='--', label='cyclopean pos error e_pd')
    # These are RETINAL signals — i.e. target-minus-eye ERROR, not world position.
    # Once the eye is on target the error is ~0 by definition, so each flash samples
    # "no error". Zoomed to ±2°: the 40° acquisition transient runs off-scale, which
    # is the whole point — the post-acquisition detail is what matters here.
    axes[1].set_ylim(-2.0, 2.0)
    axes[1].set_title('Retinal position ERROR (target − eye), zoomed ±2° — the 40° acquisition '
                      'transient is off-scale. After acquisition each flash samples a near-zero '
                      'error, because the eye is already on target', fontsize=9)
    ax_fmt(axes[1], ylabel='deg'); axes[1].legend(fontsize=7, loc='lower right')

    # ── Motion channel: is the strobe suppressing MOTION, not just position? ──
    axes[2].plot(t_np, eye_vel,  color=utils.C['dark'],  lw=0.9, alpha=0.7,
                 label='eye velocity (cause of retinal slip)')
    axes[2].plot(t_np, tvel_ret, color=utils.C['eye'],   lw=1.5, label='retina delayed target_vel')
    axes[2].plot(t_np, tvel_cyc, color=utils.C['vs'],    lw=1.2, ls='--', label='cyclopean target_vel')
    axes[2].plot(t_np, slip_cyc, color=utils.C['scene'], lw=1.0, ls=':', label='cyclopean scene slip (dark ⇒ 0)')
    axes[2].plot(t_np, pu_net,   color=utils.C['pursuit'], lw=1.3, label='pursuit net drive (downstream)')
    axes[2].set_ylim(-5.0, 5.0)
    axes[2].set_title('Retinal MOTION (clipped ±5; the acquisition saccade runs off-scale) — with the '
                      'target absent between flashes the retina reports no target motion even while '
                      'the eye moves, so nothing drives pursuit', fontsize=9)
    ax_fmt(axes[2], ylabel='deg/s'); axes[2].legend(fontsize=7, loc='lower right', ncol=2)

    axes[3].plot(t_np, mem,   color=utils.C['pursuit'], lw=1.6, label='target working memory mem_pos (yaw)')
    axes[3].plot(t_np, trust, color='#d94801',          lw=1.2, ls='--', label='mem_age [0,1]')
    axes[3].set_title('Target working memory (FEF/dlPFC) — the candidate for holding target '
                      'position across the dark gaps. NOTE it stores a RETINAL error, which is '
                      '~0 once foveated, so here it decays after acquisition and trust stays low',
                      fontsize=9)
    ax_fmt(axes[3], ylabel='deg  /  trust'); axes[3].legend(fontsize=7)

    axes[4].plot(t_np, sg['e_held'][:, 0], color=utils.C['vs'],    lw=1.5, label='e_held (sample-hold)')
    axes[4].plot(t_np, sg['z_acc'],        color='#e08214',        lw=1.2, label='z_acc (accumulator)')
    axes[4].plot(t_np, sg['z_opn'] / 100,  color='#1b7837',        lw=1.0, ls='--', label='OPN (norm)')
    # Clipped: the OPN latch takes a large negative excursion during the acquisition
    # saccade, which otherwise compresses every other trace to a flat line.
    axes[4].set_ylim(-2.0, 2.0)
    axes[4].set_title('Saccade generator (clipped ±2; the acquisition-saccade OPN excursion runs '
                      'off-scale) — does each flash trigger a corrective saccade?', fontsize=9)
    ax_fmt(axes[4], ylabel='deg / gate'); axes[4].legend(fontsize=7, loc='lower right')

    axes[5].plot(t_np, sg['u_burst'][:, 0], color=utils.C['burst'], lw=1.3, label='burst')
    axes[5].plot(t_np, sg['x_ni'][:, 0],    color=utils.C['ni'],    lw=1.3, ls='--', label='NI (x_ni)')
    axes[5].set_title('Burst + neural integrator', fontsize=9)
    ax_fmt(axes[5], ylabel='deg/s  /  deg'); axes[5].legend(fontsize=7)

    axes[6].axhline(FLASH_DEG, color='k', lw=0.6, ls=':')
    axes[6].plot(t_np, eye, color=utils.C['eye'], lw=1.5, label='eye (version)')
    axes[6].set_title('Eye position — between flashes the integrator holds open-loop', fontsize=9)
    ax_fmt(axes[6], ylabel='Eye yaw (deg)', xlabel='Time (s)')
    axes[6].set_ylim(FLASH_DEG - 3, FLASH_DEG + 3); axes[6].legend(fontsize=7)

    path, rp = utils.save_fig(fig, 'gaze_holding_flash_cascade', show=show, params=THETA,
                              conditions=f'{FLASH_DEG:.0f}° eccentric, dark, {FLASH_MS:.0f} ms '
                                         f'flash every {FLASH_PERIOD:.0f} s, noiseless')
    post = t_np > FLASH_ACQ + FLASH_PERIOD      # clear of the acquisition transient
    print(f'      flash cascade: peak delayed target_visible = '
          f'{tv_ret[post].max():.3f}  (1.0 = fully visible)')
    print(f'      motion check (post-acquisition): max |retina target_vel| = '
          f'{np.abs(tvel_ret[post]).max():.4f} deg/s, max |cyclopean target_vel| = '
          f'{np.abs(tvel_cyc[post]).max():.4f}, max |pursuit net| = '
          f'{np.abs(pu_net[post]).max():.4f} deg/s')
    return utils.fig_meta(path, rp,
        title='Gaze-Holding Cascade — flashed target (debug)',
        description=f'One 40° eccentric hold in the dark with the target genuinely flashed '
                    f'({FLASH_MS:.0f} ms every {FLASH_PERIOD:.0f} s). Rows: visibility through the '
                    f'retinal cascade, the position signal it carries, the retinal MOTION channel '
                    f'(the check that the strobe suppresses motion, not just position), target '
                    f'working memory, the saccade generator, burst + NI, and eye position.',
        expected='Retinal target motion stays ~0 between flashes even as the eye moves, so nothing '
                 'drives pursuit — the strobe suppresses motion, not merely position. '
                 'Each flash produces an attenuated visibility pulse and a brief position sample; '
                 'working memory holds the target across the dark gaps; the integrator holds gaze '
                 'open-loop between flashes. Flash duration matters: a 20 ms pulse reaches ~0.40 of '
                 'full visibility through the 6-stage cascade, a 10 ms pulse only ~0.21.',
        citation='Becker & Klein (1973) Vision Res 13:1021',
        fig_type='cascade')


SECTION = dict(
    id='gaze_holding', title='7. Gaze Holding',
    description='Eccentric gaze holding — the neural integrator plus its cerebellar (floccular) '
                'extension. Slow-phase drift is measured as a function of eye position, in the '
                'light and in the dark with a strobed target; the slope of that line is 1/τ_eff, '
                'the effective gaze-holding time constant. Pathological leak (gaze-evoked '
                'nystagmus) is benchmarked separately under Clinical.',
)


def run(show=False):
    print('\n=== Gaze Holding ===')
    figs = []
    print(f'  1/2  drift vs eye position ({len(ECCENTRICITIES)} positions x light/dark) …')
    figs.append(_drift_vs_position(show))
    print(f'  2/2  flashed-target cascade ({FLASH_DEG:.0f}°, {FLASH_MS:.0f} ms flashes) …')
    figs.append(_flash_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
