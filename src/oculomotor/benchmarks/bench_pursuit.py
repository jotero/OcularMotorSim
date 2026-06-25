"""Smooth pursuit benchmarks — velocity range, sinusoidal, signal cascade.

Usage:
    python -X utf8 scripts/bench_pursuit.py
    python -X utf8 scripts/bench_pursuit.py --show
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
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
)
from oculomotor.sim import kinematics as km
from oculomotor.models.brain_models.perception_cyclopean import C_pos  # noqa: F401
from oculomotor.analysis import (ax_fmt, extract_burst, extract_sg, ni_net,
                                 read_brain_decoded, read_brain_acts, extract_spv_states)
from oculomotor.benchmarks.bench_metrics import Metric
from oculomotor.benchmarks import bode

SHOW  = '--show' in sys.argv
DT    = 0.001
THETA = PARAMS_DEFAULT
THETA_NOISELESS = with_brain(with_sensory(THETA, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)


def _ramp(t_np, vel, t_jump=0.2):
    T = len(t_np)
    tgt = np.where(t_np >= t_jump, vel * (t_np - t_jump), 0.0)
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt))
    vt3 = np.zeros((T, 3))
    vt3[:, 0] = np.where(t_np >= t_jump, float(vel), 0.0).astype(np.float32)
    return tgt, jnp.array(pt3), jnp.array(vt3)


def _run(theta, t_np, pt3, vt3=None, target_present=True, key=0):
    t  = jnp.array(t_np)
    T  = len(t)
    tp = jnp.ones(T) if target_present else jnp.zeros(T)
    return simulate(theta, t,
                    target=km.build_target(t_np, lin_pos=np.array(pt3)),
                    scene_present_array=jnp.ones(T), target_present_array=tp,
                    max_steps=int(len(t_np) * 1.05) + 500,
                    return_states=True, key=jax.random.PRNGKey(key))


# ── Figure 1: velocity range comparison ──────────────────────────────────────

def _velocity_range(show):
    velocities = [5.0, 10.0, 20.0, 40.0]
    T_end, T_jump = 3.0, 0.2
    t_np = np.arange(0.0, T_end, DT)

    theta_pur = THETA
    theta_nop = with_brain(THETA, K_pursuit=0.0, K_phasic_pursuit=0.0)

    n_rows, n_cols = 3, len(velocities)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.8 * n_cols, 2.5 * n_rows), sharex=True)
    fig.suptitle('Smooth Pursuit — Velocity Range  (blue: pursuit on, gray: saccades only)', fontsize=11)
    row_labels = ['Position (deg)', 'Eye velocity (deg/s)', 'Pursuit drive u_pursuit (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    spv_by_vel = {}   # vel -> slow-phase eye velocity (for gain + latency metrics)
    for ci, vel in enumerate(velocities):
        tgt, pt3, vt3 = _ramp(t_np, vel, T_jump)
        st_pur = _run(theta_pur, t_np, pt3, vt3, key=ci)
        st_nop = _run(theta_nop, t_np, pt3,      key=ci + 10)

        eye_pur = (np.array(st_pur.plant.left[:, 0]) + np.array(st_pur.plant.right[:, 0])) / 2.0
        eye_nop = (np.array(st_nop.plant.left[:, 0]) + np.array(st_nop.plant.right[:, 0])) / 2.0
        ev_pur  = np.gradient(eye_pur, DT)
        ev_nop  = np.gradient(eye_nop, DT)
        spv_by_vel[vel] = extract_spv_states(st_pur, t_np)[:, 0]
        u_pur   = np.array(read_brain_decoded(st_pur, THETA).pu.net[:, 0])     # NET yaw (deg/s)

        axes[0, ci].set_title(f'{vel:.0f} deg/s', fontsize=10)
        for ax in axes[:, ci]:
            ax.axvline(T_jump, color='gray', lw=0.5, ls='--', alpha=0.4)

        axes[0, ci].plot(t_np, tgt,     color=utils.C['target'],    lw=1.5, label='target')
        axes[0, ci].plot(t_np, eye_nop, color=utils.C['dark'],      lw=1.0, ls='--', label='no pursuit')
        axes[0, ci].plot(t_np, eye_pur, color=utils.C['eye'],       lw=1.5, label='pursuit')
        ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=7)

        axes[1, ci].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7,
                            label=f'{vel} deg/s')
        axes[1, ci].plot(t_np, ev_nop, color=utils.C['dark'], lw=0.8, ls='--', label='no pursuit')
        axes[1, ci].plot(t_np, ev_pur, color=utils.C['eye'],  lw=1.2, label='pursuit')
        axes[1, ci].set_ylim(-max(vel*0.15, 3), vel * 1.35)
        ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=7)

        K_ph  = float(THETA.brain.K_phasic_pursuit)
        K_int = float(THETA.brain.K_pursuit)
        tau_p = float(THETA.brain.tau_pursuit)
        dx_pur     = np.gradient(u_pur, DT)
        e_pred_est = (dx_pur + u_pur / tau_p) / K_int
        phasic     = K_ph * e_pred_est
        u_total    = u_pur + phasic

        axes[2, ci].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7)
        axes[2, ci].plot(t_np, u_total, color=utils.C['pursuit'], lw=1.8, label='u_pursuit (total)')
        axes[2, ci].plot(t_np, u_pur,   color='#1a6ebd',          lw=1.1, ls='--', label='integrator x_p')
        axes[2, ci].plot(t_np, phasic,  color='#d94801',          lw=1.1, ls='--', label='phasic K·e_pred')
        axes[2, ci].set_ylim(-vel * 0.1, vel * 1.35)
        ax_fmt(axes[2, ci])
        axes[2, ci].set_xlabel('Time (s)', fontsize=8)
        if ci == 0: axes[2, ci].legend(fontsize=7)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_velocity_range', show=show, params=THETA,
                              conditions='Lit, foveal target ramping at 5–80 °/s (constant velocity pursuit)')

    # ── Metrics: steady-state pursuit gain (SPV/target) + onset latency ───────
    ss = t_np > 1.5
    def _gain(v):
        return float(np.mean(spv_by_vel[v][ss]) / v)
    # Onset latency at 10 deg/s: time from ramp onset to SPV crossing 2 deg/s.
    spv10 = spv_by_vel[10.0]
    after = t_np >= T_jump
    cross = np.where(after & (spv10 > 2.0))[0]
    latency_ms = float((t_np[cross[0]] - T_jump) * 1000.0) if len(cross) else float('nan')
    metrics = [
        Metric('pursuit_ss_gain_5degs', _gain(5.0), 
               lo=0.8, hi=1.1, golden_tol=0.08, units='',
               cite='Lisberger & Westbrook (1985)',
               desc='Steady-state pursuit gain (SPV ÷ target vel) at 5 deg/s ramp'),
        Metric('pursuit_ss_gain_10degs', _gain(10.0), 
               lo=0.8, hi=1.1, golden_tol=0.08, units='',
               cite='Lisberger & Westbrook (1985)',
               desc='Steady-state pursuit gain at 10 deg/s ramp'),
        Metric('pursuit_latency_ms', latency_ms, 
               lo=60.0, hi=180.0, golden_tol=0.2, units='ms',
               cite='Rashbass (1961); Carl & Gellman (1987)',
               desc='Pursuit onset latency at 10 deg/s (ramp onset → SPV > 2 deg/s)'),
    ]
    fig_meta = utils.fig_meta(path, rp,
        title='Smooth Pursuit — Velocity Range',
        description='Step-ramp target at 5, 10, 20, 40 deg/s. '
                    'Blue: smooth pursuit enabled. Gray: saccades only (pursuit off). '
                    'Rows: position, velocity, pursuit drive (total + integrator + phasic).',
        expected='At 5–10 deg/s: pursuit tracks closely, steady-state gain > 0.8. '
                 'At higher velocities: catch-up saccades + partial pursuit.',
        citation='Lisberger & Westbrook (1985) J Neurosci; Rashbass (1961)',
        fig_type='behavior')
    fig_meta['metrics'] = metrics
    return fig_meta


# ── Figure 2: pursuit frequency response (Bode) ──────────────────────────────

def _bode(show):
    """Pursuit Bode: sinusoidal target-velocity sweep, NOISELESS.
    Gain = eye-velocity (SPV) ÷ target-velocity amplitude; phase lag vs frequency.
    Replaces the old (messy) sinusoidal-pursuit figure.
    """
    AMP     = 10.0   # deg/s peak target velocity (above the position-cap knee)
    POS_MAX = 20.0   # deg — cap peak target eccentricity so it stays foveatable.
                     # The one-sided (1−cos) target peaks at 2·V/w, so cap V at the
                     # POS_MAX/2 amplitude (peak = POS_MAX). Without this the 0.1 Hz
                     # point puts the target at 32° (out of range) and the gain craters.
    FREQS  = np.array([0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0])
    N_CYC  = 5
    SETTLE = 1.5

    def run_fn(f):
        T_end = min(SETTLE + N_CYC / f, 45.0)
        t  = np.arange(0.0, T_end, DT)
        Tn = len(t)
        w  = 2 * np.pi * f
        V  = bode.capped_velocity_amp(f, AMP, POS_MAX / 2.0)
        on = t >= SETTLE
        vel = np.where(on, V * np.sin(w * (t - SETTLE)), 0.0)
        pos = np.where(on, -(V / w) * (np.cos(w * (t - SETTLE)) - 1.0), 0.0)
        pt3 = np.zeros((Tn, 3)); pt3[:, 2] = 1.0; pt3[:, 0] = np.tan(np.radians(pos))
        vt3 = np.zeros((Tn, 3)); vt3[:, 0] = vel.astype(np.float32)
        st  = _run(THETA_NOISELESS, t, jnp.array(pt3), jnp.array(vt3), key=0)
        eye_spv, slow = extract_spv_states(st, t, return_mask=True)
        return t, vel, eye_spv[:, 0], slow      # gap-aware fit over valid slow-phase samples

    freqs, gains, phases = bode.bode_sweep(run_fn, FREQS, settle_frac=0.45)
    fig, m = bode.make_bode_figure(
        freqs, gains, phases,
        'Smooth Pursuit — Frequency Response (Bode, noiseless)',
        ref_hz=0.5, gain_label='Gain (eye vel ÷ target vel)',
        expected=bode.expected_bode('lowpass', fc=2.0, g0=1.0,
            label='expected: 1st-order LP, ~2 Hz cutoff (Robinson 1965; Krauzlis & Lisberger 1994)'))
    path, rp = utils.save_fig(fig, 'pursuit_bode', show=show, params=THETA_NOISELESS,
        conditions='Lit, NOISELESS — sinusoidal target-velocity sweep 0.1–2 Hz (10 deg/s peak)')
    metrics = [
        Metric('pursuit_bode_gain_max', float(m['gain_max']),
               lo=0.7, hi=1.1, golden_tol=0.1, units='',
               cite='Lisberger et al. (1981)',
               desc='Pursuit peak gain'),
    ]
    if m['fc_hi'] is not None:
        metrics.append(
            Metric('pursuit_bode_fc_hi', float(m['fc_hi']),
                   lo=0.3, hi=None, golden_tol=0.25, units='Hz',
                   cite='Lisberger et al. (1981)',
                   desc='Pursuit −3 dB bandwidth (Hz)'))
    fm = utils.fig_meta(path, rp,
        title='Smooth Pursuit — Bode (frequency response)',
        description='Sinusoidal target-velocity sweep (0.1–2 Hz, 10 deg/s peak), NOISELESS. '
                    'Gain = eye-velocity (SPV) ÷ target-velocity; phase lag vs frequency.',
        expected='Gain ≈ 1 at low f, −3 dB near ~1 Hz; phase lag grows with frequency.',
        citation='Lisberger, Evinger, Johanson & Fuchs (1981) J Neurophysiol',
        fig_type='behavior')
    fm['metrics'] = metrics
    return fm


# ── Figure 3: pursuit signal cascade ─────────────────────────────────────────

def _cascade(show):
    """Signal cascade for 20 deg/s pursuit: target on 0.2–2.0 s, then target stops."""
    vel    = 20.0
    t_jump = 0.2
    t_stop = 2.0   # target stops; eye keeps moving (pursuit integrator decays slowly)
    T_end  = 5.0
    t_np   = np.arange(0.0, T_end, DT)
    T      = len(t_np)

    # Position: ramp until t_stop, then hold at final position
    tgt_pos = float(vel * (t_stop - t_jump))
    tgt = np.where(t_np < t_jump, 0.0,
          np.where(t_np < t_stop, vel * (t_np - t_jump), tgt_pos))
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt))
    # Velocity: on during ramp, zero after target stops
    vt3 = np.zeros((T, 3))
    vt3[:, 0] = np.where((t_np >= t_jump) & (t_np < t_stop), float(vel), 0.0).astype(np.float32)

    st  = _run(THETA_NOISELESS, t_np, jnp.array(pt3), jnp.array(vt3), key=30)
    sg  = extract_sg(st, THETA)
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    ev  = np.gradient(eye, DT)
    x_pur = np.array(read_brain_decoded(st, THETA).pu.net[:, 0])  # NET yaw memory
    # Saccadic-suppression gate on the pursuit pathway — if this sits near 0
    # for most of the inter-saccade interval, the pursuit input (= sat·(slip+ec))
    # is being throttled and the pursuit integrator drains between catch-up
    # saccades (staircase eye position instead of smooth ramp).
    sat_pu = np.array(read_brain_acts(st, THETA).cb.saccadic_suppression_target)

    n_rows = 7
    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.5 * n_rows))
    fig.suptitle(f'Smooth Pursuit Signal Cascade — {vel:.0f} deg/s ramp, target stops at {t_stop:.0f} s\n'
                 f'(pursuit integrator persists after target stop, decays with τ ≈ {THETA.brain.tau_pursuit:.0f} s)',
                 fontsize=11)

    vl_start = dict(color='gray',   lw=0.7, ls='--', alpha=0.5)
    vl_stop  = dict(color='tomato', lw=0.9, ls='--', alpha=0.7)
    for ax in axes:
        ax.axvline(t_jump, **vl_start)
        ax.axvline(t_stop, **vl_stop)
        ax_fmt(ax)

    axes[0].plot(t_np, tgt,          color=utils.C['target'],  lw=1.5, label='target pos (stops 2 s)')
    axes[0].plot(t_np, eye,          color=utils.C['eye'],     lw=1.5, label='eye pos')
    axes[0].plot(t_np, ni_net(st)[:,0], color=utils.C['ni'],   lw=0.9, ls='--', label='NI')
    axes[0].set_ylabel('Position (deg)'); axes[0].set_title('Eye + Target Position')
    axes[0].legend(fontsize=8)

    axes[1].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7, label=f'{vel:.0f} deg/s')
    axes[1].plot(t_np, ev, color=utils.C['eye'], lw=1.2, label='eye vel (persists after stop)')
    axes[1].set_ylabel('Velocity (deg/s)'); axes[1].set_title('Eye Velocity')
    axes[1].legend(fontsize=8)

    # Reconstruct phasic drive from integrator state:
    #   dx_p/dt = -x_p/tau + K_int * e_pred  →  e_pred = (dx_p/dt + x_p/tau) / K_int
    #   phasic  = K_phasic * e_pred  (fast onset, decays to 0 at steady state)
    #   u_total = x_p + phasic        (full pursuit command)
    K_ph  = float(THETA.brain.K_phasic_pursuit)
    K_int = float(THETA.brain.K_pursuit)
    tau_p = float(THETA.brain.tau_pursuit)
    dx_pur     = np.gradient(x_pur, DT)
    e_pred_est = (dx_pur + x_pur / tau_p) / K_int
    phasic     = K_ph * e_pred_est
    u_total    = x_pur + phasic

    axes[2].axhline(vel, color=utils.C['target'], lw=0.8, ls=':', alpha=0.7, label=f'target {vel:.0f} deg/s')
    axes[2].plot(t_np, u_total, color=utils.C['pursuit'],  lw=1.8, label='u_pursuit (total)')
    axes[2].plot(t_np, x_pur,   color='#1a6ebd',           lw=1.3, ls='--', label='integrator x_p (slow)')
    axes[2].plot(t_np, phasic,  color='#d94801',           lw=1.3, ls='--', label='phasic K·e_pred (fast)')
    axes[2].set_ylabel('Pursuit drive (deg/s)'); axes[2].set_title('Pursuit Drive: Integrator + Phasic Feedthrough')
    axes[2].legend(fontsize=8)

    axes[3].plot(t_np, sg['e_pd'][:,0],   color='darkorange', lw=1.0, ls='--', label='e_delayed')
    axes[3].plot(t_np, sg['e_held'][:,0], color=utils.C['vs'], lw=1.8, label='e_held (frozen)')
    axes[3].set_ylabel('Error (deg)'); axes[3].set_title('Visual Cascade Output + Sample-Hold')
    axes[3].legend(fontsize=8)

    axes[4].plot(t_np, sg['z_acc'], color='#e08214', lw=1.5, label='z_acc')
    axes[4].plot(t_np, sg['z_opn'] / 100, color='#1b7837', lw=1.5, label='OPN (norm, 1=tonic)')
    axes[4].plot(t_np, sat_pu, color='#9467bd', lw=1.5, ls='--', label='sacc. supp. gate (pursuit)')
    axes[4].axhline(THETA.brain.threshold_acc, color='#e08214', lw=0.8, ls=':')
    axes[4].axhline(1.0, color='#9467bd', lw=0.6, ls=':', alpha=0.5)
    axes[4].set_ylim(-0.05, 1.15)
    axes[4].set_ylabel('Accumulator / gate'); axes[4].set_title('Catch-up Saccade Trigger + Saccadic-Suppression Gate')
    axes[4].legend(fontsize=8)

    axes[5].plot(t_np, sg['u_burst'][:,0], color=utils.C['burst'], lw=1.5, label='burst (catch-up)')
    axes[5].set_ylabel('Burst (deg/s)'); axes[5].set_title('Saccade Burst (Catch-up Saccades)')
    axes[5].legend(fontsize=8)

    axes[6].plot(t_np, sg['z_acc'], color=utils.C['refractory'], lw=1.5, label='z_acc (accumulator)')
    axes[6].axhline(0.5, color='k', lw=0.6, ls='--', alpha=0.4)
    axes[6].set_ylabel('z_acc'); axes[6].set_title('Accumulator (refractory proxy)')
    axes[6].set_xlabel('Time (s)', fontsize=9)
    axes[6].legend(fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'pursuit_cascade', show=show, params=THETA_NOISELESS,
                              conditions='Lit, foveal ramp + catch-up saccades (noiseless cascade trace)')
    return utils.fig_meta(path, rp,
        title='Smooth Pursuit Signal Cascade (Internal)',
        description='Full signal chain for 20 deg/s pursuit: target ramps 0.2–2.0 s then holds. '
                    'Pursuit integrator persists after target stop (τ ≈ 40 s). '
                    'Rows: position, velocity, pursuit drive (total/integrator/phasic), '
                    'visual error, saccade accumulator, burst, refractory state.',
        expected='Phasic drive rises fast at onset, decays to 0 at steady state. '
                 'After target stops at 2 s: phasic reverses (retinal slip now backward), '
                 'integrator decays slowly — eye coasts then corrects with catch-up saccades. '
                 'Integrator TC visible as slow drift in x_p post-stop.',
        citation='Lisberger & Westbrook (1985)',
        fig_type='cascade')


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='pursuit', title='4. Smooth Pursuit',
    description='Smooth pursuit and catch-up saccades. Tests velocity gain at multiple speeds, '
                'sinusoidal tracking, and the pursuit integrator + saccade interaction cascade. '
                'Pursuit input = K_pursuit_direct·(saccadically-gated retinal slip) + K_cereb_pu·(cerebellar '
                'EC correction); the closed-loop pursuit memory τ_eff ≈ 1.45 s (Smith predictor) despite the '
                'long pop-leak τ_pursuit = 40 s.',
)


def run(show=False):
    print('\n=== Smooth Pursuit ===')
    figs = []
    print('  1/3  velocity range …')
    figs.append(_velocity_range(show))
    print('  2/3  frequency response (Bode) …')
    figs.append(_bode(show))
    print('  3/3  signal cascade …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
