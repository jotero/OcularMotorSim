"""Saccade benchmarks — main sequence, oblique, double-step refractoriness, cascade.

Usage:
    python -X utf8 scripts/bench_saccades.py
    python -X utf8 scripts/bench_saccades.py --show
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
from oculomotor.analysis import (
    ax_fmt, extract_burst, extract_sg, ni_net, vs_net,
    read_brain_acts, read_brain_decoded, extract_z_opn,
)
from oculomotor.models.brain_models import tvor as tv_mod
from oculomotor.models.brain_models.perception_self_motion import CANAL2CARDINAL
from oculomotor.models.sensory_models.retina import (
    velocity_saturation, ypr_to_xyz, xyz_to_ypr,
)
from oculomotor.models.plant_models.readout import rotation_matrix
from oculomotor.benchmarks.bench_metrics import Metric


def _omega_tvor_traj(states, brain_params):
    """Recompute T-VOR omega over a SimState trajectory via vmap."""
    _DEG_PER_PD = 0.5729
    def _at(bs):
        aca      = brain_params.AC_A * _DEG_PER_PD * bs.va.acc_fast
        verg_yaw = bs.va.verg_fast[0] + bs.va.verg_tonic[0] + aca
        eye_pos  = CANAL2CARDINAL @ (bs.ni.L - bs.ni.R)   # canal→cardinal
        omega, _ = tv_mod.step(bs.sm.v_lin, bs.sm.a_lin, verg_yaw, eye_pos, brain_params)
        return omega
    return np.array(jax.vmap(_at)(states.brain))


def _pre_cascade_signals(states, params, dt):
    """Pre-cascade saturated slip + EC, in eye frame.

    For the saccade bench (stationary target/scene/head), retinal slip on either
    pathway equals −R_eye.T · eye_velocity_world (using the *actual* plant
    position + velocity).  The EC pre-cascade signal comes from the cerebellum's
    motor forward model: x_p_pred = ½·(plant_pred_L + plant_pred_R) (a copy of
    fcp.step + per-eye plant), eye_vel_pred = d/dt x_p_pred (head frame), rotated
    to eye frame through x_p_pred and saturated with `v_offset = −w_est_eye` —
    exactly as `cerebellum.step` does it.  Because the forward model carries the
    FCP nonlinearities, x_p_pred ≈ the actual eye position and eye_vel_pred ≈
    the actual eye velocity — EC and retinal slip stay frame-matched mid-saccade.
    """
    sp = params.sensory
    bp = params.brain

    eye3d    = (np.array(states.plant.left) + np.array(states.plant.right)) / 2.0
    vel3d    = np.gradient(eye3d, dt, axis=0)                 # (T, 3) deg/s world
    # Cerebellum has no internal eye forward model now — it rotates through
    # NI_net and uses u_ni_in as the eye velocity.  Reconstruct the same way.
    x_p_pred = np.array(ni_net(states))                       # (T, 3) NI_net
    evp3d    = np.gradient(x_p_pred, dt, axis=0)              # (T, 3) ≈ u_ni_in
    w_est    = np.array(vs_net(states))                       # (T, 3) head frame

    def _at(vel_world, eye_pos_actual, xpp, evp, w):
        # Slip side: actual plant position + velocity (what the retina sees).
        R_act = rotation_matrix(ypr_to_xyz(eye_pos_actual))
        slip_eye        = -xyz_to_ypr(R_act.T @ ypr_to_xyz(vel_world))
        slip_target_pre = velocity_saturation(slip_eye, sp.v_max_target_vel)
        slip_scene_pre  = velocity_saturation(slip_eye, sp.v_max_scene_vel)

        # EC side: predicted eye velocity rotated through the predicted plant
        # position x_p_pred — same transform the retina applies to the actual
        # eye velocity through the actual eye position.
        R_pred       = rotation_matrix(ypr_to_xyz(xpp))
        Rt           = R_pred.T
        ev_pred_eye  = xyz_to_ypr(Rt @ ypr_to_xyz(evp))
        w_est_eye    = xyz_to_ypr(Rt @ ypr_to_xyz(w))
        v_offset     = -w_est_eye

        ec_target_pre = velocity_saturation(ev_pred_eye, bp.v_max_pursuit, v_offset=v_offset)
        ec_scene_pre  = velocity_saturation(ev_pred_eye, bp.v_max_okr,     v_offset=v_offset)

        # Raw (pre-cascade) saturation flag = 1 − cos_gain(|v_rel|), same
        # as cerebellum.step computes before the gate cascade.
        v_rel  = ev_pred_eye - v_offset
        speed  = jnp.linalg.norm(v_rel)
        def _sat_flag(spd, v_sat):
            v_zero = 2.0 * v_sat
            t      = jnp.clip((spd - v_sat) / (v_zero - v_sat), 0.0, 1.0)
            return 1.0 - 0.5 * (1.0 + jnp.cos(jnp.pi * t))
        sat_t = _sat_flag(speed, bp.v_max_pursuit)
        sat_s = _sat_flag(speed, bp.v_max_okr)

        return (slip_target_pre, slip_scene_pre, ec_target_pre, ec_scene_pre,
                sat_t, sat_s)

    tg, sc, ect, ecs, sat_t, sat_s = jax.vmap(_at)(
        jnp.array(vel3d), jnp.array(eye3d), jnp.array(x_p_pred),
        jnp.array(evp3d), jnp.array(w_est),
    )
    return (np.array(tg), np.array(sc), np.array(ect), np.array(ecs),
            np.array(sat_t), np.array(sat_s))

DT    = 0.001
THETA = with_brain(PARAMS_DEFAULT, g_burst=700.0)
THETA_NOISELESS = with_brain(with_sensory(THETA, sigma_canal=0.0, sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)
SHOW  = '--show' in sys.argv


def _primary_saccade(burst_yaw, eye_yaw, t_np, t_jump, threshold=20.0):
    """Amplitude, peak velocity and duration of the FIRST saccade after t_jump.

    Uses the burst signal (not raw velocity) to gate on/off so corrective
    saccades and post-saccadic drift are excluded.

    Returns (amplitude_deg, peak_vel_deg_s, duration_s).  Falls back to
    full-trace extremes (with NaN duration) only if no burst crossing is found.
    """
    i0     = int(t_jump / DT) + 1          # first sample after step
    burst  = burst_yaw[i0:]
    eye    = eye_yaw[i0:]
    vel    = np.gradient(eye, DT)

    is_sac = np.abs(burst) > threshold
    on_idx  = np.where(np.diff(is_sac.astype(int)) > 0)[0]
    off_idx = np.where(np.diff(is_sac.astype(int)) < 0)[0]

    if len(on_idx) == 0:
        return float(eye[-1] - eye_yaw[i0 - 1]), float(np.max(np.abs(vel))), float('nan')

    i_on  = on_idx[0] + 1                  # first sample inside saccade
    after = off_idx[off_idx >= on_idx[0]]
    i_off = int(after[0]) + 1 if len(after) > 0 else len(burst) - 1

    amplitude = float(eye[i_off] - eye[i_on - 1])
    peak_vel  = float(np.max(np.abs(vel[i_on:i_off + 1])))
    duration  = float((i_off - (i_on - 1)) * DT)        # onset → offset span
    return amplitude, peak_vel, duration


def _count_saccades(burst_yaw, t_start, threshold=20.0, min_dur_s=0.004):
    """Count distinct saccade bursts after `t_start`.

    A saccade = a contiguous run where |burst| exceeds `threshold` lasting at
    least `min_dur_s` (filters single-sample blips / microsaccade noise).
    """
    i0     = int(t_start / DT)
    is_sac = (np.abs(burst_yaw[i0:]) > threshold).astype(int)
    edges  = np.diff(np.concatenate([[0], is_sac, [0]]))
    starts = np.where(edges > 0)[0]
    ends   = np.where(edges < 0)[0]
    min_n  = max(1, int(min_dur_s / DT))
    return int(np.sum((ends - starts) >= min_n))


def _saccade_onset_times(burst_yaw, t_start, threshold=20.0, min_dur_s=0.004,
                         merge_gap_s=0.10):
    """Onset times (s, absolute) of distinct saccades after `t_start`.

    Burst runs (|burst| > threshold, ≥ min_dur_s) give candidate onsets, BUT a
    single saccade's burst can momentarily dip below threshold and split into
    two sub-runs ~50 ms apart. Onsets within `merge_gap_s` of the previously
    kept onset are therefore merged into one saccade — nothing physiological
    re-fires that fast (well under the refractory floor), so this de-glitches the
    split without merging genuine successive saccades (which are ≫ merge_gap_s
    apart). Returns the onset of each merged saccade so REALIZED inter-saccadic
    intervals (np.diff of onsets) reflect saccades, not burst sub-peaks.
    """
    i0     = int(t_start / DT)
    is_sac = (np.abs(burst_yaw[i0:]) > threshold).astype(int)
    edges  = np.diff(np.concatenate([[0], is_sac, [0]]))
    starts = np.where(edges > 0)[0]
    ends   = np.where(edges < 0)[0]
    min_n  = max(1, int(min_dur_s / DT))
    onsets = (i0 + starts[(ends - starts) >= min_n]) * DT
    merged = []
    for o in onsets:
        if not merged or (o - merged[-1]) >= merge_gap_s:
            merged.append(float(o))
    return np.asarray(merged)


# ── helpers ───────────────────────────────────────────────────────────────────

def _pt3(t_np, amp_h_deg, amp_v_deg=0.0, t_jump=0.1):
    T = len(t_np)
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.where(t_np >= t_jump, np.tan(np.radians(amp_h_deg)), 0.0)
    pt3[:, 1] = np.where(t_np >= t_jump, np.tan(np.radians(amp_v_deg)), 0.0)
    return jnp.array(pt3)


def _run(t_np, pt3, key=0, max_s=None, params=None):
    t  = jnp.array(t_np)
    T  = len(t)
    ms = max_s or int((t_np[-1] - t_np[0]) / DT) + 300
    return simulate(params or THETA, t,
                    target=km.build_target(t_np, lin_pos=np.array(pt3)),
                    scene_present_array=jnp.ones(T),
                    max_steps=ms, return_states=True,
                    key=jax.random.PRNGKey(key))


# ── Figure 1: main sequence ───────────────────────────────────────────────────

def _main_sequence(show):
    amplitudes = [0.5, 1, 2, 3, 5, 8, 10, 15, 20]
    T_end, t_jump = 0.8, 0.1
    t_np = np.arange(0.0, T_end, DT)

    amps_out, peak_vels, durations = [], [], []
    traces = {}
    for i, amp in enumerate(amplitudes):
        pt3 = _pt3(t_np, amp, t_jump=t_jump)
        st  = _run(t_np, pt3, key=i)
        # version = average of left and right eye (conjugate movement hits target)
        eye   = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
        burst = extract_burst(st, THETA)[:, 0]
        a_out, v_peak, dur = _primary_saccade(burst, eye, t_np, t_jump)
        amps_out.append(a_out)
        peak_vels.append(v_peak)
        durations.append(dur)
        # store both the position trace and its velocity (aligned to the step)
        traces[amp] = (t_np - t_jump, eye - eye[int(t_jump / DT)], np.gradient(eye, DT))

    A_ref = np.linspace(0, 22, 300)
    v_ref = 700.0 * (1.0 - np.exp(-A_ref / 7.0))

    # Duration main-sequence reference: human linear law D(ms) ≈ 2.2·A + 21.
    dur_ms  = np.array(durations) * 1000.0
    A_dref  = np.linspace(0, 22, 300)
    dur_ref = 2.2 * A_dref + 21.0

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle('Saccade Main Sequence', fontsize=12)
    cmap = plt.get_cmap('plasma')

    # [0,0] peak-velocity main sequence
    ax = axes[0, 0]
    ax.plot(A_ref, v_ref, color=utils.C['dark'], lw=1.5, ls='--',
            label='700(1−e^{−A/7})')
    ax.scatter(amps_out, peak_vels, color=utils.C['eye'], s=70, zorder=5)
    for a, v in zip(amps_out, peak_vels):
        ax.annotate(f'{a:.0f}°', (a, v), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Amplitude (deg)'); ax.set_ylabel('Peak velocity (deg/s)')
    ax.set_title('Peak-velocity main sequence')
    ax.legend(fontsize=9); ax.set_xlim(0, 22); ax.set_ylim(0)
    ax.grid(True, alpha=0.25)

    # [0,1] duration main sequence
    ax = axes[0, 1]
    ax.plot(A_dref, dur_ref, color=utils.C['dark'], lw=1.5, ls='--',
            label='2.2·A + 21 ms (human)')
    ax.scatter(amps_out, dur_ms, color=utils.C['eye'], s=70, zorder=5)
    for a, d in zip(amps_out, dur_ms):
        ax.annotate(f'{a:.0f}°', (a, d), fontsize=7,
                    xytext=(3, 3), textcoords='offset points')
    ax.set_xlabel('Amplitude (deg)'); ax.set_ylabel('Duration (ms)')
    ax.set_title('Duration main sequence')
    ax.legend(fontsize=9); ax.set_xlim(0, 22); ax.set_ylim(0)
    ax.grid(True, alpha=0.25)

    # [1,0] aligned eye-position traces
    ax = axes[1, 0]
    for i, amp in enumerate(amplitudes):
        t_al, eye_al, _ = traces[amp]
        ax.plot(t_al, eye_al, color=cmap(i / (len(amplitudes) - 1)), lw=1.3,
                label=f'{amp:.0f}°' if amp in [1, 5, 10, 20] else None)
    ax.set_xlabel('Time from step (s)'); ax.set_ylabel('Eye position (deg)')
    ax.set_title('Position traces (aligned to target step)')
    ax.set_xlim(-0.05, 0.55); ax.axvline(0, color='gray', lw=0.7, ls='--')
    ax.axhline(0, color='k', lw=0.4); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    # [1,1] aligned eye-velocity traces (same alignment as the position panel)
    ax = axes[1, 1]
    for i, amp in enumerate(amplitudes):
        t_al, _, vel_al = traces[amp]
        ax.plot(t_al, vel_al, color=cmap(i / (len(amplitudes) - 1)), lw=1.3,
                label=f'{amp:.0f}°' if amp in [1, 5, 10, 20] else None)
    ax.set_xlabel('Time from step (s)'); ax.set_ylabel('Eye velocity (deg/s)')
    ax.set_title('Velocity traces (aligned to target step)')
    ax.set_xlim(-0.05, 0.55); ax.axvline(0, color='gray', lw=0.7, ls='--')
    ax.axhline(0, color='k', lw=0.4); ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_main_sequence', show=show, params=THETA,
                              conditions='Lit, foveal target stepping 1°–40° horizontally')

    # ── Quantitative metrics (measured from states, not the figure) ──────────
    amps_arr = np.array(amps_out)
    peak_arr = np.array(peak_vels)
    req      = np.array(amplitudes, dtype=float)
    big      = req >= 5.0                      # main sequence is clean for A ≥ 5°
    ref_big  = 700.0 * (1.0 - np.exp(-amps_arr[big] / 7.0))
    resid_max = float(np.max(np.abs(peak_arr[big] - ref_big) / ref_big))
    peak_20   = float(peak_arr[amplitudes.index(20)])
    gain      = float(np.mean(amps_arr[big] / req[big]))

    metrics = [
        Metric('sac_peak_vel_20deg', peak_20, 
               lo=550.0, hi=750.0, golden_tol=0.05, units='deg/s',
               cite='Bahill et al. (1975)',
               desc='Peak velocity of a 20° saccade (main-sequence saturation)'),
        Metric('sac_mainseq_resid_max', resid_max,
               lo=None, hi=0.25, golden_tol=0.15, units='',
               cite='Bahill et al. (1975)',
               desc='Max fractional deviation from 700(1−e^−A/7), amplitudes ≥5° '
                    '(the idealized curve over-predicts small saccades, so ~20% at 5° is expected)'),
        Metric('sac_primary_gain', gain, 
               lo=0.80, hi=1.05, golden_tol=0.05, units='',
               cite='Robinson (1975)',
               desc='Mean primary-saccade amplitude / target amplitude (≥5°)'),
    ]

    fig = utils.fig_meta(path, rp,
        title='Saccade Main Sequence',
        description='Top: peak-velocity and duration main sequences (scatter + reference curves) vs amplitude. '
                    'Bottom: eye position and velocity traces aligned to the target step, for amplitudes 0.5–20°.',
        expected='Peak velocity within ±20% of 700(1−e^{−A/7}) (≈660 deg/s at 20°); duration grows roughly '
                 'linearly with amplitude; velocity profiles are single-peaked and scale with amplitude.',
        citation='Bahill et al. (1975) Science; Robinson (1975) J Neurophysiol',
        fig_type='behavior')
    fig['metrics'] = metrics
    return fig


# ── Figure 2: oblique saccades ────────────────────────────────────────────────

def _oblique(show):
    """Saccade fan: 12° saccades from centre to 8 equally-spaced directions, each
    followed by a return to centre. Tests the hallmarks of oblique saccades —
    straight 2-D trajectories and synchronised horizontal/vertical components.
    Noiseless, so curvature reflects the model rather than fixational jitter.
    """
    AMP  = 12.0
    DIRS = np.arange(0.0, 360.0, 45.0)      # 8 directions, 45° apart
    HOLD = 0.35                              # dwell at each target / at centre
    T0   = 0.2

    targets = []
    for d in DIRS:
        targets.append((AMP * np.cos(np.radians(d)), AMP * np.sin(np.radians(d))))
        targets.append((0.0, 0.0))           # return to centre between directions
    T_end = T0 + len(targets) * HOLD + 0.2
    t_np  = np.arange(0.0, T_end, DT)
    T     = len(t_np)

    tgt_h = np.zeros(T); tgt_v = np.zeros(T)
    out_events = []   # (onset_idx, dir_deg) for each outward saccade
    for i, (h, v) in enumerate(targets):
        ts = T0 + i * HOLD
        tgt_h[t_np >= ts] = h; tgt_v[t_np >= ts] = v
        if i % 2 == 0:
            out_events.append((int(ts / DT), float(DIRS[i // 2])))
    pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.tan(np.radians(tgt_h)); pt3[:, 1] = np.tan(np.radians(tgt_v))

    st  = _run(t_np, jnp.array(pt3), max_s=int(T_end / DT) + 500, params=THETA_NOISELESS)
    eye = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
    eye_h, eye_v = eye[:, 0], eye[:, 1]

    # ── Per-direction straightness (robust to timing lag + endpoint-free) ─────
    # Two robustness problems, both solved here:
    #  (1) Timing lag. At 350 ms dwell the ~220 ms saccade latency leaves little
    #      margin, so lag accumulates and by the later directions the eye is a
    #      full step behind — a window fixed at the target-onset index catches the
    #      *previous* return, not the outward saccade. Fix: search a generous
    #      2·HOLD window and pick the contiguous high-speed segment whose net
    #      displacement points at the commanded direction (the outward saccade),
    #      wherever the lag put it. The return tail / next return point the other
    #      way and are rejected by the sign of the projection.
    #  (2) Endpoint detection. Once the segment is isolated, curvature = max
    #      perpendicular spread about the segment's own best-fit line (PCA) ÷ the
    #      along-axis extent. No speed-threshold "endpoint" to be fooled by a
    #      mid-saccade velocity notch (the old 0.84 / 1.29 fan artifacts).
    search_n = int(2 * HOLD / DT)
    THR_SP   = 20.0          # deg/s — 2-D speed threshold for saccade samples
    merge_n  = int(0.03 / DT)   # bridge <30 ms gaps (mid-saccade velocity notch)
    straight = {}
    for i0, d in out_events:
        h = eye_h[i0:i0 + search_n] - eye_h[i0]
        v = eye_v[i0:i0 + search_n] - eye_v[i0]
        a = (np.hypot(np.gradient(h, DT), np.gradient(v, DT)) > THR_SP).astype(np.int8)
        if not a.any():
            continue
        edges  = np.diff(np.concatenate([[0], a, [0]]))
        starts = list(np.where(edges == 1)[0]); ends = list(np.where(edges == -1)[0])
        runs = []                                            # merge sub-30 ms gaps
        for s, e in zip(starts, ends):
            if runs and s - runs[-1][1] <= merge_n:
                runs[-1] = (runs[-1][0], e)
            else:
                runs.append((s, e))
        ux, uy = np.cos(np.radians(d)), np.sin(np.radians(d))
        best, best_proj = None, 5.0                          # ≥5° toward target = the outward saccade
        for s, e in runs:
            proj = (h[e - 1] - h[s]) * ux + (v[e - 1] - v[s]) * uy
            if proj > best_proj:
                best_proj, best = proj, (s, e)
        if best is None:
            continue
        s, e = best
        c    = np.column_stack([h[s:e], v[s:e]]); c = c - c.mean(axis=0)
        axis = np.linalg.svd(c, full_matrices=False)[2][0]   # principal (saccade) direction
        extent = float(np.ptp(c @ axis))
        if extent < 1.0:
            continue
        perp = c @ np.array([-axis[1], axis[0]])
        straight[d] = float(np.max(np.abs(perp)) / extent)   # max perp spread ÷ along-axis extent

    obliq    = [d for d in DIRS if d % 90 != 0]     # 45,135,225,315 (true obliques)
    str_obl  = [straight[d] for d in obliq if d in straight]
    m_straight = float(np.max(str_obl))  if str_obl  else float('nan')

    # ── Plot: 2-D fan trajectories | per-direction straightness bars ──────────
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle('Oblique Saccades — 12° fan, 8 directions (noiseless)', fontsize=12)

    ax = axes[0]
    ax.plot(eye_h, eye_v, color=utils.C['eye'], lw=1.0, alpha=0.9)
    for d in DIRS:
        ax.plot(AMP * np.cos(np.radians(d)), AMP * np.sin(np.radians(d)), 'x',
                color=utils.C['target'], ms=9, markeredgewidth=2.0)
    ax.set_aspect('equal'); ax.grid(True, alpha=0.25)
    ax.axhline(0, color='k', lw=0.4); ax.axvline(0, color='k', lw=0.4)
    ax.set_xlabel('Yaw (deg)'); ax.set_ylabel('Pitch (deg)')
    ax.set_title('2-D trajectories (straight radial lines expected)')

    ax = axes[1]
    ds = sorted(straight.keys())
    ax.bar([str(int(d)) for d in ds], [straight[d] * 100 for d in ds],
           color=['#d62728' if d % 90 != 0 else '#888888' for d in ds])
    ax.set_ylabel('Max curvature (% of amplitude)'); ax.set_xlabel('Direction (deg)')
    ax.set_title('Trajectory straightness (red = oblique)')
    ax.grid(True, alpha=0.25, axis='y')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_oblique', show=show, params=THETA_NOISELESS,
                              conditions='Lit, NOISELESS — 12° saccades to 8 directions (fan); curvature = model, not jitter')
    metrics = [
        Metric('sac_oblique_straightness', m_straight,
               lo=None, hi=0.08, golden_tol=0.2, units='',
               cite='Smit et al. (1990); van Gisbergen et al. (1985)',
               desc='Max trajectory curvature (perp dev ÷ amplitude) over oblique directions'),
    ]
    fig_meta = utils.fig_meta(path, rp,
        title='Oblique Saccades (8-direction fan)',
        description='12° saccades from centre to 8 directions (45° apart), noiseless. '
                    'Left: 2-D trajectories. Right: per-direction trajectory curvature.',
        expected='Straight radial trajectories (curvature < ~5% of amplitude). Curvature '
                 'is the readout of H/V component desynchrony, so it suffices on its own.',
        citation='Smit et al. (1990) J Neurophysiol; van Gisbergen et al. (1985)',
        fig_type='behavior')
    fig_meta['metrics'] = metrics
    return fig_meta


# ── Figure 3: double-step refractoriness ──────────────────────────────────────

def _refractoriness(show):
    """Double-step paradigm: 4 amplitude pairs × 6 ISIs.

    Each column: first step to A/2, second step to A (A = 10, 20, 30, 40°).
    Rows: eye + target position (target dashed, same colour), burst signal.
    ISIs span 50–500 ms to show refractoriness → full two-saccade range.
    """
    AMPS = [10, 20, 30, 40]    # full target amplitude (first step = A/2)
    isis = [0.02, 0.05, 0.10, 0.15, 0.20, 0.35, 0.50]
    T_end = 1.2
    t1    = 0.15

    t_np = np.arange(0.0, T_end, DT)
    T    = len(t_np)

    cmap   = plt.get_cmap('viridis')
    colors = [cmap(i / (len(isis) - 1)) for i in range(len(isis))]

    sac_onsets = {}   # (A, isi) -> onset times (s) of the detected saccades

    fig, axes = plt.subplots(2, len(AMPS), figsize=(4.5 * len(AMPS), 8),
                             sharex=True)
    fig.suptitle(
        'Saccade Double-Step Refractoriness\n'
        'First step: 0→A/2  |  Second step: A/2→A  at varying ISI  '
        '(dashed = target, solid = eye version)',
        fontsize=11)

    for ci, A in enumerate(AMPS):
        A1 = A / 2.0
        A2 = float(A)

        for ri, isi in enumerate(isis):
            t2  = t1 + isi
            tgt = np.where(t_np < t1, 0.0,
                  np.where(t_np < t2, A1, A2)).astype(np.float32)
            pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
            pt3[:, 0] = np.tan(np.radians(tgt))

            st  = _run(t_np, jnp.array(pt3), key=ri * len(AMPS) + ci)
            eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
            bst = extract_burst(st, THETA)[:, 0]
            sac_onsets[(A, isi)] = _saccade_onset_times(bst, t1)

            lbl = f'ISI={isi*1000:.0f}ms'
            col = colors[ri]

            # Row 0: eye (solid) + target (dashed, same colour, thinner)
            axes[0, ci].plot(t_np, eye, color=col, lw=1.8)
            axes[0, ci].plot(t_np, tgt, color=col, lw=0.9, ls='--', alpha=0.55,
                             label=lbl if ci == 0 else None)
            # Row 1: burst
            axes[1, ci].plot(t_np, bst, color=col, lw=1.4, alpha=0.85)

        axes[0, ci].set_title(f'0→{A1:.0f}°→{A2:.0f}°', fontsize=10)
        axes[0, ci].axhline(A1, color='gray', lw=0.5, ls=':', alpha=0.3)
        axes[0, ci].axhline(A2, color='gray', lw=0.5, ls=':', alpha=0.3)
        axes[0, ci].axhline(0.0, color='k', lw=0.4)
        axes[0, ci].axvline(t1, color='gray', lw=0.8, ls='--', alpha=0.5)
        axes[0, ci].grid(True, alpha=0.2)

        axes[1, ci].axhline(0.0, color='k', lw=0.4)
        axes[1, ci].axvline(t1, color='gray', lw=0.8, ls='--', alpha=0.5)
        axes[1, ci].set_xlabel('Time (s)', fontsize=8)
        axes[1, ci].grid(True, alpha=0.2)

    axes[0, 0].set_ylabel('Eye / target position (deg)', fontsize=8)
    axes[0, 0].legend(fontsize=8, loc='upper left')
    axes[1, 0].set_ylabel('Burst command (deg/s)', fontsize=8)

    axes[0, 0].set_title(
        f'0→{AMPS[0]//2}°→{AMPS[0]}°\n(Eye solid · target dashed)', fontsize=9)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'saccade_refractoriness', show=show, params=THETA,
                              conditions='Lit, double-step target with varying inter-step intervals')

    # ── Quantitative metrics ─────────────────────────────────────────────────
    # REALIZED inter-saccadic interval = the onset→onset gap between successive
    # detected saccades in each double-step trial. This is the MEASURED interval,
    # NOT the commanded step ISI: for short commands the second saccade is held
    # off by the refractory period, so the realized interval saturates near the
    # refractory floor regardless of how close the two steps were commanded.
    # Floor (smallest realized interval) + median across all two-saccade trials.
    isi_realized = []
    for A in AMPS:
        for isi in isis:
            ons = sac_onsets.get((A, isi))
            if ons is not None and len(ons) >= 2:
                isi_realized.extend(np.diff(ons))          # s, onset→onset gaps
    isi_realized = np.asarray(isi_realized) * 1000.0       # ms
    refr_isi_min_ms    = float(np.min(isi_realized))    if isi_realized.size else float('nan')
    refr_isi_median_ms = float(np.median(isi_realized)) if isi_realized.size else float('nan')

    metrics = [
        Metric('sac_refractory_isi_ms', refr_isi_min_ms,
               lo=100.0, hi=350.0, golden_tol=0.25, units='ms',
               cite='Becker & Jürgens (1979)',
               desc='Smallest REALIZED inter-saccadic interval (onset→onset) over all double-step trials — refractory floor'),
        Metric('sac_refractory_isi_median_ms', refr_isi_median_ms,
               lo=150.0, hi=500.0, golden_tol=0.25, units='ms',
               cite='Becker & Jürgens (1979)',
               desc='Median REALIZED inter-saccadic interval (onset→onset) across double-step trials'),
    ]

    fig = utils.fig_meta(path, rp,
        title='Saccade Double-Step Refractoriness',
        description='Double-step paradigm: target jumps 0→A/2 then A/2→A for A=10,20,30,40° '
                    'after variable ISI (20–500 ms). '
                    'Target shown dashed in same colour as eye trace. '
                    'Short ISIs produce one amended saccade; longer ISIs produce two.',
        expected='ISI < ~100 ms: 1 saccade (merged). ISI > ~150 ms: 2 separate saccades. '
                 'Refractory period roughly constant across amplitudes (~150 ms).',
        citation='Becker & Jürgens (1979) Vision Res',
        fig_type='behavior')
    fig['metrics'] = metrics
    return fig


# ── Figure 4: signal cascade ──────────────────────────────────────────────────

def _cascade(show):
    # One figure: the noiseless cascade across 1°/5°/20°/40°, plus a single 1°
    # column WITH noise (last). The noise condition only needs the small saccade
    # — that's where fixational drift + microsaccadic RT variability show; at
    # large amplitudes the burst dominates and noisy ≈ noiseless.
    columns = [(1.0, False), (5.0, False), (20.0, False), (40.0, False), (1.0, True)]
    fname   = 'saccade_cascade'
    T_end, t_jump = 0.9, 0.1
    t_np = np.arange(0.0, T_end, DT)
    T    = len(t_np)

    n_rows, n_cols = 9, len(columns)
    # Row 5 (pre-cascade) is zoomed in on the time axis, so x-axes are not
    # shared across rows.  All other rows display the full 0-T_end window.
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.6 * n_cols, 2.2 * n_rows), sharex=False)
    fig.suptitle('Saccade Signal Cascade  ·  '
                 'cascade → accumulate → latch/freeze → burst → copy → refractory',
                 fontsize=11)

    row_labels = ['Eye + target pos (deg)', 'Cascade output + hold (deg)',
                  'Accum / latch + refractory',
                  'Burst (deg/s) + eye velocity',
                  'Saccadic-suppression gates',
                  'Pre-cascade slip + EC (deg/s)',
                  'Post-cascade slip + EC (deg/s)',
                  'Pursuit / VS drives (deg/s)',
                  'Vert + tors eye vel (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)

    # Post-saccade window: well after even the 40° saccade has completed
    # (saccades finish by ~0.25 s for t_jump=0.1). Accuracy = endpoint error,
    # stability = residual eye speed (post-saccadic drift / oscillation).
    # We accumulate per amplitude both "with corrective saccades" and
    # "without" (correctives masked via the OPN fast-phase latch).
    POST_T0, POST_T1 = 0.5, 0.85          # LONG hold window (sustained drift/oscillation)
    SHORT_T1 = 0.45                        # SHORT window end: ~130 ms after even the 40°
                                           # saccade finishes — the immediate ring/glissade
    post_win = (t_np >= POST_T0) & (t_np <= POST_T1)
    # Post-saccadic STABILITY collected per column. Accuracy now lives ONLY in the
    # main-sequence figure (it spans more amplitudes); the cascade only answers
    # "is the eye stable after the saccade?". Only the noiseless peaks become
    # metrics — the noisy 1° column is noise-floor-dominated, not an EC signal.
    spd_peak_s  = {False: [], True: []}   # PEAK speed, SHORT window (immediate ring/glissade)
    spd_peak_l  = {False: [], True: []}   # PEAK speed, LONG window (sustained oscillation/drift)
    spd_verg    = {False: [], True: []}   # PEAK vergence (L−R) speed, SHORT window (per-eye/MLF)
    sac_count   = []                      # # distinct saccades per NOISELESS column (should be 1–2)

    for ci, (amp, noisy) in enumerate(columns):
        params = THETA if noisy else THETA_NOISELESS
        pt3 = _pt3(t_np, amp, t_jump=t_jump)
        st  = _run(t_np, pt3, key=ci, max_s=int(T_end/DT)+200, params=params)
        sg  = extract_sg(st, params)
        eye3d = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0    # (T, 3)
        vel3d = np.gradient(eye3d, DT, axis=0)                                 # (T, 3)
        eye   = eye3d[:, 0]
        vel   = vel3d[:, 0]
        tgt = np.degrees(np.arctan2(np.array(pt3[:,0]), np.array(pt3[:,2])))

        # ── Post-saccadic stability window ───────────────────────────────────
        # slow = post-saccade samples that are NOT inside a (corrective) fast
        # phase, detected via the OPN latch (z_opn < 50 ⇒ saccade in progress).
        z_opn = extract_z_opn(st)
        slow  = post_win & (z_opn >= 50.0)
        # Regression guard (noiseless only): a single target step should produce
        # ONE saccade, or at most TWO (primary + one corrective). More ⇒ the burst
        # is oscillating / mis-terminating, even if the eye happens to land right.
        if not noisy:
            sac_count.append(len(_saccade_onset_times(sg['u_burst'][:, 0], t_jump)))
        # PEAK residual slow-phase speed in TWO windows — they reflect different
        # failure modes (fast phases masked via the OPN latch ⇒ pure smooth
        # residual; noiseless both should be ~0):
        #   SHORT [saccade end, +~130 ms] → immediate ring/glissade = EC
        #     forward-model transient + pulse-step mismatch.
        #   LONG  [0.5, 0.85] hold        → sustained oscillation / steady drift =
        #     EC steady-state leak + gaze-holding.
        short_win = (t_np >= t_jump + 0.05) & (t_np <= SHORT_T1) & (z_opn >= 50.0)
        spd_peak_s[noisy].append(float(np.max(np.abs(vel[short_win]))) if short_win.any() else float('nan'))
        spd_peak_l[noisy].append(float(np.max(np.abs(vel[slow])))      if slow.any()      else float('nan'))
        # Side-2: the per-eye (MLF) glissade asymmetry the version metric hides —
        # peak post-saccadic vergence (L−R) velocity in the short window.
        verg_vel = np.gradient(np.array(st.plant.left[:, 0]) - np.array(st.plant.right[:, 0]), DT)
        spd_verg[noisy].append(float(np.max(np.abs(verg_vel[short_win]))) if short_win.any() else float('nan'))

        col_title = f'{amp:.0f}°' + ('\n(with noise)' if noisy else '')
        axes[0, ci].set_title(col_title, fontsize=10,
                              color=('#b2182b' if noisy else 'black'))
        vl  = dict(color='gray', lw=0.6, ls='--', alpha=0.5)
        for ax in axes[:, ci]:
            ax.axvline(t_jump, **vl); ax.grid(True, alpha=0.15)
            if noisy: ax.set_facecolor('#fbf7ef')   # tint the noise column

        axes[0, ci].plot(t_np, tgt, color=utils.C['target'], lw=1.5, label='target')
        axes[0, ci].plot(t_np, eye, color=utils.C['eye'],    lw=1.5, label='eye')
        ax_fmt(axes[0, ci])
        if ci == 0: axes[0, ci].legend(fontsize=7)

        axes[1, ci].plot(t_np, sg['e_pd'][:,0],   color='darkorange', lw=1.0, ls='--', label='e_delayed')
        axes[1, ci].plot(t_np, sg['e_held'][:,0], color=utils.C['vs'], lw=1.8, label='e_held (frozen)')
        ax_fmt(axes[1, ci])
        if ci == 0: axes[1, ci].legend(fontsize=7)

        # Row 2: accumulator / trigger / latch + refractory (all 0–1 scale, same axis)
        axes[2, ci].plot(t_np, sg['z_acc'],         color='#e08214', lw=1.5, label='z_acc')
        axes[2, ci].plot(t_np, sg['z_trig'],         color='#c51b8a', lw=1.5, label='z_trig (trigger IBN)')
        axes[2, ci].plot(t_np, sg['z_opn'] / 100,   color='#1b7837', lw=1.5, label='OPN (norm)')
        axes[2, ci].axhline(params.brain.threshold_acc, color='#e08214', lw=0.8, ls=':')
        axes[2, ci].set_ylim(-0.05, 1.15)
        if ci == 0: axes[2, ci].legend(fontsize=7)

        # Row 3: eye velocity (yaw) + burst (yaw)
        axes[3, ci].plot(t_np, vel,                color=utils.C['eye'],   lw=1.4, label='eye vel')
        axes[3, ci].plot(t_np, sg['u_burst'][:,0], color=utils.C['burst'], lw=1.5, label='burst', zorder=3)
        ax_fmt(axes[3, ci])
        ylo, yhi = axes[3, ci].get_ylim()
        axes[3, ci].set_ylim(min(ylo, -1.0), max(yhi, 1.0))
        if ci == 0: axes[3, ci].legend(fontsize=7)

        # ── Pursuit / OKR signal chain (rows 4–7) ────────────────────────────
        # Pull canonical signals from the brain registries (vmapped over time)
        # instead of reaching into raw state slices.
        acts     = read_brain_acts(st, params)             # Activations(T, …)
        decoded  = read_brain_decoded(st, params)          # Decoded(T, …)
        vel_del  = np.array(acts.pc.target_vel)            # (T, 3) delayed target vel
        slip_del = np.array(acts.pc.scene_angular_vel)     # (T, 3) delayed scene slip

        x_purs   = np.array(decoded.pu.net)                # (T, 3) NET pursuit memory

        # Pre-cascade signals (used in row 4 + row 5).  Also returns the raw
        # (instant) saturation flags so we can plot the pre-cascade gate
        # alongside the post-cascade strengthened gate in row 4.
        (tgt_pre, scn_pre, ect_pre, ecs_pre,
         raw_sat_tgt, raw_sat_scn) = _pre_cascade_signals(st, params, DT)

        # Row 4: saccadic-suppression gates.  Solid lines = POST-cascade
        # strengthened gates (acts.cb.saccadic_suppression_*, what actually multiplies
        # the PE downstream).  Dotted lines = PRE-cascade raw gates
        # (= 1 − instant saturation flag, no delay).  The cascade-induced
        # delay between the two is the same as the slip/EC cascade delay,
        # so the post-cascade gate is delay-aligned with PE.
        gate_target_post = np.array(acts.cb.saccadic_suppression_target)
        gate_scene_post  = np.array(acts.cb.saccadic_suppression_scene)
        gate_target_pre  = 1.0 - raw_sat_tgt
        gate_scene_pre   = 1.0 - raw_sat_scn
        axes[4, ci].plot(t_np, gate_target_pre,  color='#7b2d8b', lw=1.0, ls=':',  label='target gate (pre)')
        axes[4, ci].plot(t_np, gate_target_post, color='#7b2d8b', lw=1.5, label='target gate (post)')
        axes[4, ci].plot(t_np, gate_scene_pre,   color='#1a7a4a', lw=1.0, ls=':',  label='scene gate (pre)')
        axes[4, ci].plot(t_np, gate_scene_post,  color='#1a7a4a', lw=1.5, label='scene gate (post)')
        axes[4, ci].axhline(1.0, color='gray', lw=0.6, ls=':')
        axes[4, ci].set_ylim(-0.05, 1.10)
        ax_fmt(axes[4, ci])
        if ci == 0: axes[4, ci].legend(fontsize=7)

        # Effective EC = exactly what the brain adds to slip in pred_err:
        #   scene path:   PE = scene_slip  + scene_visible  · ec_d_scene
        #   target path:  PE = target_slip + K_cereb_pu · target_visible · ec_no_torsion
        # RAW EC cascade outputs — what the cerebellum's forward model emits,
        # independent of the downstream K_cereb / saccadic-suppression / visibility
        # gains.  This is the right thing to compare against the slip cascade:
        # if the EC is a good forward model, −EC ≈ slip (the eye-motion
        # contribution to retinal slip).  The torsion axis is zeroed on the
        # target path (matching pred_err's ec_no_torsion).
        ec_scene   = np.array(acts.cb.ec_scene)                      # (T, 3)
        ec_target  = np.array(acts.cb.ec_target)                     # (T, 3)
        ec_target_no_torsion = ec_target.copy()
        ec_target_no_torsion[:, 2] = 0.0

        # Row 5: PRE-cascade slip + effective EC (after clipping, before the
        # 6-stage gamma + LP).  Saturated retinal slip in eye frame for
        # stationary target/scene/head (= −R_eye.T·eye_velocity), saturated EC
        # = saturated mn_lp_eye with v_offset = −w_est_eye (same as
        # cerebellum.step).  EC sign-flipped to overlay slip when cancellation
        # is perfect at the input of the cascades.  Pre-cascade signals were
        # already computed above (for the gate row).
        axes[5, ci].plot(t_np, tgt_pre[:, 0], color='#7b2d8b', lw=1.2, label='tgt slip (pre)')
        axes[5, ci].plot(t_np, -ect_pre[:, 0],color='#d62728', lw=1.0, ls=':',  label='−EC target (pre)')
        axes[5, ci].plot(t_np, scn_pre[:, 0], color='#1a7a4a', lw=1.2, label='scene slip (pre)')
        axes[5, ci].plot(t_np, -ecs_pre[:, 0],color='#1f4dab', lw=1.0, ls='--', label='−EC scene (pre)')
        ax_fmt(axes[5, ci])
        ylo, yhi = axes[5, ci].get_ylim()
        axes[5, ci].set_ylim(min(ylo, -1.0), max(yhi, 1.0))
        if ci == 0: axes[5, ci].legend(fontsize=7)

        # Row 6: POST-cascade slip + RAW EC cascade output.  EC sign-flipped so
        # it overlays the slip when the forward model is good (−EC ≈ slip ⇔
        # slip + EC ≈ 0).  The gap between them is the EC-vs-slip residual
        # (dynamics mismatch — MN/MLF model vs actual FCP path, fl_drive, plant).
        # This is the raw cerebellar EC cascade tail (acts.cb.ec_*), NOT scaled
        # by K_cereb / saccadic suppression / visibility — so it stays meaningful
        # even when pursuit is disabled (K_cereb_pu = 0).
        axes[6, ci].plot(t_np, vel_del[:, 0],               color='#7b2d8b', lw=1.2, label='tgt slip')
        axes[6, ci].plot(t_np, -ec_target_no_torsion[:, 0], color='#d62728', lw=1.0, ls=':',  label='−EC target (cascade)')
        axes[6, ci].plot(t_np, slip_del[:, 0],              color='#1a7a4a', lw=1.2, label='scene slip')
        axes[6, ci].plot(t_np, -ec_scene[:, 0],             color='#1f4dab', lw=1.0, ls='--', label='−EC scene (cascade)')
        ax_fmt(axes[6, ci])
        ylo, yhi = axes[6, ci].get_ylim()
        axes[6, ci].set_ylim(min(ylo, -1.0), max(yhi, 1.0))
        if ci == 0: axes[6, ci].legend(fontsize=7)

        # Row 7: total pursuit motor command + w_est (= VS net = the total
        # VOR/OKR drive) + omega_tvor.  The actual pursuit input (= what
        # brain_model passes to pu.step as target_slip_ec) is the SUM of the
        # brainstem-direct path and the cerebellar EC path:
        #   pursuit_in = K_pursuit_direct·sat·cyc.target_vel + K_cereb_pu·vpf_drive
        # NOT vpf_drive alone — vpf_drive (the cerebellar EC tail) can carry a
        # ~150 ms post-saccadic decay, but it is cancelled by the matching
        # retinal slip tail in the sum, so the true pursuit drive stays ~0.
        K_ph         = float(params.brain.K_phasic_pursuit)
        vpf_drive    = np.array(acts.cb.vpf_drive)                       # (T, 3) already sat·vis·ec
        pursuit_in   = (float(params.brain.K_pursuit_direct)
                          * gate_target_post[:, None] * vel_del
                        + float(params.brain.K_cereb_pu) * vpf_drive)   # (T, 3)
        e_pred       = (pursuit_in - x_purs) / (1.0 + K_ph)
        u_purs_arr   = x_purs + K_ph * e_pred
        w_est_arr    = vs_net(st)                            # (T, 3) VS net = w_est
        tvor_arr     = _omega_tvor_traj(st, params.brain)    # (T, 3) omega_tvor
        axes[7, ci].plot(t_np, u_purs_arr[:, 0], color='steelblue', lw=1.5, label='u_pursuit')
        axes[7, ci].plot(t_np, w_est_arr[:, 0],  color='#d45500',   lw=1.2, label='w_est = VS/OKR drive')
        axes[7, ci].plot(t_np, tvor_arr[:, 0],   color='#6a3d9a',   lw=1.0, ls='--', label='omega_tvor')
        ax_fmt(axes[7, ci])
        ylo, yhi = axes[7, ci].get_ylim()
        axes[7, ci].set_ylim(min(ylo, -1.0), max(yhi, 1.0))
        if ci == 0: axes[7, ci].legend(fontsize=7)

        # Row 8: vertical (pitch) + torsional (roll) eye velocity — diagnostic
        # check for unexpected off-axis components during pure-horizontal saccades.
        # Floor ylim at ±1 deg/s so autoscale never magnifies float-precision noise.
        axes[8, ci].plot(t_np, vel3d[:, 1], color='#1f78b4', lw=1.2, label='vertical (pitch)')
        axes[8, ci].plot(t_np, vel3d[:, 2], color='#b15928', lw=1.2, label='torsional (roll)')
        ax_fmt(axes[8, ci])
        ylo, yhi = axes[8, ci].get_ylim()
        axes[8, ci].set_ylim(min(ylo, -1.0), max(yhi, 1.0))
        axes[8, ci].set_xlabel('Time (s)', fontsize=8)
        if ci == 0: axes[8, ci].legend(fontsize=7)

    # Per-row x-axis limits.  Row 5 zooms into the pre-cascade window;
    # all other rows span the full trial.  Hide x-tick labels everywhere
    # except the zoom row and the bottom row.
    PRE_XLIM = (0.3, 0.6)
    FULL_XLIM = (float(t_np[0]), float(t_np[-1]))
    for r in range(n_rows):
        for ci in range(n_cols):
            ax = axes[r, ci]
            if r == 5:
                ax.set_xlim(*PRE_XLIM)
            else:
                ax.set_xlim(*FULL_XLIM)
            if r not in (5, n_rows - 1):
                ax.tick_params(labelbottom=False)
            else:
                ax.tick_params(labelbottom=True)
                ax.set_xlabel('Time (s)', fontsize=8)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, fname, show=show, params=THETA_NOISELESS,
                              conditions='Lit, foveal targets at 1°/5°/20°/40° (noiseless) + 1° (with noise)')
    # ── Quantitative metrics ─────────────────────────────────────────────────
    # Cascade reports post-saccadic STABILITY only (accuracy lives in the main-
    # sequence figure, which spans more amplitudes). Two numbers, both NOISELESS
    # (the noisy 1° column is noise-floor-dominated, not an EC signal): the
    # immediate ring and the sustained residual, worst-case over 1/5/20/40°.
    # Both should be ~0 — whatever remains is imperfect EC self-motion suppression.
    _aggmax = lambda xs: float(np.nanmax(xs)) if len(xs) and not np.all(np.isnan(xs)) else float('nan')
    metrics = [
        Metric('sac_postsac_peak_short_noiseless', _aggmax(spd_peak_s[False]),
               lo=0.0, hi=1.0, golden_tol=0.25, units='deg/s',
               cite='post-saccadic stability — EC self-motion suppression',
               desc='WORST-amplitude PEAK slow-phase speed in the SHORT window just after the '
                    'saccade — immediate ring/glissade (target ~0 = perfect EC suppression)'),
        Metric('sac_postsac_peak_long_noiseless', _aggmax(spd_peak_l[False]),
               lo=0.0, hi=0.8, golden_tol=0.25, units='deg/s',
               cite='post-saccadic stability — EC self-motion suppression',
               desc='WORST-amplitude PEAK slow-phase speed in the LONG hold window — sustained '
                    'post-saccadic oscillation/drift (target ~0 = perfect EC suppression)'),
        Metric('sac_postsac_verg_peak_noiseless', _aggmax(spd_verg[False]),
               lo=0.0, hi=1.5, golden_tol=0.25, units='deg/s',
               cite='post-saccadic stability — per-eye (MLF) motor asymmetry',
               desc='WORST-amplitude PEAK post-saccadic vergence (L−R) velocity — the per-eye '
                    'MN/MLF glissade asymmetry the version metric averages away (Side 2; target ~0)'),
        Metric('sac_cascade_count_noiseless', float(np.max(sac_count)) if sac_count else float('nan'),
               lo=1.0, hi=2.0, golden_tol=0.0, units='saccades',
               cite='burst stop integrity (Robinson 1975 local feedback)',
               desc='WORST-amplitude saccade count per noiseless target step — must be 1 (clean) or '
                    '2 (primary + 1 corrective); >2 ⇒ burst oscillating / mis-terminating'),
    ]

    fig = utils.fig_meta(path, rp,
        title='Saccade Signal Cascade',
        description='Row-by-row signal flow for the 1°/5°/20°/40° saccades (noiseless) plus a 1° saccade WITH '
                    'noise (last column, tinted): position, visual cascade + hold, accumulator/latch, residual '
                    'error, burst, eye velocity, suppression gates, EC vs slip, and pursuit/VS drives.',
        expected='e_held freezes at saccade onset; burst proportional to e_res; accumulator floor locks out the '
                 'next saccade for ~270 ms; the noise column adds fixational drift + occasional microsaccades.',
        citation='Robinson (1975) J Neurophysiol; Scudder et al. (2002); Kapoula, Robinson & Hain (1986)',
        fig_type='cascade')
    fig['metrics'] = metrics
    return fig


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='saccades', title='1. Saccades',
    description='Saccade kinematics and internal signal cascade. Tests main sequence (amplitude–velocity), '
                'oblique trajectory linearity, double-step refractoriness, and the full Robinson cascade. '
                'The cascade figure also shows the cerebellar saccadic-suppression gates (pre- and post-delay), '
                'the pre- vs post-cascade slip + efference-copy comparison (cerebellum EC matched to the retinal '
                'cascade incl. the two-stage MN / MLF forward model), and the pursuit / VS / T-VOR drives during a saccade.',
)


def run(show=False):
    print('\n=== Saccades ===')
    figs = []
    print('  1/4  main sequence …')
    figs.append(_main_sequence(show))
    print('  2/4  oblique saccades …')
    figs.append(_oblique(show))
    print('  3/4  double-step refractoriness …')
    figs.append(_refractoriness(show))
    print('  4/4  signal cascade (noiseless + 1° noise) …')
    figs.append(_cascade(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
