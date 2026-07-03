"""Post-hoc analysis helpers for oculomotor simulations.

Shared across notebooks and scripts — import instead of copy-pasting.

Functions
---------
vs_net(states)
    Net velocity storage signal x_L − x_R, shape (T, 3).

ni_net(states)
    Net neural integrator signal x_L − x_R, shape (T, 3).

vs_null(states)
    VS null-adaptation state, shape (T, 3).

ni_null(states)
    NI null-adaptation state, shape (T, 3).

pupil_size(states)
    Per-eye pupil diameter, shape (T, 2) = [left, right], mm.

luminance(states)
    Per-eye afferent retinal luminance, shape (T, 2) = [left, right], ~[0, 1].

extract_canal(states)
    Canal head-velocity estimate (yaw) from SimState trajectory.

extract_burst(states, theta)
    Recompute u_burst (T, 3) via vmap over the saccade generator.

extract_sg(states, theta)
    Full SG signal extraction: x_copy, e_held, z_opn, z_acc,
    e_res, e_pd, u_burst, x_ni.

extract_fcp_cascade(states, theta)
    Full final-common-pathway cascade as a dict: version + vergence
    pulse/step, AC/A, MN(14) firing, per-muscle nerves(12).

extract_z_opn(states)
    OPN latch state z_opn directly from state (no recomputation). (T,)

extract_spv_states(states, t, margin_s)
    Slow-phase velocity from SimState, all 3 axes (T, 3). Uses OPN latch.

extract_spv(t, eye_vel, burst=None, burst_threshold, margin_s, z_opn=None)
    Low-level SPV: mask one velocity trace. Prefers z_opn when given.

saccade_metrics(eye_pos_yaw, eye_vel_yaw, t_jump_idx, dt)
    Amplitude, peak velocity, and duration of the primary saccade.

fit_tc(t, y, t_start, t_end, label)
    Fit A·exp(−t/τ) + offset; return (tau, t_fit, y_fit).

ax_fmt(ax, ylabel, xlabel, ylim)
    Standard axes formatting (zero line, grid, labels, tick size).
"""

import numpy as np
import jax
import jax.numpy as jnp
from scipy.ndimage import binary_dilation, median_filter
from scipy.optimize import curve_fit

from oculomotor.models.sensory_models.sensory_model import (
    N_CANALS, FLOOR, _SOFTNESS, PINV_SENS,
)
from oculomotor.models.brain_models.perception_cyclopean import C_pos, C_target_visible
from oculomotor.models.brain_models import saccade_generator   as sg_mod
from oculomotor.models.brain_models import brain_model         as brain_mod
from oculomotor.models.brain_models import final_common_pathway as fcp_mod
from oculomotor.models.brain_models import vergence_accommodation as va_mod

try:
    from scipy.special import softplus as _sp_softplus
except ImportError:  # fallback
    def _sp_softplus(x):
        return np.log1p(np.exp(x))


# ── Brain registries (vmapped over time) ──────────────────────────────────────
# These wrap the per-step `read_activations` / `decode_activations` /
# `read_weights` from `brain_model` and the cyclopean output reader from
# `perception_cyclopean` so benches and notebooks can pull canonical signals
# directly from a (T, …) `SimState` trajectory without reaching into raw
# state slices.  Each returns a NamedTuple with the same field structure as
# the per-step registry, but with a leading time axis on every leaf.

def read_brain_acts(states, theta):
    """Brain-wide activations (T, …) — vmaps `brain_model.read_activations`."""
    return jax.vmap(lambda bs: brain_mod.read_activations(bs, theta.brain))(states.brain)


def read_brain_decoded(states, theta):
    """Brain-wide decoded push-pull nets (T, …) — vmaps `decode_activations`."""
    acts = read_brain_acts(states, theta)
    return jax.vmap(brain_mod.decode_activations)(acts)


def read_brain_weights(states):
    """Brain-wide tonic/null/setpoint registers (T, …) — vmaps `read_weights`."""
    return jax.vmap(brain_mod.read_weights)(states.brain)


# ── State extractors ──────────────────────────────────────────────────────────

# VS pops/null are stored in canal-plane coords [ω_H, ω_LARP, ω_RALP]; recombine
# to cardinal [yaw,pitch,roll] for readout (row-vector form: v @ C2Cᵀ).
_S_C2C = 2 ** -0.5
_CANAL2CARDINAL = np.array([[1.,  0.,       0.],
                            [0.,  _S_C2C,   _S_C2C],
                            [0., -_S_C2C,   _S_C2C]])
_C2C_JAX = jnp.asarray(_CANAL2CARDINAL)   # jnp copy for vmapped per-state reconstructions


def vs_net(states):
    """Net velocity storage signal: x_L − x_R, shape (T, 3), deg/s (cardinal)."""
    net_canal = np.array(states.brain.sm.vs_L - states.brain.sm.vs_R)
    return net_canal @ _CANAL2CARDINAL.T


def ni_net(states):
    """Net neural integrator signal: x_L − x_R, shape (T, 3), deg (cardinal)."""
    return np.array(states.brain.ni.L - states.brain.ni.R) @ _CANAL2CARDINAL.T


def vs_null(states):
    """VS null-adaptation state, shape (T, 3), deg/s (cardinal)."""
    return np.array(states.brain.sm.vs_null) @ _CANAL2CARDINAL.T


def ni_null(states):
    """NI null-adaptation state, shape (T, 3), deg (cardinal)."""
    return np.array(states.brain.ni.null) @ _CANAL2CARDINAL.T


# ── Pupil / luminance ────────────────────────────────────────────────────────

def pupil_size(states):
    """Per-eye pupil diameter, shape (T, 2) = [left, right], mm.

    Actual iris-plant output (physiological ~3–8 mm). The two pupils differ only
    under an efferent / iris lesion (anisocoria). For the avatar morph target,
    convert with d = (mm/2 − 1.54) / 4.35 (see web/avatar.js setPupilDiameter).
    """
    return np.array(states.pupil_plant)


def luminance(states):
    """Per-eye afferent retinal luminance, shape (T, 2) = [left, right], ~[0, 1].

    The low-passed light signal driving the pupillary light reflex (from the
    luminance sensory afferent), NOT the physical scene/target presence.
    """
    return np.array(states.sensory.lum.x_lum)


# ── Canal ──────────────────────────────────────────────────────────────────────

def extract_canal(states, params=None):
    """Canal head-velocity estimate (yaw) from a SimState trajectory.

    Applies the same push-pull nonlinearity as canal.nonlinearity() but in
    NumPy so it works on the full trajectory at once.

    Args:
        states: SimState with .sensory of shape (T, N_SENSORY_STATES)
        params: Params NamedTuple (reads sensory.canal_floor); uses FLOOR=80 if None

    Returns:
        (T,) array  yaw canal estimate (deg/s); positive = head rotating right.
    """
    x2   = np.array(states.sensory.canal.x2)             # (T, 6) inertia states
    k    = float(_SOFTNESS)
    f    = float(params.sensory.canal_floor) if params is not None else float(FLOOR)
    y_c  = -f + _sp_softplus(k * (x2 + f)) / k + _sp_softplus(k * (x2 - f)) / k
    pinv = np.array(PINV_SENS)
    return (pinv @ y_c.T).T[:, 0]                        # yaw component (T,)


# ── Saccade generator ──────────────────────────────────────────────────────────

def extract_burst(states, theta):
    """Recompute u_burst (T, 3) via vmap over the saccade generator.

    Reads the delayed position error and visual-field gate from the visual
    delay cascade at each time step, then calls sg.step() to get u_burst.

    Args:
        states: SimState with .sensory and .brain, shape (T, ...)
        theta:  Params NamedTuple (reads theta.brain for SG params)

    Returns:
        (T, 3) float array  saccade burst command (yaw, pitch, roll) in deg/s.
        Slice [:, 0] for yaw only.
    """
    # The cyclopean readout matrices C_pos / C_target_visible expect a flat
    # (43,) cyclopean state — convert via pc.to_array.
    from oculomotor.models.brain_models import perception_cyclopean as pc_mod
    def _at(state):
        x_vis    = pc_mod.to_array(state.brain.pc)
        e_pd     = C_pos @ x_vis
        gate     = (C_target_visible @ x_vis)[0]
        x_ni_net = _C2C_JAX @ (state.brain.ni.L - state.brain.ni.R)   # (3,) canal→cardinal
        sg_acts  = sg_mod.read_activations(state.brain.sg)
        sg_w     = sg_mod.read_weights(state.brain.sg)
        _, u     = sg_mod.step(sg_acts, sg_w, e_pd, gate, x_ni_net,
                                jnp.zeros(3), jnp.zeros(3), theta.brain)
        return u
    return np.array(jax.vmap(_at)(states))   # (T, 3)


# ── Final common pathway ────────────────────────────────────────────────────────

def extract_fcp_cascade(states, theta):
    """Recompute the full final-common-pathway cascade from a trajectory.

    Decomposes the conjugate (version) and vergence drive into Robinson
    pulse / step, then carries the command through the motoneurons to the
    per-muscle nerve drive — matching ``final_common_pathway.step`` exactly
    (same f-I clip, MLF strip, two-sided nerve clip, pull-only output).

    Args:
        states: SimState trajectory (T, …)
        theta:  Params NamedTuple

    Returns:
        dict of numpy arrays (angles deg, firing rates spk/s):
            eye        (T,3)  version eye position 0.5·(L+R)
            eye_verg   (T,)   ocular vergence L_yaw − R_yaw
            ni_L, ni_R (T,3)  NI bilateral pops (the version step)
            v_pulse    (T,3)  version pulse = τp·burst (feed-through)
            verg_step  (T,3)  vergence step = verg_fast + verg_tonic
            verg_pulse (T,3)  vergence pulse = τp·(K_phasic·disparity + SVBN)
            aca        (T,)   AC/A cross-link (horizontal only)
            mn         (T,14) motoneuron firing (≥0 nucleus output; AIN signed — MLF)
            nerves     (T,12) per-muscle nerve drive (pull-only)
    """
    bp       = theta.brain
    tau_p    = float(theta.plant.tau_p)
    NM       = float(fcp_mod._NERVE_MAX)
    AC_A     = float(bp.AC_A)
    DEG_PD   = float(va_mod._DEG_PER_PD)
    K_phasic = float(bp.K_phasic_verg)
    g_svbn   = float(bp.g_svbn_conv)
    X_svbn   = float(bp.X_svbn_conv)
    opn_tonic = float(sg_mod._OPN_TONIC)

    # version: eye position, NI step, burst feed-through pulse
    L = np.array(states.plant.left); R = np.array(states.plant.right)
    eye      = 0.5 * (L + R)
    eye_verg = L[:, 0] - R[:, 0]
    ni_L = np.array(states.brain.ni.L) @ _CANAL2CARDINAL.T   # canal→cardinal
    ni_R = np.array(states.brain.ni.R) @ _CANAL2CARDINAL.T
    v_pulse = tau_p * extract_burst(states, theta)

    # nucleus firing rate (14, by-nucleus) + the nerve (route + g_nerve conduction
    # lesion) — the same stages as fcp.step().
    mn = jnp.asarray(states.brain.fcp.mn)
    def _act(m): return fcp_mod.read_activations(fcp_mod.State(mn=m), bp).mn
    rates  = np.array(jax.vmap(_act)(mn))                                                                       # (T,14) ≥0 nucleus firing
    nerves = np.array(jax.vmap(lambda m: fcp_mod._smooth_clip(fcp_mod._ROUTE @ _act(m), bp.g_nerve * NM))(mn))  # (T,12) nerve

    # vergence pulse / step decomposition (H/V/T)
    verg_step = np.array(states.brain.va.verg_fast) + np.array(states.brain.va.verg_tonic)
    disp     = np.array(states.brain.pc.target_disparity)[:, -3:]            # current disparity
    vc       = np.array(states.brain.va.verg_copy)
    gate_opn = np.array(states.brain.sg.z_opn) / opn_tonic
    z_act    = np.clip(1.0 - gate_opn, 0.0, 1.0)[:, None]                    # (T,1) OPN open gate
    br       = disp - vc
    u_svbn   = z_act * np.sign(br) * g_svbn * (1.0 - np.exp(-np.abs(br) / X_svbn))
    verg_pulse = tau_p * (K_phasic * disp + u_svbn)
    aca = AC_A * DEG_PD * (np.array(states.brain.va.acc_fast)
                           + np.array(states.brain.va.acc_slow))             # (T,) H-only

    return dict(eye=eye, eye_verg=eye_verg, ni_L=ni_L, ni_R=ni_R, v_pulse=v_pulse,
                verg_step=verg_step, verg_pulse=verg_pulse, aca=np.asarray(aca),
                mn=rates, nerves=nerves)


def extract_sg(states, theta):
    """Full SG internal signal extraction.

    Returns a dict with all SG sub-states plus the recomputed burst.
    Useful for Robinson cascade diagnostic plots.

    Args:
        states: SimState trajectory (T, ...)
        theta:  Params NamedTuple

    Returns:
        dict with keys:
            e_held  (T, 3)  error estimator = residual error (e_res = e_held)
            z_opn   (T,)    OPN state (100=tonic, 0=paused during saccade)
            z_acc   (T,)    accumulator state
            e_pd    (T, 3)  delayed position error from visual cascade
            u_burst (T, 3)  recomputed saccade burst command
            x_ni    (T, 3)  neural integrator state (eye-position proxy)
    """
    sg_st = states.brain.sg
    e_held  = np.array(sg_st.e_held)
    z_opn   = np.array(sg_st.z_opn)
    z_acc   = np.array(sg_st.z_acc)
    z_trig  = np.array(sg_st.z_trig)
    x_ebn_R = np.array(sg_st.ebn_R)
    x_ebn_L = np.array(sg_st.ebn_L)
    x_ibn_R = np.array(sg_st.ibn_R)
    x_ibn_L = np.array(sg_st.ibn_L)

    # Eye-position estimate (NI net) and delayed position error (cyclopean)
    x_ni  = np.array(states.brain.ni.L - states.brain.ni.R) @ _CANAL2CARDINAL.T
    from oculomotor.models.brain_models import perception_cyclopean as pc_mod
    x_vis = np.array(jax.vmap(pc_mod.to_array)(states.brain.pc))   # (T, 43)
    e_pd  = x_vis @ np.array(C_pos).T   # (T, 3) cyclopean delayed position error

    u_burst = extract_burst(states, theta)

    return dict(
        e_held=e_held,
        z_opn=z_opn,     z_acc=z_acc,   z_trig=z_trig,
        x_ebn_R=x_ebn_R, x_ebn_L=x_ebn_L,
        x_ibn_R=x_ibn_R, x_ibn_L=x_ibn_L,
        e_pd=e_pd,       u_burst=u_burst, x_ni=x_ni,
    )


def extract_z_opn(states):
    """OPN latch state directly from simulation states — no recomputation.

    z_opn = 100 between saccades (OPN tonic, burst blocked).
    z_opn = 0   during saccades (OPN paused, burst active).

    Returns:
        (T,) z_opn array
    """
    return np.array(states.brain.sg.z_opn)


def _dilate_fast(is_fast, dt, margin_pre_s, margin_post_s):
    """Expand each fast-phase run by margin_pre_s before its start and
    margin_post_s after its end (ASYMMETRIC).  The post side covers post-saccadic
    plant ringing; the pre side is kept small because the slow phase *before* a
    quick phase is clean and the z_opn edge already marks onset sharply.  A
    symmetric ±50 ms margin throws away good pre-saccade slow-phase data and, at
    high oscillation frequencies, blanks whole cycles (corrupting Bode gain)."""
    f = np.asarray(is_fast, dtype=bool)
    out = f.copy()
    for k in range(1, int(round(margin_post_s / dt)) + 1):
        out[k:]  |= f[:-k]      # extend forward  (post-saccade settling)
    for k in range(1, int(round(margin_pre_s / dt)) + 1):
        out[:-k] |= f[k:]       # extend backward (pre-saccade)
    return out


def extract_spv_states(states, t, margin_pre_s=0.01, margin_post_s=0.06,
                       margin_s=None, eye='left', return_mask=False):
    """Slow-phase velocity from a SimState trajectory, all 3 axes.

    Uses the OPN latch state (z_opn) for fast-phase detection, which is reliable
    even when the slow phase is fast (high-velocity nystagmus, VN infarct, etc.).

    Args:
        states:   SimState trajectory
        t:        (T,) time array (s)
        margin_pre_s / margin_post_s: asymmetric window expansion before / after
                  each fast-phase epoch (s).  Default 0.01 / 0.06.
        margin_s: legacy symmetric override (sets pre=post=margin_s) for callers
                  that need the old behaviour.
        return_mask: if True, also return the (T,) boolean slow-phase mask (the
                  valid, non-masked samples) — used by the gap-aware Bode fit.
        eye:      'left'    — left-eye plant state (legacy default)
                  'right'   — right-eye plant state
                  'version' — cyclopean (L+R)/2 — true conjugate slow-phase velocity
                  'vergence' — (L−R), convergence positive — slow-phase vergence rate

    Returns:
        (T, 3) slow-phase velocity [yaw, pitch, roll] in deg/s.
        Access yaw with extract_spv_states(states, t)[:, 0].
    """
    dt    = float(t[1] - t[0])
    z_opn = extract_z_opn(states)
    ep_L  = np.array(states.plant.left)    # left-eye rotation vector (T, 3)
    ep_R  = np.array(states.plant.right)   # right-eye rotation vector (T, 3)
    if eye == 'left':
        ep = ep_L
    elif eye == 'right':
        ep = ep_R
    elif eye == 'version':
        ep = 0.5 * (ep_L + ep_R)
    elif eye == 'vergence':
        ep = ep_L - ep_R
    else:
        raise ValueError(f"eye={eye!r}; expected 'left', 'right', 'version', or 'vergence'")
    spv = np.stack([
        extract_spv(t, np.gradient(ep[:, i], dt), z_opn=z_opn,
                    margin_pre_s=margin_pre_s, margin_post_s=margin_post_s, margin_s=margin_s)
        for i in range(3)
    ], axis=1)
    if return_mask:
        mp, mq = (margin_s, margin_s) if margin_s is not None else (margin_pre_s, margin_post_s)
        slow = ~_dilate_fast(np.asarray(z_opn) < 50.0, dt, mp, mq)
        return spv, slow
    return spv


# ── Slow-phase velocity ────────────────────────────────────────────────────────

def extract_spv(t, eye_vel, burst=None, burst_threshold=10.0,
                margin_pre_s=0.01, margin_post_s=0.06, margin_s=None,
                z_opn=None, smooth_s=0.0, return_mask=False):
    """Slow-phase velocity by masking fast phases and interpolating.

    Prefers z_opn (OPN latch state) when provided — z_opn transitions sharply at
    saccade onset/offset regardless of how fast the slow phase is, so it never
    misclassifies a high-velocity slow phase as a fast phase.  Falls back to a
    burst-amplitude threshold when z_opn is not available.

    Args:
        t:               (T,) time array (s)
        eye_vel:         (T,) eye angular velocity (deg/s)
        burst:           (T,) burst signal (yaw component); used only when z_opn
                         is None.  At least one of burst / z_opn must be given.
        burst_threshold: deg/s — burst amplitude threshold (burst fallback only)
        margin_pre_s / margin_post_s: asymmetric expansion before / after each
                         fast-phase epoch (s). Post (~0.06) covers plant ringing;
                         pre (~0.01) is small (pre-saccade slow phase is clean).
        margin_s:        legacy symmetric override (sets pre=post=margin_s).
        return_mask:     if True, also return the (T,) boolean slow-phase mask.
        z_opn:           (T,) OPN state from SG or extract_z_opn(states).
                         Fast phases detected as z_opn < 50.
        smooth_s:        s — if > 0, median-filter the result over this window to
                         de-spike residual quick-phase contamination and within-beat
                         jitter while preserving the slow-phase shape (0 = off).

    Returns:
        (T,) slow-phase velocity — fast-phase samples replaced by linear
        interpolation across the masked epochs (optionally median-smoothed).
    """
    dt = float(t[1] - t[0])
    if margin_s is not None:                       # back-compat: symmetric ±margin_s
        margin_pre_s = margin_post_s = margin_s
    if z_opn is not None:
        is_fast = np.asarray(z_opn) < 50.0
    elif burst is not None:
        is_fast = np.abs(np.asarray(burst)) > burst_threshold
    else:
        raise ValueError("extract_spv: provide either burst or z_opn")
    is_fast = _dilate_fast(is_fast, dt, margin_pre_s, margin_post_s)
    slow    = ~is_fast
    if slow.sum() < 2:
        return (eye_vel.copy(), slow) if return_mask else eye_vel.copy()
    spv = np.interp(t, t[slow], eye_vel[slow])
    if smooth_s and smooth_s > 0:
        w = max(1, int(round(smooth_s / dt)))
        if w % 2 == 0:
            w += 1
        spv = median_filter(spv, size=w, mode='nearest')
    return (spv, slow) if return_mask else spv


# ── Saccade kinematics ────────────────────────────────────────────────────────

def saccade_metrics(eye_pos_yaw, eye_vel_yaw, t_jump_idx, dt):
    """Amplitude, peak velocity, and duration of the primary saccade.

    Duration is measured from the first crossing of 10% peak velocity to the
    first time velocity drops back below 10% peak velocity AFTER the peak —
    ignoring any corrective saccades that follow.

    Args:
        eye_pos_yaw: (T,) eye position yaw (deg)
        eye_vel_yaw: (T,) eye velocity yaw (deg/s)
        t_jump_idx:  int  index of the target jump in the arrays
        dt:          float  sample period (s)

    Returns:
        (amplitude_deg, peak_velocity_degs, duration_ms)
    """
    vel = eye_vel_yaw[t_jump_idx:]
    pos = eye_pos_yaw[t_jump_idx:]

    peak_v    = float(np.max(np.abs(vel)))
    threshold = 0.1 * peak_v
    above     = np.abs(vel) > threshold

    if above.any():
        start     = int(np.argmax(above))
        peak_idx  = int(np.argmax(np.abs(vel)))
        after_peak = np.abs(vel[peak_idx:]) < threshold
        end = peak_idx + int(np.argmax(after_peak)) if after_peak.any() else len(vel) - 1
        dur = (end - start) * dt * 1000      # ms
        amp = float(pos[end] - pos[0])
    else:
        dur = float('nan')
        amp = float(pos[-1] - pos[0])

    return amp, peak_v, dur


# ── Time-constant fitting ─────────────────────────────────────────────────────

def fit_tc(t, y, t_start, t_end, label=''):
    """Fit A·exp(−t/τ) + offset to a segment of a signal.

    Args:
        t:       (T,) time array (s)
        y:       (T,) signal array
        t_start: float  start of fit window (s)
        t_end:   float  end of fit window (s)
        label:   str    printed with the result (optional)

    Returns:
        (tau, t_fit, y_fit) on success, or (None, None, None) on failure.
        tau:   fitted time constant (s)
        t_fit: time array for the fit segment
        y_fit: fitted curve values
    """
    mask  = (t >= t_start) & (t <= t_end)
    t_seg = t[mask] - t_start
    y_seg = y[mask]
    try:
        p0   = (y_seg[0] - y_seg[-1], (t_end - t_start) / 3, y_seg[-1])
        popt, _ = curve_fit(
            lambda t, A, tau, c: A * np.exp(-t / tau) + c,
            t_seg, y_seg, p0=p0, maxfev=8000,
            bounds=([-np.inf, 0.1, -np.inf], [np.inf, 200, np.inf]))
        tau   = abs(popt[1])
        y_fit = popt[0] * np.exp(-t_seg / tau) + popt[2]
        if label:
            print(f'  {label}: τ = {tau:.1f} s')
        return tau, t[mask], y_fit
    except Exception as e:
        if label:
            print(f'  {label}: fit failed — {e}')
        return None, None, None


# ── Plot formatting ───────────────────────────────────────────────────────────

def ax_fmt(ax, ylabel='', xlabel='', ylim=None):
    """Standard axes formatting: zero line, grid, labels, tick size."""
    ax.axhline(0, color='gray', lw=0.4)
    ax.grid(True, alpha=0.2)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(labelsize=7)
