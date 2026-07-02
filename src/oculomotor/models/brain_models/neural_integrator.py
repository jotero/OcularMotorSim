"""Neural Integrator SSM — bilateral push-pull with null-point adaptation.

Mirrors the velocity_storage bilateral architecture.  Two populations (Left,
Right) model the bilateral nucleus prepositus hypoglossi (NPH) / interstitial
nucleus of Cajal (INC) organisation.

State:  x_ni = [x_L (3,) | x_R (3,) | x_null (3,)]         (9,)
Input:  u_vel                                                 (3,)  velocity command
Output: u_p — pulse-step motor command to plant               (3,)

ABCD system (bilateral core):
────────────────────────────────────────────────────────────────────────
    dx_L /dt = −(1/τ_i)·(x_L − b_ni − x_null/2) + u_vel/2
    dx_R /dt = −(1/τ_i)·(x_R − b_ni + x_null/2) − u_vel/2

    Net position:  x_net = x_L − x_R   (identical to old scalar x_ni)
    d(x_net)/dt  = −(1/τ_i)·(x_net − x_null) + u_vel      ← leaks toward null, not 0

    u_p  =  x_net  +  τ_p · u_vel     (pulse-step: lag-cancelled motor command)

Null-point adaptation:
────────────────────────────────────────────────────────────────────────
    dx_null/dt = (x_net − x_null) / τ_ni_adapt

    The null slowly tracks the current net position.  During sustained eccentric
    gaze (x_net = E), x_null → E.  On return to centre (x_net = 0), the NI leaks
    toward x_null = E → eye drifts back (slow phase eccentric) → fast phase toward
    centre → rebound nystagmus. ✓

    τ_ni_adapt default 20 s: ~half-period rebound after sustained right gaze.
    τ_ni_adapt → ∞ (very large):  null frozen at 0 → reverts to old NI behaviour.

Bilateral conventions (mirror velocity_storage):
    Model LEFT pop  (x_L, 0:3) codes RIGHTWARD gaze = anatomical RIGHT NPH.
    Model RIGHT pop (x_R, 3:6) codes LEFTWARD  gaze = anatomical LEFT  NPH.
    Net x_L − x_R > 0  →  rightward eye position command.

    b_ni (default 0):  NPH intrinsic resting bias.  At net level (C @ x = 0) the
    resting discharge cancels, so b_ni=0 is physiologically appropriate unless
    modelling unilateral NPH lesions (future work).

Anti-windup:
    Applied to the net derivative d(x_net)/dt before distributing back to
    individual populations.  Prevents integrator wind-up beyond ±orbital_limit.

Parameters:
    τ_i         (s)   — leak TC.         Default 25 s (healthy).
    τ_p         (s)   — plant TC copy.   Default 0.15 s.
    b_ni        (deg) — NPH resting bias. Default 0.
    τ_ni_adapt  (s)   — null adaptation TC.  Default 20 s.
    orbital_limit (deg) — oculomotor range half-width.  Default 50 deg.
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor import config as _config
# The NI carries eye position in canal-plane coords [H, LARP, RALP]; cardinal is
# reconstructed at the sinks (u_p → FCP, decoded net → SG/T-VOR/Listing, anti-windup).
from oculomotor.models.brain_models.perception_self_motion import (
    CANAL2CARDINAL, CARDINAL2CANAL,
)

# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """NI state — bilateral push-pull pops + null adaptation + lead filter.

    Pops are in CANAL-PLANE coordinates [H, LARP, RALP]: H (horizontal) = NPH
    (nucleus prepositus hypoglossi) + MVN; LARP/RALP (vertical/torsional) = INC
    (interstitial nucleus of Cajal).  Cardinal eye position is reconstructed at
    the FCP sink, CANAL2CARDINAL·(L − R)."""
    L:    jnp.ndarray   # (3,) pop A, canal-plane [H, LARP, RALP]  (rectified ≥ 0)
    R:    jnp.ndarray   # (3,) pop B, canal-plane [H, LARP, RALP]  (rectified ≥ 0)
    null: jnp.ndarray   # (3,) signed adaptation register, canal-plane (drifts toward x_net)
    u_lp: jnp.ndarray   # (3,) fast-LP of u_vel — supplies smoothed u_vel'
                        # for the 2nd-order pulse-step (MN-LP cancellation)


class Activations(NamedTuple):
    """NI firing rates — bilateral pops only (null is a setpoint, in Weights)."""
    L: jnp.ndarray   # (3,) left  NPH/INC pop
    R: jnp.ndarray   # (3,) right NPH/INC pop


class Decoded(NamedTuple):
    """NI decoded readout — net eye position consumed by FCP."""
    net: jnp.ndarray   # (3,) signed = L − R   eye position estimate (deg)


class Weights(NamedTuple):
    """NI tonic / null / setpoint registers (long-term: learned weights)."""
    null: jnp.ndarray   # (3,) signed   slow null adaptation register
    u_lp: jnp.ndarray   # (3,) fast-LP register supplying smoothed u_vel'
                        # for the 2nd-order MN-LP cancellation in the pulse-step


def rest_state():
    """Zero state — used for SimState initialisation."""
    return State(L=jnp.zeros(3), R=jnp.zeros(3), null=jnp.zeros(3), u_lp=jnp.zeros(3))


def read_activations(state):
    """NI bilateral pops are firing rates — RECTIFIED (max 0).

    step() is driven by these activations, so the rectification sits INSIDE the
    recurrent loop (the recurrence is fed the rectified rate, not the raw state).
    With the b_ni resting baseline the pops sit above threshold across the whole
    oculomotor range, so relu(x)=x and this is inert / transparent in normal
    operation.  It engages only when a pop is driven below floor — extreme
    eccentric gaze, or a lesion — where the off-direction pop cuts off; that
    cutoff is what lets the two-population structure express direction-dependent
    (asymmetric) holding, and it also prevents the off-pop from feeding the loop."""
    return Activations(L=jnp.maximum(state.L, 0.0), R=jnp.maximum(state.R, 0.0))


def decode_states(acts):
    """NI net eye position — canal-plane pops recombined to cardinal [yaw,pitch,roll]."""
    return Decoded(net=CANAL2CARDINAL @ (acts.L - acts.R))


def read_weights(state):
    """NI null adaptation register + lead-filter register."""
    return Weights(null=state.null, u_lp=state.u_lp)


def step(activations, weights, u_vel, brain_params, u_tonic=0.0):
    """Single ODE step: bilateral NI dynamics + null adaptation + motor command.

    Activation-driven: bilateral pop firing rates come from `activations`
    (acts.ni); the null adaptation register comes from `weights` (weights.ni).

    Args:
        activations:  ni.Activations  (L, R) firing rates, each (3,)
        weights:      ni.Weights      (null,) adaptation register, (3,)
        u_vel:        (3,)  combined eye-velocity command (deg/s) — sign-flipped upstream
        brain_params: BrainParams
        u_tonic:      (3,)  tonic position-offset set-point (e.g. OCR).
                            Acts as a shift on x_null for the population leak target,
                            so x_net leaks toward (x_null + u_tonic). A saccade landing
                            at the OCR position is therefore stable (no drift back to 0).
                            Not added to u_p directly — it flows through the integrator,
                            so x_ni already reflects the offset and ec_pos stays
                            consistent with the actual eye position.

    Returns:
        dstate: ni.State   state derivative (L, R, null)
        u_p:    (3,)       pulse-step motor command to plant
    """
    x_L    = activations.L        # canal-plane pops [H, LARP, RALP]
    x_R    = activations.R
    x_null = weights.null

    # u_vel and u_tonic arrive CARDINAL (the brain_model merge is cardinal); rotate
    # into the canal-plane basis at entry.  u_p and the decoded net are reconstructed
    # to cardinal at the sinks (u_p → FCP; decode_states → SG/T-VOR/Listing).
    u_vel_c   = CARDINAL2CANAL @ jnp.asarray(u_vel, dtype=jnp.float32)
    u_tonic_c = CARDINAL2CANAL @ (jnp.zeros(3, dtype=jnp.float32) + u_tonic)

    b_ni   = jnp.asarray(brain_params.b_ni,  dtype=jnp.float32)   # uniform → basis-free
    L      = brain_params.orbital_limit
    # Leak A_leak: the cardinal per-axis tau_i (yaw + pitch/roll fractions) expressed
    # in the canal-plane basis via the EXACT rotation M·diag·Mᵀ — behaviour-identical
    # for any frac.  It is NON-diagonal when pitch≠roll (torsional integrator leakier,
    # Crawford & Vilis 1991 → roll_frac=0.3): roll<pitch is a sum/difference (cardinal)
    # anisotropy, so in canal coords it lives as LARP↔RALP coupling.  If torsion is
    # later left to Listing's law (roll_frac→1.0) this auto-diagonalises like the VS.
    tau_i_card = brain_params.tau_i * jnp.array([1.0,
                                                 brain_params.tau_i_pitch_frac,
                                                 brain_params.tau_i_roll_frac])
    A_leak = CARDINAL2CANAL @ jnp.diag(-1.0 / tau_i_card) @ CANAL2CARDINAL   # (3×3) canal leak

    # u_tonic shifts the effective null/leak target without altering the stored
    # x_null state. Without quick-phase resets, x_net only reaches a fraction
    # τ_ni_adapt / (τ_i + τ_ni_adapt) ≈ 0.44 of u_tonic at SS — saccades and
    # quick phases drive the rest of the way (visible in the OCR cascade bench).
    x_null_eff = x_null + u_tonic_c

    # ── Population equilibria: leak toward b_ni ± half-(shifted)-null ────────
    # b_eff_L = b_ni + x_null_eff/2   (left  pop target rises with rightward null)
    # b_eff_R = b_ni - x_null_eff/2   (right pop target falls with rightward null)
    dx_L_raw = A_leak @ (x_L - b_ni - x_null_eff / 2.0) + u_vel_c / 2.0
    dx_R_raw = A_leak @ (x_R - b_ni + x_null_eff / 2.0) - u_vel_c / 2.0

    # ── Anti-windup on net — clipped in CARDINAL (the orbital limit is a physical
    # eye-position bound; clipping canal-plane components would mis-limit vertical/
    # torsional gaze).  Reconstruct the cardinal net + net-derivative, clip per
    # cardinal axis, rotate the clipped derivative back to canal.
    x_net   = x_L - x_R                       # canal net position
    dx_net  = dx_L_raw - dx_R_raw             # canal net derivative before clipping
    dx_sum  = dx_L_raw + dx_R_raw             # common-mode: unaffected by windup

    x_net_card  = CANAL2CARDINAL @ x_net
    dx_net_card = CANAL2CARDINAL @ dx_net
    dx_net_card = jnp.where(x_net_card >= L,  jnp.minimum(dx_net_card, 0.0), dx_net_card)
    dx_net_card = jnp.where(x_net_card <= -L, jnp.maximum(dx_net_card, 0.0), dx_net_card)
    dx_net      = CARDINAL2CANAL @ dx_net_card  # clipped net derivative, back to canal

    # Reconstruct individual derivatives from clipped net + unchanged sum
    dx_L = (dx_net + dx_sum) / 2.0
    dx_R = (dx_sum - dx_net) / 2.0

    # ── Null adaptation: null tracks (x_net − x_null_eff) ────────────────────
    # With sustained u_tonic and no input the system has a 1-D family of
    # equilibria along x_net = x_null + u_tonic. Starting from (0,0) it settles
    # at  x_net = u_tonic·τ_ni_adapt/(τ_i+τ_ni_adapt)  and
    #     x_null = -u_tonic·τ_i/(τ_i+τ_ni_adapt)  on TC τ_eff = τ_i·τ_ni_adapt/(τ_i+τ_ni_adapt).
    # So the null partially adapts to OCR — when OCR is later removed, x_null
    # stays negative briefly and drives a small post-OCR rebound, which is at
    # least directionally consistent with reported post-tilt-removal drift.
    dx_null = (x_net - x_null_eff) / brain_params.tau_ni_adapt

    # ── Pulse-step motor command: full MN_LP × plant_LP lag cancellation ──────
    # Effective plant from the NI's perspective is `MN_LP × plant_LP` — the FCP
    # motoneurons add a per-axis LP between motor_cmd_ni and the muscle nerves.
    # In Laplace:
    #     plant_inverse(s) = (1 + s·tau_p)(1 + s·tau_mn_eff)
    #                      = 1 + (tau_p + tau_mn_eff)·s + tau_p·tau_mn_eff·s²
    # → in time, motor_cmd = x_net + (tau_p + tau_mn_eff)·u_vel + tau_p·tau_mn_eff·u_vel'.
    # The first two terms are algebraic.  The second-derivative term needs u_vel';
    # implement via a fast lead-filter state `u_lp` (LP of u_vel at tau_fast ~ dt):
    #   du_lp/dt = (u_vel − u_lp) / tau_fast       state — fast 1st-order LP
    #   u_vel'   ≈ du_lp/dt = (u_vel − u_lp)/tau_fast    smoothed derivative
    # tau_mn_eff per axis: yaw uses mn_ff_yaw·tau_mn; pitch/roll use tau_mn.
    # mn_ff_yaw is the conjugate-yaw MN-LP feedforward factor. The version-
    # averaged value is 1.5 (half ABN→LR direct 1-stage + half AIN→MLF→CN3_MR
    # 2-stage). That averaging over- and under-compensates the two eyes equally;
    # the EXACT per-eye split is mn_ff_yaw=1.0 (the common 1st stage, here) +
    # fcp.mlf_lead (the adducting MR's extra stage, in the FCP). 1.5 + mlf_lead=0
    # = the legacy compromise.
    # tau_fast = dt = 0.001 s sets the lead-filter pole — the residual eye lag
    # behind NI_net is ~tau_fast (~1 ms), down from ~tau_mn_eff (~7 ms).
    tau_mn_eff   = brain_params.tau_mn * jnp.array([brain_params.mn_ff_yaw, 1.0, 1.0])
    # Fast poles to cancel: the MN membrane (tau_mn_eff, per-axis conjugacy factor)
    # AND the muscle force-development pole (tau_muscle, uniform) of the 2nd-order
    # plant.  Both are << the orbital tau_p, so lump them into one fast pole (the
    # residual is only the small cross-term τ_musc·τ_mn).  Effective plant then
    # ≈ (1+s·tau_p)(1+s·tau_fast_pole), inverted by the pulse + accel terms below.
    tau_fast_pole = tau_mn_eff + brain_params.tau_muscle
    tau_fast     = _config.DT_SOLVE   # lead-filter pole; tracks solver dt so the
                                      # filter stays at dt/tau_fast = 1 (Heun-stable)
                                      # as DT_SOLVE is raised. Residual eye lag ~tau_fast.
    u_vel_dot    = (u_vel_c - weights.u_lp) / tau_fast     # smoothed u_vel' (canal)
    du_lp        = u_vel_dot                                # state derivative of u_lp
    # Pulse-step computed canal-side; its only anisotropy (mn_ff_yaw on H, vertical
    # isotropic) commutes with the rotation, so reconstructing u_p to cardinal equals
    # the cardinal pulse-step exactly.  Cardinal u_p → FCP muscle map (the sink).
    u_p_canal = (x_net
                 + (brain_params.tau_p + tau_fast_pole) * u_vel_c
                 + (brain_params.tau_p * tau_fast_pole) * u_vel_dot)
    u_p = CANAL2CARDINAL @ u_p_canal

    return State(L=dx_L, R=dx_R, null=dx_null, u_lp=du_lp), u_p


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 12  # x_L(3) + x_R(3) + x_null(3) + u_lp(3)
N_INPUTS  = 3
N_OUTPUTS = 3


def from_array(x_ni):
    """(12,) flat array → ni.State."""
    return State(L=x_ni[0:3], R=x_ni[3:6], null=x_ni[6:9], u_lp=x_ni[9:12])


def to_array(state):
    """ni.State → (12,) flat array."""
    return jnp.concatenate([state.L, state.R, state.null, state.u_lp])
