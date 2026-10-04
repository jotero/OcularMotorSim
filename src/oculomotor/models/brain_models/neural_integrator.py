"""Neural integrator — bilateral push-pull populations, pulse-slide-step output,
and null-point adaptation.

Two integrator populations (L, R) model the bilateral organisation of the nucleus
prepositus hypoglossi + MVN (horizontal) and the interstitial nucleus of Cajal
(vertical / torsional), in canal-plane coordinates [H, LARP, RALP]. Two slide
populations (slide_L, slide_R) carry the low-passed velocity command for the
pulse-slide-step output.

The module is split into fast dynamics and slow adaptation (learning):

step_dynamics — block state-space form (canal-plane):
────────────────────────────────────────────────────────────────────────
    x   = [L | R | slide_L | slide_R]    (12,)  rectified pops (integrator + slide)
    u   = [u_vel⁺ | u_vel⁻]              (6,)   velocity command split into its two
                                                rectified halves
    dx  = A·(x − x_rest) + B·u
    x_rest = [b_ni + setpoint/2 | b_ni − setpoint/2 | 0 | 0]   resting point of each pop;
                                                setpoint = null + u_tonic
    u_p = C·x + D·u                      (rotated to cardinal → final common pathway)

  Per population this is
    dL/dt       = leak·(L − b_ni − setpoint/2) + u_vel/2
    dR/dt       = leak·(R − b_ni + setpoint/2) − u_vel/2
    dslide_L/dt = (u_vel⁺ − slide_L)/Ts,    dslide_R/dt = (u_vel⁻ − slide_R)/Ts
  so the net position x_net = L − R leaks toward the set point, not toward 0:
    d(x_net)/dt = leak·(x_net − setpoint) + u_vel
  the net slide = slide_L − slide_R is exactly LP(u_vel, Ts) (the low-pass is linear),
  and the motor command is the pulse-slide-step (Optican & Miles 1985):
    u_p = x_net + (τ_p + τ_fast − Ts)·slide + τ_p·τ_fast·slide'

  Two nonlinear wrappers stay outside the matrices: the populations are rectified
  inside the recurrence (inert in the normal range), and anti-windup clips the net
  derivative at ±orbital_limit (in cardinal).

step_adaptation — null-point adaptation (slow, local statistical learning):
────────────────────────────────────────────────────────────────────────
    dnull/dt = (x_net − setpoint) / τ_ni_adapt

  During sustained eccentric gaze the null drifts toward the held position. Back at
  centre the integrator leaks toward that null → slow phases away from centre with
  corrective fast phases → rebound nystagmus. τ_ni_adapt → ∞ freezes the null.

Bilateral conventions (mirror velocity storage):
    Model LEFT pop  (L) codes RIGHTWARD gaze = anatomical RIGHT NPH.
    Model RIGHT pop (R) codes LEFTWARD  gaze = anatomical LEFT  NPH.
    Net L − R > 0  →  rightward eye position command.

    b_ni: uniform resting rate of both integrator populations. It cancels in the net
    (no effect on eye position) but sets the rectification floor: with
    b_ni ≥ orbital_limit/2 the pops stay above zero across the oculomotor range.

    slide_L / slide_R: direction-selective populations, each low-passing one rectified
    half of the velocity command, so they stay ≥ 0 with no resting bias (silent at
    rest, like burst neurons). Anatomical identity provisional: Optican & Miles' slide
    component, plausibly NPH/MVN burst-tonic cells.

Parameters (defaults in BrainParams): tau_i (+ tau_i_pitch_frac, tau_i_roll_frac), b_ni,
tau_ni_adapt, tau_p, tau_mn + mn_ff_yaw, tau_muscle, tau_slide, orbital_limit.
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor import config as _config
# The NI carries eye position in canal-plane coords [H, LARP, RALP]. It is rotated back
# to cardinal wherever cardinal is needed: the motor command u_p (→ final common pathway),
# the decoded net position (→ saccade generator, T-VOR, Listing), and the orbital-limit clip.
from oculomotor.models.brain_models.perception_self_motion import (
    CANAL2CARDINAL, CARDINAL2CANAL,
)

# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """NI state — bilateral integrator pops + bilateral slide pops + null adaptation.

    Pops are in CANAL-PLANE coordinates [H, LARP, RALP]: H (horizontal) = NPH
    (nucleus prepositus hypoglossi) + MVN; LARP/RALP (vertical/torsional) = INC
    (interstitial nucleus of Cajal).  Cardinal eye position is reconstructed at
    the FCP sink, CANAL2CARDINAL·(L − R)."""
    L:    jnp.ndarray   # (3,) pop A, canal-plane [H, LARP, RALP]  (rectified ≥ 0)
    R:    jnp.ndarray   # (3,) pop B, canal-plane [H, LARP, RALP]  (rectified ≥ 0)
    null: jnp.ndarray   # (3,) signed adaptation register, canal-plane (drifts toward x_net)
    slide_L: jnp.ndarray   # (3,) slide pop: LP of the positive half of u_vel (≥ 0)
    slide_R: jnp.ndarray   # (3,) slide pop: LP of the negative half of u_vel (≥ 0)


class Activations(NamedTuple):
    """NI firing rates — integrator and slide pops (null is a set point, in Weights)."""
    L:       jnp.ndarray   # (3,) left  NPH/INC integrator pop
    R:       jnp.ndarray   # (3,) right NPH/INC integrator pop
    slide_L: jnp.ndarray   # (3,) slide pop, positive-direction half (provisional anatomy)
    slide_R: jnp.ndarray   # (3,) slide pop, negative-direction half


class Decoded(NamedTuple):
    """NI decoded readout — net eye position consumed by FCP."""
    net: jnp.ndarray   # (3,) signed = L − R   eye position estimate (deg)


class Weights(NamedTuple):
    """NI learned register — the null set point (slow statistical learning)."""
    null: jnp.ndarray   # (3,) signed   slow null adaptation register


def rest_state():
    """Zero state — used for SimState initialisation."""
    return State(L=jnp.zeros(3), R=jnp.zeros(3), null=jnp.zeros(3),
                 slide_L=jnp.zeros(3), slide_R=jnp.zeros(3))


def read_activations(state):
    """NI bilateral pops are firing rates — RECTIFIED (max 0).

    step() is driven by these activations, so the rectification sits INSIDE the
    recurrent loop (the recurrence is fed the rectified rate, not the raw state).
    With the b_ni resting baseline the pops sit above threshold across the whole
    oculomotor range, so relu(x)=x and this is inert / transparent in normal
    operation.  It engages only when a pop is driven below floor — extreme
    eccentric gaze, or a lesion — where the off-direction pop cuts off; that
    cutoff is what lets the two-population structure express direction-dependent
    (asymmetric) holding, and it also prevents the off-pop from feeding the loop.

    The slide pops are non-negative by construction (each low-passes a rectified
    input); rectifying them here is only a guard."""
    return Activations(L=jnp.maximum(state.L, 0.0), R=jnp.maximum(state.R, 0.0),
                       slide_L=jnp.maximum(state.slide_L, 0.0),
                       slide_R=jnp.maximum(state.slide_R, 0.0))


def decode_states(acts):
    """NI net eye position — canal-plane pops recombined to cardinal [yaw,pitch,roll]."""
    return Decoded(net=CANAL2CARDINAL @ (acts.L - acts.R))


def read_weights(state):
    """NI null adaptation register (the module's only learned register)."""
    return Weights(null=state.null)


# ── Fast dynamics: bilateral populations + pulse-slide-step output ─────────────

def step_dynamics(activations, setpoint_c, u_vel_c, brain_params):
    """Fast NI dynamics in block state-space form (all canal-plane).

        x   = [L | R | slide_L | slide_R]    (12,)
        u   = [u_vel⁺ | u_vel⁻]              (6,)
        dx  = A·(x − x_rest) + B·u            then anti-windup on the net (nonlinear)
        u_p = C·x + D·u                       rotated to cardinal

    Strictly dx = A·(φ(x) − x_rest) + B·u with φ the rectification of the populations
    (activations = max(0, state)); it is inert in the normal oculomotor range.

    Args:
        activations:  ni.Activations  (L, R, slide_L, slide_R) rectified rates, (3,) each
        setpoint_c:   (3,)  leak target of the net position = null + u_tonic
        u_vel_c:      (3,)  velocity command (signed; split into rectified halves here)
        brain_params: BrainParams

    Returns:
        dL, dR, dslide_L, dslide_R: (3,) each  state derivatives
        u_p:                        (3,)       pulse-slide-step motor command, CARDINAL → FCP
    """
    I = jnp.eye(3)
    Z = jnp.zeros((3, 3))

    # ── Per-axis leak (canal-plane) ──────────────────────────────────────────
    # The cardinal per-axis tau_i (yaw + pitch/roll fractions) expressed
    # in the canal-plane basis via the EXACT rotation M·diag·Mᵀ — behaviour-identical
    # for any frac.  It is NON-diagonal when pitch≠roll (torsional integrator leakier,
    # Crawford & Vilis 1991 → roll_frac=0.3): roll<pitch is a sum/difference (cardinal)
    # anisotropy, so in canal coords it lives as LARP↔RALP coupling.  If torsion is
    # later left to Listing's law (roll_frac→1.0) this auto-diagonalises like the VS.
    tau_i_card = brain_params.tau_i * jnp.array([1.0,
                                                 brain_params.tau_i_pitch_frac,
                                                 brain_params.tau_i_roll_frac])
    leak = CARDINAL2CANAL @ jnp.diag(-1.0 / tau_i_card) @ CANAL2CARDINAL   # (3×3)

    # ── Pulse-slide-step coefficients (Optican & Miles 1985) ─────────────────
    # Effective plant (NI → eye) = orbital LP (tau_p) × lumped fast pole
    # (tau_fast_pole = MN membrane tau_mn_eff + muscle force-development tau_muscle).
    # The exact inverse (1+s·tau_p)(1+s·tau_fast_pole)·x_net has a second-derivative
    # (acceleration) term; realised as a raw derivative it spikes at the OPN-clamp
    # burst offset and the eye RINGS (a glissade, in Optican & Miles' terms — a
    # slide/step mismatch).
    #
    # Their fix: the SLIDE — a low-pass of the pulse, with time constant Ts, summed
    # with pulse and step.  It realises the 2nd-order compensation as a SMOOTH branch
    # (two zeros cancel the two plant poles; the slide pole 1/Ts makes it proper), so
    #     eye = LP(NI_net, Ts)   — a clean 1st-order lag, NO overshoot, NO ring.
    #     u_p = x_net + (tau_p + tau_fast_pole − Ts)·slide + (tau_p·tau_fast_pole)·slide'
    # with slide = LP(u_vel, Ts) and slide' = (u_vel − slide)/Ts its smooth derivative.
    # The velocity term uses the SMOOTHED velocity (slide), not raw u_vel — that
    # consistency is what removes the ring (a raw-velocity term + smoothed accel
    # under-compensates the slide → glissade).  Ts→dt recovers the sharp inverse;
    # larger Ts rounds the pulse (small peak-vel cost, sets the eye lag).  Ts=brain
    # tau_slide.  Full derivation: manuscripts/pulse_slide_step.md
    #
    # mn_ff_yaw: conjugate-yaw MN-LP feedforward factor (× tau_mn) on the H axis; the
    # exact per-eye split is mn_ff_yaw=1.0 (common 1st stage) + fcp.mlf_lead (in FCP).
    # The only anisotropy (mn_ff_yaw on H, vertical isotropic) commutes with the
    # canal↔cardinal rotation, so computing u_p canal-side and rotating it is exact.
    tau_mn_eff    = brain_params.tau_mn * jnp.array([brain_params.mn_ff_yaw, 1.0, 1.0])
    tau_fast_pole = tau_mn_eff + brain_params.tau_muscle               # lumped fast pole
    Ts            = jnp.maximum(brain_params.tau_slide, _config.DT_SOLVE)   # slide TC (≥ dt)
    c_slide = brain_params.tau_p + tau_fast_pole - Ts                  # × slide
    c_accel = brain_params.tau_p * tau_fast_pole                       # × slide'

    # ── Recurrent connectivity A (12×12) ─────────────────────────────────────
    #   rows: dL, dR, dslide_L, dslide_R   ·   cols: L, R, slide_L, slide_R
    A = jnp.block([[leak, Z,    Z,       Z      ],
                   [Z,    leak, Z,       Z      ],
                   [Z,    Z,    -I / Ts, Z      ],
                   [Z,    Z,    Z,       -I / Ts]])

    # ── Biases: resting point x_rest that each pop relaxes toward ────────────
    #   Integrator pops: b_ni, the uniform resting rate (same for every pop → no
    #   rotation), shifted by ±setpoint/2 so the NET leaks toward the set point.
    #   Slide pops: 0 — they are silent at rest.
    b_ni   = brain_params.b_ni * jnp.ones(3)
    x_rest = jnp.concatenate([b_ni + setpoint_c / 2.0, b_ni - setpoint_c / 2.0,
                              jnp.zeros(3), jnp.zeros(3)])

    # ── Input matrix B (12×6) ────────────────────────────────────────────────
    #   cols: u_vel⁺, u_vel⁻.  The velocity command arrives as its two rectified
    #   halves: the integrator pops see (u⁺ − u⁻)/2 = u_vel/2 push-pull, and each
    #   slide pop low-passes its own half (so it stays ≥ 0 with no bias).
    B = jnp.block([[ I / 2,  -I / 2],
                   [-I / 2,   I / 2],
                   [ I / Ts,  Z    ],
                   [ Z,       I / Ts]])

    # ── Output C (3×12), D (3×6): pulse-slide-step, then canal → cardinal ────
    #   With slide = slide_L − slide_R, u_vel = u⁺ − u⁻, and slide' = (u_vel − slide)/Ts
    #   substituted into u_p:
    #   C = [ I  −I  K_slide  −K_slide ],   K_slide = diag(c_slide − c_accel/Ts)
    #   D = [ K_vel  −K_vel ],              K_vel   = diag(c_accel/Ts)
    K_slide = jnp.diag(c_slide - c_accel / Ts)
    K_vel   = jnp.diag(c_accel / Ts)
    C = jnp.block([[I, -I, K_slide, -K_slide]])
    D = jnp.block([[K_vel, -K_vel]])

    # ── State-space step:  dx = A·(x − x_rest) + B·u,   u_p = C·x + D·u ───────
    x   = jnp.concatenate([activations.L, activations.R,
                           activations.slide_L, activations.slide_R])
    u   = jnp.concatenate([jnp.maximum(u_vel_c, 0.0), jnp.maximum(-u_vel_c, 0.0)])
    dx  = A @ (x - x_rest) + B @ u
    u_p = CANAL2CARDINAL @ (C @ x + D @ u)
    dL_raw, dR_raw, dslide_L, dslide_R = dx[0:3], dx[3:6], dx[6:9], dx[9:12]

    # ── Anti-windup on net — clipped in CARDINAL (the orbital limit is a physical
    # eye-position bound; clipping canal-plane components would mis-limit vertical/
    # torsional gaze).  Reconstruct the cardinal net + net-derivative, clip per
    # cardinal axis, rotate the clipped derivative back to canal.  A nonlinear
    # correction to dL, dR only — the output u_p does not depend on it.
    x_net   = activations.L - activations.R   # canal net position
    dx_net  = dL_raw - dR_raw                 # canal net derivative before clipping
    dx_sum  = dL_raw + dR_raw                 # common-mode: unaffected by windup

    x_net_card  = CANAL2CARDINAL @ x_net
    dx_net_card = CANAL2CARDINAL @ dx_net
    dx_net_card = jnp.where(x_net_card >=  brain_params.orbital_limit,
                            jnp.minimum(dx_net_card, 0.0), dx_net_card)
    dx_net_card = jnp.where(x_net_card <= -brain_params.orbital_limit,
                            jnp.maximum(dx_net_card, 0.0), dx_net_card)
    dx_net      = CARDINAL2CANAL @ dx_net_card  # clipped net derivative, back to canal

    # Reconstruct individual derivatives from clipped net + unchanged sum
    dL = (dx_net + dx_sum) / 2.0
    dR = (dx_sum - dx_net) / 2.0

    return dL, dR, dslide_L, dslide_R, u_p


# ── Adaptation: null-point set point (slow, local statistical learning) ───────

def step_adaptation(activations, setpoint_c, brain_params):
    """Null-point adaptation — the NI's set point slowly learns where the eye is held.

        dnull/dt = (x_net − setpoint) / τ_ni_adapt,   setpoint = null + u_tonic

    With sustained u_tonic and no input the system has a 1-D family of equilibria
    along x_net = null + u_tonic. Starting from (0, 0) it settles at
        x_net = u_tonic·τ_ni_adapt/(τ_i + τ_ni_adapt),  null = −u_tonic·τ_i/(τ_i + τ_ni_adapt)
    on TC τ_eff = τ_i·τ_ni_adapt/(τ_i + τ_ni_adapt). So the null partially adapts to OCR —
    when OCR is later removed, the null stays negative briefly and drives a small
    post-OCR rebound, directionally consistent with reported post-tilt-removal drift.

    Args:
        activations:  ni.Activations  (L, R) rectified firing rates, (3,) each
        setpoint_c:   (3,)  current leak target = null + u_tonic (canal-plane)
        brain_params: BrainParams

    Returns:
        dnull: (3,)  null derivative (canal-plane)
    """
    x_net = activations.L - activations.R
    return (x_net - setpoint_c) / brain_params.tau_ni_adapt


def step(activations, weights, u_vel, brain_params, u_tonic=0.0):
    """Single ODE step: fast dynamics (step_dynamics) + null adaptation (step_adaptation).

    Activation-driven: bilateral pop firing rates come from `activations`
    (acts.ni), slide pops included; the null register comes from `weights` (weights.ni).

    Args:
        activations:  ni.Activations  (L, R, slide_L, slide_R) firing rates, each (3,)
        weights:      ni.Weights      (null,) register, (3,)
        u_vel:        (3,)  combined eye-velocity command (deg/s) — sign-flipped upstream
        brain_params: BrainParams
        u_tonic:      (3,)  tonic position-offset set-point (OCR + Listing's torsion).
                            Shifts the leak target: x_net leaks toward (null + u_tonic).
                            A saccade landing at the OCR position is therefore stable
                            (no drift back to 0). Not added to u_p directly — it flows
                            through the integrator, so x_ni already reflects the offset
                            and ec_pos stays consistent with the actual eye position.

    Returns:
        dstate: ni.State   state derivative (L, R, null, slide_L, slide_R)
        u_p:    (3,)       pulse-slide-step motor command to plant (cardinal)
    """
    # ── Inputs (cardinal → canal-plane basis) ────────────────────────────────
    # u_vel and u_tonic arrive CARDINAL (the brain_model merge is cardinal); rotate
    # into the canal-plane basis at entry.  Both outputs are rotated back to cardinal
    # where they leave: u_p at the end of step_dynamics() (→ final common pathway),
    # and the net position in decode_states() (→ saccade generator, T-VOR, Listing).
    u_vel_c   = CARDINAL2CANAL @ jnp.asarray(u_vel, dtype=jnp.float32)
    u_tonic_c = CARDINAL2CANAL @ (jnp.zeros(3, dtype=jnp.float32) + u_tonic)

    # Set point of the net position: the adapted null shifted by the tonic offset.
    # u_tonic moves the target without altering the stored null. Without quick-phase
    # resets, x_net only reaches a fraction τ_ni_adapt / (τ_i + τ_ni_adapt) ≈ 0.44 of
    # u_tonic at SS — saccades and quick phases drive the rest of the way (visible in
    # the OCR cascade bench).
    setpoint_c = weights.null + u_tonic_c

    dL, dR, dslide_L, dslide_R, u_p = step_dynamics(activations, setpoint_c, u_vel_c,
                                                    brain_params)
    dnull = step_adaptation(activations, setpoint_c, brain_params)

    return State(L=dL, R=dR, null=dnull, slide_L=dslide_L, slide_R=dslide_R), u_p


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 15  # x_L(3) + x_R(3) + x_null(3) + slide_L(3) + slide_R(3)
N_INPUTS  = 3
N_OUTPUTS = 3
