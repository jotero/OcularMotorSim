"""Self-motion estimation — single SSM module: VS + Gravity + Heading.

Bayesian-style observer of head motion + gravity (Laurens & Angelaki 2011, 2017).
Three internal subsystems share state and are stepped in a fixed order; the
public surface is one ``step()`` function plus index constants for the
combined state and bundled-input layouts.

Internal subsystems (kept as private helpers, no external import path):

  Velocity storage (VS) — bilateral push-pull VN populations + null adaptation
                         angular velocity estimate ω̂ (deg/s) → NI
  Gravity estimator (GE) — Laurens cross-product transport + linear-acc estimate
                          gravity vector ĝ (m/s²) and a_lin (m/s², head frame)
  Heading estimator (HE) — leaky integration of a_lin + visual translational flow
                          head linear velocity v_lin (m/s, head frame) → T-VOR

State layout (relative to x_self_motion, total 21 states):

    x_vs   (9,)   _IDX_VS    [0:9]   — VS L pop + R pop + null
    x_grav (9,)   _IDX_GRAV  [9:18]  — gravity estimate, a_lin, rf state
    x_head (3,)   _IDX_HEAD  [18:21] — head linear velocity v_lin

Inputs to step() (explicit named arguments):

    canal           (6,)   canal afferents (deg/s)
    scene_slip      (3,)   delayed retinal scene slip (deg/s, eye frame)
    gia             (3,)   otolith GIA (m/s², head frame)
    scene_lin_vel   (3,)   delayed scene translational flow (m/s)
    scene_visible   scalar delayed cyclopean scene presence gate
    ec_d_scene      (3,)   delayed efference copy matched to scene_angular_vel cascade

Outputs from step():

    dx_self_motion (21,)   ODE derivative of the combined state
    w_est          (3,)    angular velocity estimate (deg/s)
    g_est          (3,)    gravity estimate (m/s², head frame)
    v_lin          (3,)    head linear velocity (m/s, head frame)
    a_lin_est      (3,)    linear-acc estimate (m/s², head frame) — for T-VOR direct

References:
    Raphan, Matsuo & Cohen (1979) — Velocity storage in vestibular nystagmus.
    Cannon & Robinson (1985)      — NI leak / GEN model.
    Laurens & Angelaki (2011) JNS — VS-GE coupling; rf rotational feedback.
    Laurens, Meng & Angelaki (2013) PLoS Comp Bio — translation prior τ_a_lin.
    Paige & Tomko (1991) JN       — empirical T-VOR dark gain.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models.sensory_model import PINV_SENS, N_CANALS
from oculomotor.models.sensory_models.retina        import ypr_to_xyz, xyz_to_ypr


# ─────────────────────────────────────────────────────────────────────────────
# Module-wide constants
# ─────────────────────────────────────────────────────────────────────────────

G0 = 9.81   # standard gravity (m/s²)

# Default initial GE substate: ĝ upright, â=0, rf=0
GRAV_X0 = jnp.array([0.0, G0, 0.0,  0.0, 0.0, 0.0,  0.0, 0.0, 0.0])

# VS-internal scaling — population-health normaliser
_B_NOMINAL = 100.0   # healthy resting bias (deg/s)

# ── Canal-plane coordinate frame (VN native basis) ───────────────────────────
# The VN populations carry angular velocity in CANAL-PLANE coordinates
# [ω_H, ω_LARP, ω_RALP] — the three coplanar push-pull pairs — not cardinal
# [yaw,pitch,roll]. This maps the anatomy: each population is one coplanar pair,
# ipsilateral excitation minus contralateral commissural inhibition (Shimazu-Precht
# Type-I).  H = horizontal (LHC/RHC = left/right); LARP = LAC/RPC plane (up-down +
# CCW); RALP = RAC/LPC plane (up-down + CW).  See project_coordinate_rectification.
#
# PINV_CANAL reconstructs the three canal-plane velocities straight from the 6
# afferents — each row is one clean coplanar-pair push-pull:
#     ω_H = (RHC−LHC)/2,   ω_LARP = (LAC−RPC)/2,   ω_RALP = (RAC−LPC)/2.
# CANAL2CARDINAL recombines canal-plane → cardinal (the 45° pitch/roll mix). It is
# the ONE reconstruction applied at each cardinal SINK — for now the VS output
# (w_est feeds the still-cardinal NI merge + GE); as the NI/FCP go canal it slides
# out to the FCP. Everything upstream of a sink stays canal-native.
_S = 2 ** -0.5
CANAL2CARDINAL = jnp.array([[1.,  0.,  0.],
                            [0.,  _S,  _S],
                            [0., -_S,  _S]])   # v_cardinal = CANAL2CARDINAL @ v_canal
CARDINAL2CANAL = CANAL2CARDINAL.T              # orthonormal → inverse = transpose
PINV_CANAL     = CARDINAL2CANAL @ PINV_SENS    # (3×6) afferents → [ω_H, ω_LARP, ω_RALP]

# GE-internal — fast first-order tracking of rf for the VS↔GE algebraic-loop break
_TAU_RF_STATE = 0.005   # 5 ms

# Default initial state for the whole subsystem (used by brain_model.make_x0)
X0 = jnp.concatenate([
    jnp.zeros(9),     # VS — populations and null start at 0; brain_model overrides
                      # with b_vs equilibrium for both L and R pops.
    GRAV_X0,          # GE
    jnp.zeros(3),     # HE — v_lin starts at 0
])


# ─────────────────────────────────────────────────────────────────────────────
# State + input layout
# ─────────────────────────────────────────────────────────────────────────────

# VS sub-state slices (within x_vs, which is the first 9 of x_self_motion)
_VS_IDX_A    = slice(0, 3)   # population A (preferred-direction)
_VS_IDX_B    = slice(3, 6)   # population B (opposite preferred)
_VS_IDX_POP  = slice(0, 6)   # both populations
_VS_IDX_NULL = slice(6, 9)   # null adaptation

# GE sub-state slices (within x_grav, which is x_self_motion[9:18])
_GE_IDX_G  = slice(0, 3)   # g_est  — gravity estimate
_GE_IDX_A  = slice(3, 6)   # a_lin  — linear-acc estimate
_GE_IDX_RF = slice(6, 9)   # rf     — rotational feedback state

# ── State + registries ────────────────────────────────────────────────────────
# VS pops are called A/B internally (population A=preferred, B=opposite), but
# exposed externally as L/R to match the codebase-wide convention
# (model L pop ≡ A; net = L − R = A − B).

class State(NamedTuple):
    """Self-motion state: VS pops + null + GE observer + heading."""
    vs_L:    jnp.ndarray   # (3,)  VN pop A, canal-plane [H, LARP, RALP]
    vs_R:    jnp.ndarray   # (3,)  VN pop B, canal-plane [H, LARP, RALP]
    vs_null: jnp.ndarray   # (3,)  VS adaptation register (slow null drift), canal-plane
    g_est:   jnp.ndarray   # (3,)  gravity estimate (head frame)  [VN/cb gravity cells]
    a_lin:   jnp.ndarray   # (3,)  linear-accel estimate          [VN linear-accel cells]
    rf:      jnp.ndarray   # (3,)  rotational feedback            [Laurens observer]
    v_lin:   jnp.ndarray   # (3,)  head linear velocity estimate  [MST / heading cells]


class Activations(NamedTuple):
    """Self-motion firing rates (VS pops + GE/HE observer)."""
    vs_R:  jnp.ndarray
    vs_L:  jnp.ndarray
    g_est: jnp.ndarray
    a_lin: jnp.ndarray
    rf:    jnp.ndarray
    v_lin: jnp.ndarray


class Decoded(NamedTuple):
    """VS net head angular velocity readout."""
    vs_net: jnp.ndarray   # (3,) signed = vs_L − vs_R   head ang vel estimate (deg/s)


class Weights(NamedTuple):
    """VS adaptation register (long-term: learned weight)."""
    vs_null: jnp.ndarray   # (3,) signed slow-null


def rest_state():
    """Initial state — VS pops at b_vs equilibrium, GE at gravity vertical."""
    return State(
        vs_L    = jnp.zeros(3),
        vs_R    = jnp.zeros(3),
        vs_null = jnp.zeros(3),
        g_est   = X0[_GE_IDX_G] if 'X0' in globals() else jnp.array([G0, 0.0, 0.0]),
        a_lin   = jnp.zeros(3),
        rf      = jnp.zeros(3),
        v_lin   = jnp.zeros(3),
    )


def read_activations(state):
    """Project self-motion State → Activations (firing rates only)."""
    return Activations(
        vs_R  = state.vs_R,
        vs_L  = state.vs_L,
        g_est = state.g_est,
        a_lin = state.a_lin,
        rf    = state.rf,
        v_lin = state.v_lin,
    )


def decode_states(acts):
    # pops are canal-plane; recombine the net to cardinal (the registry SINK) so
    # every downstream reader sees head-frame [yaw,pitch,roll] as before.
    return Decoded(vs_net=CANAL2CARDINAL @ (acts.vs_L - acts.vs_R))


def read_weights(state):
    # vs_null is stored canal-plane; expose it cardinal so it pairs with the
    # cardinal vs_net for any consumer (e.g. cerebellum leak cancellation).
    return Weights(vs_null=CANAL2CARDINAL @ state.vs_null)

N_OUTPUTS = 3   # primary output for SSM convention is w_est; auxiliaries via tuple


# ─────────────────────────────────────────────────────────────────────────────
# Velocity Storage — bilateral push-pull (Raphan, Matsuo & Cohen 1979)
# ─────────────────────────────────────────────────────────────────────────────

def _vs_step(x_vs, canal, slip, fl_vs_drive, nu_drive, brain_params):
    """VS internal step.

    Two populations (A/B) with opposite preferred directions; net = x_A − x_B.
    Push-pull on canal and visual inputs; per-axis tau (yaw/pitch/roll fractions);
    null adaptation extends effective TC for sustained stimuli.

    Cerebellar inputs (Cohen, Raphan, Wearne for NU; Cannon-Robinson for FL):
        nu_drive    — nodulus+uvula gravity-axis dumping (replaces the old
                       in-place K_gd · rf computation).
        fl_vs_drive — floccular leak cancellation, extends effective tau_vs.

    Args:
        x_vs:        (9,) [x_A | x_B | x_null] (deg/s)
        canal:       (6,) canal afferents (deg/s)
        slip:        (3,) scene retinal slip [yaw,pitch,roll] (deg/s)
        fl_vs_drive: (3,) NET-level FL feedback (yaw/pitch/roll); applied
                          push-pull to bilateral pops to cancel leak.
        nu_drive:    (3,) NET-level NU dumping signal (yaw/pitch/roll);
                          applied push-pull (replaces old K_gd · rf).
        brain_params: BrainParams

    Returns:
        dx_vs: (9,) state derivative
        w_est: (3,) angular velocity estimate [yaw,pitch,roll] (deg/s)
    """
    x_null = x_vs[_VS_IDX_NULL]   # canal-plane [ω_H, ω_LARP, ω_RALP]
    x_pop  = x_vs[_VS_IDX_POP]    # (6,) two canal-plane populations [pop_A | pop_B]

    # Rotate the CARDINAL inputs into the canal-plane basis.  Canal afferents are
    # already labyrinth-native (6-vector); slip is retinal (cardinal) and the
    # cerebellar feedbacks (nu/fl) arrive cardinal for now, so each gets one
    # CARDINAL2CANAL.  (The optokinetic drive is anatomically organised in canal
    # planes via NOT/AOS, so rotating slip here is fidelity, not a hack.)
    # Canal saturation is applied sensor-side, so afferents arrive pre-clipped.
    slip_c = CARDINAL2CANAL @ slip
    nu_c   = CARDINAL2CANAL @ nu_drive
    fl_c   = CARDINAL2CANAL @ fl_vs_drive
    u_lin  = jnp.concatenate([canal, slip_c])      # (9,) canal afferents + canal-plane slip

    # Set point: population-uniform resting bias b_vs (a common mode that cancels in
    # the net A−B, hence basis-free) plus a slow null-adapted push-pull shift.
    SP    = brain_params.b_vs + jnp.concatenate([x_null / 2.0, -x_null / 2.0])
    g_pop = brain_params.b_vs / _B_NOMINAL   # (6,) population health: 1=healthy, 0=silent

    # Leak A: per canal-plane channel — horizontal TC = tau_vs, both vertical
    # planes share tau_vs·vert_frac (LARP/RALP mirror-symmetric).  Canal-native, so
    # it is a plain diagonal — no rotation needed.
    tau3 = jnp.array([brain_params.tau_vs,
                      brain_params.tau_vs * brain_params.tau_vs_vert_frac,
                      brain_params.tau_vs * brain_params.tau_vs_vert_frac])
    A1 = -jnp.diag(1.0 / tau3)      # (3×3) diag([1/τ_H, 1/τ_vert, 1/τ_vert])
    A  = jnp.block([[A1, jnp.zeros((3, 3))],
                    [jnp.zeros((3, 3)), A1]])

    # B (6×9): canal drive via PINV_CANAL (clean coplanar-pair rows), visual push-pull.
    B_top = jnp.concatenate([ g_pop[:3, None] * brain_params.K_vs * PINV_CANAL,
                             -brain_params.K_vis * jnp.eye(3)], axis=1)
    B_bot = jnp.concatenate([-g_pop[3:, None] * brain_params.K_vs * PINV_CANAL,
                              brain_params.K_vis * jnp.eye(3)], axis=1)
    B = jnp.concatenate([B_top, B_bot], axis=0)

    # C (3×6): canal-plane net = pop_A − pop_B
    C = jnp.concatenate([jnp.eye(3), -jnp.eye(3)], axis=1)

    # D (3×9): canal + visual feedthrough on the (canal-plane) net output
    D = jnp.concatenate([brain_params.g_vor * PINV_CANAL,
                        -brain_params.g_vis * jnp.eye(3)], axis=1)

    # Cerebellar inputs to VS (canal-plane, push-pull split of the net-level signals):
    #   - nu_drive: nodulus gravity-axis dumping.
    #   - fl_vs_drive: floccular leak cancellation (extends effective tau_vs).
    nu6    = jnp.concatenate([ nu_c, -nu_c])
    fl_vs6 = 0.5 * jnp.concatenate([ fl_c, -fl_c])

    dx_pop      = A @ (x_pop - SP) + B @ u_lin - nu6 + fl_vs6
    w_est_canal = C @ x_pop + D @ u_lin              # canal-plane net [ω_H, ω_LARP, ω_RALP]
    dx_null     = (w_est_canal - x_null) / brain_params.tau_vs_adapt

    # SINK: recombine canal-plane → cardinal for the (still-cardinal) NI merge + GE.
    w_est = CANAL2CARDINAL @ w_est_canal
    return jnp.concatenate([dx_pop, dx_null]), w_est


# ─────────────────────────────────────────────────────────────────────────────
# Gravity Estimator — Laurens & Angelaki cross-product dynamics
# ─────────────────────────────────────────────────────────────────────────────

def _ge_step(x_grav, w_est, gia, brain_params):
    """GE internal step.

    Tracks gravity ĝ (slow, anchored to GIA) and translation â (transient,
    decays toward 0 in absence of evidence). VS angular velocity transports ĝ
    in the head frame. rf is the Laurens rotational feedback fed BACK into VS
    next step.

    Args:
        x_grav: (9,) [ĝ | â | rf] (head frame, m/s² for first six, deg/s for rf)
        w_est:  (3,) VS net angular velocity [yaw,pitch,roll] (deg/s)
        gia:    (3,) otolith GIA (m/s², head frame)
        brain_params: BrainParams

    Returns:
        dx_grav: (9,) state derivative
        g_est:   (3,) gravity estimate (passed through from state)
    """
    g_est    = x_grav[_GE_IDX_G]
    a_lin    = x_grav[_GE_IDX_A]
    rf_state = x_grav[_GE_IDX_RF]

    # Residual: GIA minus the brain's two estimates. The two states (ĝ, â)
    # compete for it via their own gains (K_grav, K_lin). Once â captures
    # the translation component, residual → 0 and ĝ stops drifting toward
    # transient acceleration. Translation prior (decay on â) keeps it from
    # locking on a sustained DC.
    residual = gia - g_est - a_lin

    # Transport: rotate ĝ with VS angular velocity (VN → uvula/nodulus pathway).
    w_rad_xyz = jnp.radians(ypr_to_xyz(w_est))
    transport = -jnp.cross(w_rad_xyz, g_est)

    # Kalman-derived state-dependent gain modulation.
    # The bilinear gravity-transport term [ω]_× couples (ω, g) in the EKF Riccati
    # equation; the resulting Kalman gain has the structure:
    #   K_grav,eff = K_grav · √(1 + ρ)         (boost when rotation ⊥ gravity)
    #   K_lin,eff  = K_lin  / √(1 + ρ)         (suppress same regime)
    # where ρ = |ω̂ × ĝ_hat| / w_canal_gate is the "rotation-perpendicular-to-
    # gravity" Bayes factor. Rotation parallel to gravity (e.g. upright yaw)
    # gives ρ → 0 → no gating, since parallel rotations don't change head-frame
    # gravity and produce no spurious otolith residual to misattribute.
    g_hat       = g_est / (jnp.linalg.norm(g_est) + 1e-9)
    w_xyz       = ypr_to_xyz(w_est)
    w_perp_g    = jnp.linalg.norm(jnp.cross(w_xyz, g_hat))   # deg/s
    rho         = w_perp_g / brain_params.w_canal_gate
    gate_factor = jnp.sqrt(1.0 + rho)
    K_grav_eff  = brain_params.K_grav * gate_factor
    K_lin_eff   = brain_params.K_lin  / gate_factor

    # Gravity correction: pulled toward residual with state-modulated K_grav.
    dg = transport + K_grav_eff * residual

    # Linear acceleration: tracks residual (gated), decays toward 0 on TC τ_a_lin
    # (deterministic stand-in for L&A's translation-duration prior).
    da = K_lin_eff * residual - a_lin / brain_params.tau_a_lin

    # Rotational feedback (Laurens 2011): GIA × G_down / G0². Zero at SS;
    # active when ĝ lags GIA. Stored in state (1-step delayed) so brain_model
    # can read it next step without an algebraic loop.
    rf_new = xyz_to_ypr(jnp.cross(gia, -g_est)) / (G0 ** 2)
    drf    = (rf_new - rf_state) / _TAU_RF_STATE

    return jnp.concatenate([dg, da, drf]), g_est


# ─────────────────────────────────────────────────────────────────────────────
# Heading Estimator — leaky integration of a_lin + visual flow
# ─────────────────────────────────────────────────────────────────────────────

def _he_step(x_head, a_lin, scene_lin_vel, scene_visible, brain_params):
    """HE internal step.

    Vestibular path: leaky integral of a_lin (the GE's translation-attributed
    component of GIA, NOT raw gia − g_est — avoids gravity-mismatch drift).
    Visual path: scene flow pulls v_lin toward −scene_lin_vel; gated by
    scene visibility (zeroed in dark).

    Args:
        x_head:        (3,) v_lin estimate (m/s, head frame)
        a_lin:         (3,) GE's linear-acc estimate (m/s², head frame)
        scene_lin_vel: (3,) cyclopean scene flow (m/s, head frame)
        scene_visible: scalar in [0, 1] — visual fusion gate
        brain_params:  BrainParams (reads tau_head, K_he_vis)

    Returns:
        dx_head: (3,) dv_lin/dt (m/s²)
        v_lin:   (3,) v_lin (passed through from state)
    """
    v_lin    = x_head
    v_visual = -scene_lin_vel
    K_vis    = brain_params.K_he_vis * scene_visible

    dx = a_lin - v_lin / brain_params.tau_head + K_vis * (v_visual - v_lin)
    return dx, v_lin


# ─────────────────────────────────────────────────────────────────────────────
# Public step() — orchestrates VS → GE → HE
# ─────────────────────────────────────────────────────────────────────────────

def step(activations, weights,
         # Vestibular afferents
         canal, gia,
         # Visual scene PE (sat·slip + cerebellum's gated EC correction; assembled in brain_model)
         slip_pe_for_vs, scene_lin_pe, scene_visible,
         # Cerebellar leak-cancellation / gravity dumping (from acts.cb)
         fl_vs_drive, nu_drive,
         brain_params):
    """Single ODE step for the unified self-motion observer.

    Activation-driven: cross-projections and recurrence read firing rates from
    `activations` (supplied by the caller via the brain-wide registry).
    Setpoint-like registers (vs_null) are read from `weights`.  Most current
    projections are identity copies of state, but read_activations is the
    formal hook for L/R splits + rectification when those land.

    Internal sequencing (matches Laurens & Angelaki 2017):
      0. Prediction error on the scene path, gated by cerebellar saccadic
         suppression:
             PE_scene_raw = scene_slip + scene_visible · ec_d_scene
             PE_scene     = saccadic_suppression_scene · PE_scene_raw
         The gate closes during high-speed self-motion (when both slip and
         EC have saturated), down-weighting retinal evidence that would
         otherwise drive VS in the wrong direction.
      1. VS uses rf_state from the (1-step delayed) ODE state — breaks the
         VS↔GE algebraic loop with negligible lag (τ_rf_state ≈ 5 ms).
      2. GE then runs with the freshly-computed w_est from VS.
      3. HE consumes a_lin from GE state (1-step delayed) for its own
         leaky integration toward v_lin.

    Args:
        activations:    sm.Activations  vs_R/L, g_est, a_lin, rf, v_lin
        weights:        sm.Weights      vs_null
        canal:          (6,)   canal afferents (deg/s)
        scene_slip:     (3,)   RAW delayed scene slip (eye frame, deg/s)
        gia:            (3,)   otolith GIA (m/s², head frame)
        scene_lin_vel:  (3,)   RAW delayed scene linear velocity (m/s)
        scene_visible:  scalar delayed cyclopean scene presence gate ∈ [0,1]
        ec_d_scene:     (3,)   delayed EC matched to scene_angular_vel cascade
        brain_params:   BrainParams — VS/GE/HE params

    Returns:
        dstate         : sm.State  state derivative
        w_est          : (3,)  angular velocity estimate (deg/s)
        g_est          : (3,)  gravity estimate (m/s², head frame)
        v_lin          : (3,)  head linear velocity (m/s, head frame)
        a_lin_est      : (3,)  linear-acc estimate (m/s², head frame) — for T-VOR direct
    """
    # Repack the registry views into the flat sub-blocks the internal helpers
    # expect.  vs_null comes from `weights` (it's a setpoint register, not a
    # firing rate); the rest come from `activations`.
    x_vs   = jnp.concatenate([activations.vs_L, activations.vs_R, weights.vs_null])
    x_grav = jnp.concatenate([activations.g_est, activations.a_lin, activations.rf])
    x_head = activations.v_lin

    # a_lin used here is 1-step delayed via the ODE state — the activations
    # registry was built at the start of the brain step.  rf_state was the
    # input to the old K_gd · rf computation; that has now moved to the
    # cerebellum (acts.cb.nu_drive), so rf is no longer used here.
    a_lin_est = activations.a_lin

    # 0. Scene PE is assembled in brain_model as:
    #        slip_pe_for_vs = K_vor_direct · sat · cyc.scene_angular_vel
    #                       + K_cereb_okr  · fl_okr_drive
    #    where fl_okr_drive = sat · scene_visible · ec_scene is the cerebellum's
    #    pre-gated EC correction.  Saccadic suppression acts on the raw retinal
    #    input directly upstream of this step.  scene_lin_pe is the matching
    #    gated linear-velocity signal for the heading estimator.

    # 1. VS — angular velocity estimate
    dx_vs, w_est = _vs_step(x_vs, canal, slip_pe_for_vs, fl_vs_drive, nu_drive, brain_params)

    # 2. GE — gravity + linear-acc estimates (rf updated for next step)
    dx_grav, g_est = _ge_step(x_grav, w_est, gia, brain_params)

    # 3. HE — head linear velocity (consumes the prior step's a_lin to avoid
    #         needing the freshly-computed â here)
    dx_head, v_lin = _he_step(x_head, a_lin_est, scene_lin_pe, scene_visible, brain_params)

    dstate = State(
        vs_L    = dx_vs[_VS_IDX_A],
        vs_R    = dx_vs[_VS_IDX_B],
        vs_null = dx_vs[_VS_IDX_NULL],
        g_est   = dx_grav[_GE_IDX_G],
        a_lin   = dx_grav[_GE_IDX_A],
        rf      = dx_grav[_GE_IDX_RF],
        v_lin   = dx_head,
    )
    return dstate, w_est, g_est, v_lin, a_lin_est


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES  = 9 + 9 + 3   # 21
_IDX_VS   = slice(0, 9)
_IDX_GRAV = slice(9, 18)
_IDX_HEAD = slice(18, 21)


def from_array(x_self_motion):
    """(21,) flat array → sm.State."""
    x_vs   = x_self_motion[_IDX_VS]
    x_grav = x_self_motion[_IDX_GRAV]
    return State(
        vs_L    = x_vs[_VS_IDX_A],
        vs_R    = x_vs[_VS_IDX_B],
        vs_null = x_vs[_VS_IDX_NULL],
        g_est   = x_grav[_GE_IDX_G],
        a_lin   = x_grav[_GE_IDX_A],
        rf      = x_grav[_GE_IDX_RF],
        v_lin   = x_self_motion[_IDX_HEAD],
    )


def to_array(state):
    """sm.State → (21,) flat array."""
    return jnp.concatenate([
        state.vs_L, state.vs_R, state.vs_null,
        state.g_est, state.a_lin, state.rf,
        state.v_lin,
    ])


__all__ = [
    "step", "State", "Activations", "Decoded", "Weights",
    "rest_state", "read_activations", "decode_states", "read_weights",
    "from_array", "to_array",
    "X0", "GRAV_X0", "G0",
    "N_STATES", "N_INPUTS", "N_OUTPUTS",
    "_IDX_VS", "_IDX_GRAV", "_IDX_HEAD",
    "_IDX_INPUT_CANAL", "_IDX_INPUT_SLIP", "_IDX_INPUT_GIA",
    "_IDX_INPUT_SCENE_LIN_VEL", "_IDX_INPUT_SCENE_VISIBLE",
]
