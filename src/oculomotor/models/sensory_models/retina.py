"""Retinal geometry + per-eye visual delay cascade.

Two responsibilities:

  1. Geometry — convert Cartesian target position to angular gaze error,
     compute retinal velocity of the tracked target (pursuit drive), and apply
     the visual-field gate (eccentricity limit of the retina).  See
     ``world_to_retina``.

  2. Visual delay — a PER-EYE gamma-distributed delay implemented as a cascade
     of N first-order LP stages, approximating a pure transport delay of
     ``tau_vis_sharp`` seconds (the Pugh & Lamb 1993 photo-transduction rise
     shape).  Each eye runs its OWN cascade and knows nothing about the other
     eye — binocular fusion happens AFTER the delay, downstream in
     ``perception_cyclopean``.  There is no cyclopean cascade in this module.

     With N = ``_N_STAGES_OTHER`` (6) stages the cascade output is the last
     ``n_axes`` elements of each buffer (see ``read_outputs``).

     Per-eye signals delayed (``retina.State`` → ``RetinaOut``), all EYE FRAME:
         scene_angular_vel (3)  rotational optic flow              → OKR / VS
         scene_linear_vel  (3)  translational optic flow (parallax) → looming / T-VOR
         target_pos        (3)  target direction [yaw, pitch, 0]   → saccade error
         target_vel        (3)  target retinal velocity            → pursuit
         scene_visible     (1)  delay(scene_present)
         target_visible    (1)  delay(target_present × target_in_vf)
         defocus           (1)  delay(per-eye defocus, D)          → accommodation
     Plus a 1-pole afferent luminance register (NOT a sharp cascade):
         luminance         (1)  low-passed retinal illumination    → pupil light reflex

     Binocular constructions — target_disparity (vergence) and the per-eye
     scene-flow differential (heading / vergence) — are NOT produced here; they
     are assembled in ``perception_cyclopean`` from the two eyes' delayed
     target_pos / scene-flow signals.

Implicit depth-map assumption
-----------------------------
The very fact that this retina extracts both *rotational* (scene_angular_vel) and
*translational* (scene_linear_vel) optic flow as separable quantities is itself a
strong assumption: it requires the visual system to have already solved the depth
problem.  In a depthless world you can only infer rigid-body angular flow; you
cannot decompose head-translation parallax from rotation without depth structure
(the heading direction needs the focus-of-expansion of the depth-aware flow field).

We side-step this by assuming the brain has a depth map (e.g., from binocular
disparity, motion parallax, accommodation cues) and can therefore expose:
    - clean angular flow (scene_angular_vel) as if from rigid rotation
    - clean translational flow (scene_linear_vel) as the head-translation parallax
    - a per-eye flow differential (formed downstream from the two eyes) as a depth-rate cue

In practice for a depthless or uniform scene that per-eye differential degenerates
to 0, which the brain correctly interprets as "no depth-rate change" → constrains
heading-z and vergence-rate estimates.  This is the right zero-point behavior;
in a depth-structured scene the same signal would carry rich heading information.
"""

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix


# ── Cascade parameters ──────────────────────────────────────────────────────────
#
# Two-tier transmission model, split across modules:
#   - SHARP cascade (THIS module): a gamma cascade modelling photo-transduction +
#     axonal/synaptic transport delay (Pugh & Lamb 1993, Dunn & Rieke 2006).
#     Produces a near-pure transport delay of mean = tau_sharp. EVERY per-eye
#     channel — target_pos included — uses the same short cascade (see step()).
#   - SMOOTH LP (DOWNSTREAM): optional multi-stage smoothing after the sharp
#     cascade, modelling channel-specific neural integration (MT/MST motion
#     window, V1 stereo correspondence, accommodation circuit). It is applied in
#     perception_cyclopean (brain LP) and the cerebellum EC forward models via the
#     shared cascade_lp_step helper below — NOT in the retina's own step.

# Sharp-cascade stage count — the SAME for every per-eye channel (Pugh-Lamb
# photo-transduction: a 4-6 stage biochemical cascade gives the right rise shape).
# Channel-specific sluggishness belongs downstream in perception_cyclopean's brain
# LP, not here. N also sets how well a brief flash survives (spread ∝ 1/√N at fixed
# total delay): at N=6 a 20 ms flash reaches ~0.4 of full amplitude, a 10 ms ~0.2.
_N_STAGES_OTHER = 6

# ── Per-eye retina state layout ────────────────────────────────────────────────
# Each eye has its own sharp gamma cascade per signal (N stages × τ_retina/N).
# target_disparity is NOT here — it's a binocular construction computed in
# perception_cyclopean from delayed per-eye target_pos.
_RETINA_PER_EYE_LAYOUT = [
    ('scene_angular_vel', _N_STAGES_OTHER, 3),   # 18
    ('scene_linear_vel',  _N_STAGES_OTHER, 3),   # 18
    ('target_pos',        _N_STAGES_OTHER, 3),   # 18
    ('target_vel',        _N_STAGES_OTHER, 3),   # 18
    ('scene_visible',     _N_STAGES_OTHER, 1),   #  6
    ('target_visible',    _N_STAGES_OTHER, 1),   #  6
    ('defocus',           _N_STAGES_OTHER, 1),   #  6
]
N_STATES_PER_EYE = sum(N * n for _, N, n in _RETINA_PER_EYE_LAYOUT) + 1  # +1 luminance


# ── Per-eye State NamedTuple ──────────────────────────────────────────────────
from typing import NamedTuple


class State(NamedTuple):
    """Per-eye retina state — sharp gamma cascades for 7 signal channels,
    plus a 1-pole luminance afferent register for the pupillary light reflex.

    Each cascade buffer is N=_N_STAGES_OTHER stages × n_axes.  The cascade
    output (delayed signal) is the last n_axes elements.  `luminance` is a
    single-pole light-adaptation LP (τ_lum), NOT a sharp cascade — it lives here
    because retinal illumination is a per-eye retinal afferent (one optic nerve
    per eye), so a monocular afferent defect (RAPD) stays per-eye.
    """
    scene_angular_vel: jnp.ndarray   # (N*3,) cascade buffer
    scene_linear_vel:  jnp.ndarray   # (N*3,)
    target_pos:        jnp.ndarray   # (N*3,)
    target_vel:        jnp.ndarray   # (N*3,)
    scene_visible:     jnp.ndarray   # (N,)
    target_visible:    jnp.ndarray   # (N,)
    defocus:           jnp.ndarray   # (N,)
    luminance:         jnp.ndarray   # (1,) afferent luminance LP register (normalised, ~[0,1])


def rest_state():
    """Zero state (all cascade buffers zero; dark → zero afferent luminance)."""
    N = _N_STAGES_OTHER
    return State(
        scene_angular_vel = jnp.zeros(N * 3),
        scene_linear_vel  = jnp.zeros(N * 3),
        target_pos        = jnp.zeros(N * 3),
        target_vel        = jnp.zeros(N * 3),
        scene_visible     = jnp.zeros(N),
        target_visible    = jnp.zeros(N),
        defocus           = jnp.zeros(N),
        luminance         = jnp.zeros(1),
    )


# ── Sensor saturation ───────────────────────────────────────────────────────────

def velocity_saturation(v, v_sat, v_zero=None, v_offset=None):
    """Smooth velocity saturation: passes at low speed, gain ramps to zero at high speed.

    Models NOT/AOS / MT-MST firing-rate ceiling — neurons are band-pass tuned
    for speed and stop firing for implausibly fast retinal motion.

        |v| ≤ v_sat          → output = v           (gain = 1)
        v_sat < |v| < v_zero → output = v · gain    (cosine rolloff, 1 → 0)
        |v| ≥ v_zero         → output = 0           (gain = 0)

    Args:
        v:        (N,) velocity vector (deg/s); norm computed over the full vector
        v_sat:    saturation onset (deg/s) — gain is exactly 1 below this
        v_zero:   speed where gain reaches 0 (deg/s); default = 2 × v_sat
        v_offset: (N,) background velocity to shift the clip window (deg/s)

    Returns:
        Same shape as v, scaled by smooth gain ∈ [0, 1], plus v_offset if given.
    """
    if v_zero is None:
        v_zero = 2.0 * v_sat
    if v_offset is not None:
        v_rel = v - v_offset
    else:
        v_rel = v
    speed = jnp.linalg.norm(v_rel)
    t     = jnp.clip((speed - v_sat) / (v_zero - v_sat), 0.0, 1.0)
    gain  = 0.5 * (1.0 + jnp.cos(jnp.pi * t))
    result = v_rel * gain
    if v_offset is not None:
        result = result + v_offset
    return result


# ── Coordinate helpers ──────────────────────────────────────────────────────────
#
# World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).
#
# Angular vectors are stored as [yaw, pitch, roll] — NOT the same index order
# as xyz.  The mapping to xyz rotation axes used by rotation_matrix / cross():
#
#   ypr_to_xyz([yaw, pitch, roll]) = [−pitch,  yaw,  roll]
#   xyz_to_ypr([x,   y,     z  ]) = [y,       −x,   z   ]
#
#   yaw   (idx 0): rotation about +y  (left-hand: forward → right = rightward turn)
#   pitch (idx 1): rotation about −x  (left-hand: forward → up   = look up)
#   roll  (idx 2): rotation about +z  (left-hand: right → up)
#
# Always call ypr_to_xyz() before matrix ops; xyz_to_ypr() after.

def ypr_to_xyz(q):
    """[yaw, pitch, roll] (deg or deg/s) → xyz rotation-axis vector (same units)."""
    return jnp.array([-q[1], q[0], q[2]])


def xyz_to_ypr(v):
    """xyz rotation-axis vector → [yaw, pitch, roll] (same units). Inverse of ypr_to_xyz."""
    return jnp.array([v[1], -v[0], v[2]])


# ── Geometry ────────────────────────────────────────────────────────────────────

def world_to_retina(x_target, eye_offset_head, q_head, w_head, x_head, v_head,
                    q_eye, w_eye, w_scene, v_scene, v_target,
                    scene_present, target_present, vf_limit, k_vf):
    """Compute instantaneous retinal signals and per-eye visibility gates.

    Angular outputs (scene_angular_vel, target_vel) are in EYE coordinates
    [yaw, pitch, roll] (deg/s).  scene_linear_vel is in HEAD-frame xyz [right,
    up, forward] (m/s), with per-eye parallax (each eye is offset from the head
    centre, so head rotation moves the two eyes at different velocities — they
    therefore see different translational optic flow even when v_head is shared).
    Computing head-frame at retina (using actual q_eye implicitly via R_eye in
    R_gaze) is mathematically equivalent to delaying an eye-frame signal and
    derotating later by a delay-matched ec_pos, but is simpler to implement.
    target_pos is [yaw, pitch, 0] (deg).

    All head and eye geometry is handled here — sensory_model only passes
    anatomical offsets (eye_offset_head) and does not touch rotation matrices.

    World frame is LEFT-HANDED: x=right, y=up, z=forward  (x × y = −z).
    See module-level ypr_to_xyz / xyz_to_ypr for the [yaw,pitch,roll] ↔ xyz mapping.

    Inputs
    ------
    x_target:        target 3-D position in world frame (m)  [x=right, y=up, z=fwd]
    eye_offset_head: this eye's fixed position in head frame (m)
                     left=[-ipd/2,0,0]  right=[+ipd/2,0,0]
    q_head:          head rotation vector [yaw,pitch,roll]  (deg, world frame)
    w_head:          head angular velocity [yaw,pitch,roll]  (deg/s, world frame)
    x_head:          head linear position  [x,y,z]           (m, world frame)
    v_head:          head linear velocity  [x,y,z]           (m/s, world frame)
    q_eye:           eye rotation vector relative to head (deg, head frame)
    w_eye:           eye angular velocity relative to head (deg/s, head frame)
    w_scene:         scene angular velocity [yaw,pitch,roll] (deg/s, world frame)
    v_scene:         scene linear velocity  [x,y,z]          (m/s,   world frame)
    v_target:        target linear velocity [x,y,z] (m/s, world frame)
    scene_present:   scalar ∈ [0,1] — is the scene lit for this eye?
    target_present:  scalar ∈ [0,1] — is the target visible (not occluded) for this eye?
    vf_limit:        visual field half-width (deg)
    k_vf:            visual field gate sigmoid steepness (1/deg)

    Geometry
    --------
    R_head : world ← head  (from q_head rotation vector)
    R_eye  : head  ← eye   (from q_eye  rotation vector)
    R_gaze = R_head @ R_eye : world ← eye

    Target position in eye frame (exact, no small-angle approximation):
        eye_world  = x_head + R_head @ eye_offset_head   eye position in world frame
        p_from_eye = x_target − eye_world                target direction from this eye
        p_eye      = R_gaze.T @ p_hat                   target direction in eye frame
        target_pos = [arctan2(x,z), arctan2(y,√(x²+z²)), 0]  (deg, eye frame)

    Target angular velocity (computed from Cartesian position + velocity):
        w_target = xyz_to_ypr( cross(x_target, v_target) / |x_target|² )  [deg/s, world frame]

    Retinal velocities in eye frame:
        w_eye_world     = w_head + R_head @ w_eye             total eye angular velocity, world frame
        v_eye_world     = v_head + ω_head × (R_head @ eye_offset_head)
                          per-eye linear velocity in world frame; second term is the
                          parallax velocity from head rotation moving an eccentric eye.
        scene_angular_vel = R_gaze.T @ (w_scene − w_eye_world)  rotational optic flow, [yaw,pitch,roll] deg/s
        scene_linear_vel  = R_head.T @ (v_scene − v_eye_world)  translational optic flow, [x,y,z] m/s, HEAD frame, per-eye
        target_vel        = R_gaze.T @ (w_target − w_eye_world) target velocity on retina, [yaw,pitch,roll] deg/s

    Returns
    -------
        target_pos:       (3,)   target direction [yaw, pitch, 0] (deg)
        scene_angular_vel:(3,)   rotational optic flow [yaw,pitch,roll] (deg/s)
        scene_linear_vel: (3,)   translational optic flow [x,y,z] (m/s, head frame, per-eye)
        target_vel:       (3,)   target velocity on retina [yaw,pitch,roll] (deg/s)
        scene_vis:        scalar scene presence gate = scene_present ∈ [0,1]
        target_vis:       scalar combined target gate = target_present × target_in_vf ∈ [0,1]
    """
    # ── Rotation matrices ─────────────────────────────────────────────────────
    R_head   = rotation_matrix(ypr_to_xyz(q_head))   # world ← head
    R_eye    = rotation_matrix(ypr_to_xyz(q_eye))    # head  ← eye
    R_gaze_T = R_eye.T @ R_head.T                    # world → eye frame

    # ── Target position in eye frame ──────────────────────────────────────────
    eye_world  = x_head + R_head @ eye_offset_head           # eye position, world frame
    p_from_eye = x_target - eye_world                        # target from this eye, world frame
    p_hat      = p_from_eye / (jnp.linalg.norm(p_from_eye) + 1e-9)
    p_eye      = R_gaze_T @ p_hat                            # target direction, eye frame

    # Rotation-vector extraction (axis-angle), CONSISTENT with how q_eye and other
    # angular positions are treated throughout the code.  Fick-style arctan2 would
    # give numerically different yaw/pitch at non-primary positions (~1° at 30° gaze),
    # and rotation_matrix(ypr_to_xyz(target_pos)) applied to (0,0,1) wouldn't recover
    # the target direction.  With rotation-vector extraction this is exact.
    #
    # IMPLICIT LISTING'S LAW: this extraction inherently produces a Listing-compliant
    # target position.  The rotation that takes the primary direction (0,0,1) to the
    # target direction has its axis in the (x,y) plane — i.e., perpendicular to the
    # primary direction — which is exactly Listing's plane.  So target_pos[2] (torsion)
    # is 0 not because "target has only 2 DOF" but because Listing's law says the
    # eye rotation from primary stays in Listing's plane.  A non-Listing eye position
    # would require a non-(x,y) rotation axis, which this representation can't express.
    #
    # TODO: factor into a helper function `direction_to_rotvec_ypr(p_eye, primary)`
    # that takes a unit gaze direction and returns the YPR-style rotation vector.
    # Same logic should be reusable for any "look-at" geometry.
    #
    # Algorithm: rotation that takes (0,0,1) to p_eye is
    #   axis  = (0,0,1) × p_eye  = (−p_eye[1], p_eye[0], 0)
    #   angle = arctan2(|axis|, p_eye[2])     (robust)
    #   q_xyz = axis · (angle / |axis|)        (rotation vector, rad)
    # Then xyz_to_ypr → [yaw, pitch, roll=0] in degrees.
    axis_unscaled = jnp.array([-p_eye[1], p_eye[0], 0.0])
    r             = jnp.sqrt(p_eye[0]**2 + p_eye[1]**2)
    angle         = jnp.arctan2(r, p_eye[2])
    scale         = jnp.where(r > 1e-9, angle / (r + 1e-9), 1.0)
    q_xyz         = axis_unscaled * scale
    q_ypr         = jnp.degrees(xyz_to_ypr(q_xyz))    # [yaw, pitch, roll=0]
    target_pos    = jnp.array([q_ypr[0], q_ypr[1], 0.0])  # roll=0: implicit Listing's compliance

    # ── Retinal velocities in eye frame ───────────────────────────────────────
    # Angular velocities: convert ypr→xyz before rotation matrix ops, xyz→ypr after.
    # Without this, sustained rotation rotates the yaw axis into pitch/roll, causing
    # OKR to fight VOR.
    w_head_xyz   = ypr_to_xyz(w_head)
    w_eye_xyz    = ypr_to_xyz(w_eye)
    w_scene_xyz  = ypr_to_xyz(w_scene)
    target_dist  = jnp.sqrt(jnp.dot(x_target, x_target)) + 1e-9
    w_target     = jnp.degrees(xyz_to_ypr(jnp.cross(x_target, v_target)) / target_dist ** 2)
    w_target_xyz = ypr_to_xyz(w_target)
    w_eye_world  = w_head_xyz + R_head @ w_eye_xyz

    scene_angular_vel = xyz_to_ypr(R_gaze_T @ (w_scene_xyz - w_eye_world))  # [yaw,pitch,roll] deg/s
    # Per-eye linear velocity in world frame: v_head + ω_head × eye_offset_world.
    # The cross-product term is the parallax velocity — for a head rotating about its
    # own centre, an eccentric eye traces a small arc, and that motion contributes to
    # the optic flow at that eye even when the head is not translating.
    omega_head_rad = jnp.radians(w_head_xyz)
    v_eye_world    = v_head + jnp.cross(omega_head_rad, R_head @ eye_offset_head)
    scene_linear_vel  = R_head.T @ (v_scene - v_eye_world)                   # [x,y,z] m/s, HEAD frame, per-eye
    target_vel        = xyz_to_ypr(R_gaze_T @ (w_target_xyz - w_eye_world))        # [yaw,pitch,roll] deg/s
    target_vel        = target_vel.at[2].set(0.0)   # retina is 2D: target translates H/V only

    # ── Visibility gates ──────────────────────────────────────────────────────
    e_mag      = jnp.linalg.norm(target_pos) + 1e-9
    target_in_vf = 1.0 - jax.nn.sigmoid(k_vf * (e_mag - vf_limit))
    scene_vis  = jnp.asarray(scene_present, dtype=jnp.float32)
    target_vis = jnp.asarray(target_present, dtype=jnp.float32) * target_in_vf

    return target_pos, scene_angular_vel, scene_linear_vel, target_vel, scene_vis, target_vis


# ── Delay cascade ───────────────────────────────────────────────────────────────

def delay_cascade_step(x, u, tau_vis, N):
    """Advance a delay cascade of N stages for any signal shape.

    Works for both 3-D signals (N*3 states) and scalar signals (N states) with
    the same code.  A and B are built from N and the signal width at JAX trace
    time, so there is no runtime overhead vs pre-computed matrices.  Callers pass
    N explicitly (retina channels use _N_STAGES_OTHER; the brain LP block uses 1
    or _N_STAGES_BRAIN_POS).

    Args:
        x:       (N*n,)          cascade state  (n = signal width)
        u:       (n,) or scalar  input signal
        tau_vis: float           total cascade delay (s)
        N:       int             number of stages

    Returns:
        dx: (N*n,)  state derivative
    """
    u1d = jnp.atleast_1d(u)
    n   = u1d.shape[0]        # signal width — static at JAX trace time
    k   = N / tau_vis         # per-stage rate: stage tau = tau_vis / N, so the
                              # TOTAL delay is tau_vis for any N
    # Each stage is fed by the one before it: dx_i = k·(x_{i-1} − x_i), with the
    # input feeding stage 0. Written as a shift rather than the equivalent dense
    # A @ x: that matrix is (N·n)², so a 40-stage 3-axis channel would be 120×120
    # for what is structurally an O(N·n) operation. At N=40 the dense form costs
    # ~13x more per ODE evaluation than the whole cascade block does today.
    x2d  = x.reshape(N, n)
    prev = jnp.concatenate([u1d[None, :], x2d[:-1]], axis=0)
    return (k * (prev - x2d)).reshape(-1)


def cascade_lp_step(x_block, u, tau_sharp, tau_smooth, N, n_axes, N_lp):
    """Sharp gamma cascade + optional multi-stage smoothing.

    Block layout (concatenated cascade then LP):
        [sharp cascade (N · n_axes) | smoothing cascade (N_lp · n_axes)]

    The smoothing stage is itself a gamma cascade of N_lp poles with total mean
    delay tau_smooth (so per-stage TC = tau_smooth/N_lp). N_lp = 1 reduces to a
    single 1-pole LP (exponential rise/fall, long tail). N_lp ≥ 2 gives a
    sharper rolloff and a more concentrated impulse response — the right
    choice when residual signal after the input goes to zero must drain quickly
    (e.g. binocular disparity after one eye closes).

    Args:
        x_block:    block state ((N + N_lp)·n_axes,)
        u:          (n_axes,) or scalar  current input
        tau_sharp:  total sharp-cascade mean delay (s)
        tau_smooth: total smoothing-cascade mean delay (s); ignored if N_lp=0
        N:          sharp-cascade stage count
        n_axes:     1 (scalar) or 3 (3-vector)
        N_lp:       smoothing-cascade stage count; 0 = no smoothing,
                    1 = single 1-pole LP, ≥2 = multi-pole gamma smoothing

    Returns:
        dx_block:   state derivative, same shape as x_block
    """
    n_cascade = N * n_axes
    x_cascade = x_block[:n_cascade]
    dx_cascade = delay_cascade_step(x_cascade, u, tau_sharp, N=N)

    if N_lp == 0:
        return dx_cascade

    # Smoothing stage: feed the sharp-cascade output into another gamma cascade.
    cascade_out = x_cascade[n_cascade - n_axes : n_cascade]
    x_lp = x_block[n_cascade : n_cascade + N_lp * n_axes]
    dx_lp = delay_cascade_step(x_lp, cascade_out, tau_smooth, N=N_lp)
    return jnp.concatenate([dx_cascade, dx_lp])


# ── Per-eye retina step ─────────────────────────────────────────────────────────

from typing import NamedTuple


class RetinaOut(NamedTuple):
    """Per-eye delayed retinal signals after the sharp gamma cascade.

    All signals are in EYE FRAME at the delayed time (≈ τ_retina ago). Field
    order matches the cascade state-block layout for consistency. Read into
    perception_cyclopean for binocular fusion + brain LP smoothing.
    """
    scene_angular_vel: jnp.ndarray  # (3,) [yaw, pitch, roll] (deg/s) — gated by scene_visible + saturated
    scene_linear_vel:  jnp.ndarray  # (3,) [x, y, z] (m/s, head frame, per-eye) — gated by scene_visible
    target_pos:        jnp.ndarray  # (3,) [yaw, pitch, 0] (deg) — gated by target_visible
    target_vel:        jnp.ndarray  # (3,) [yaw, pitch, 0] (deg/s) — gated by target_visible + saturated
    scene_visible:     jnp.ndarray  # scalar — delayed scene_present
    target_visible:    jnp.ndarray  # scalar — delayed target_present × target_in_vf
    defocus:           jnp.ndarray  # scalar — delayed defocus (D)
    luminance:         jnp.ndarray  # scalar — afferent retinal luminance (~[0,1]) → pupil light reflex


def step(state,
         eye_offset_head, q_head, w_head, x_head, v_head,
         q_eye, w_eye,
         w_scene, v_scene, x_target, v_target,
         acc,
         scene_present, target_present,
         sensory_params):
    """Per-eye retina step: world_to_retina + sensor saturation + sharp cascade.

    Each eye runs an independent sharp gamma cascade (N = _N_STAGES_OTHER stages
    × τ_retina). Binocular fusion happens downstream in perception_cyclopean,
    so this function knows nothing about the other eye.

    Args:
        state:           retina.State   per-eye retina cascade state
        eye_offset_head: (3,) this eye's anatomical offset in head frame (m)
        q_head, w_head, x_head, v_head: head pose / velocity (world frame)
        q_eye, w_eye:    eye pose / velocity (head frame)
        w_scene, v_scene, x_target, v_target: world-frame scene / target stimulus
        acc:             scalar — this eye's accommodation (D): the lens
                         plant's accommodation already offset by any external lens
                         (simulator._apply_lens). refractive_error comes from
                         sensory_params.
        scene_present, target_present: scalar visibility flags (this eye). A strobe
                         is a flash train on these (sim.stimuli.strobe_train) —
                         there is no separate strobe gate.
        sensory_params:  SensoryParams — reads tau_vis_sharp, v_max_scene_vel,
                         v_max_target_vel, visual_field_limit, k_visual_field,
                         lum_scene, lum_target, tau_lum (luminance afferent)

    Returns:
        dstate: retina.State  state derivative (same NT shape as state).  The
                              delayed per-eye signals are the SSM output y — read
                              via read_outputs(state) (RetinaOut, incl. luminance).
    """
    # ── 1. Geometry — world_to_retina projection ─────────────────────────────
    target_pos, scene_angular_vel, scene_linear_vel, target_vel, scene_vis, target_vis = \
        world_to_retina(
            x_target, eye_offset_head, q_head, w_head, x_head, v_head,
            q_eye, w_eye, w_scene, v_scene, v_target,
            scene_present, target_present,
            sensory_params.visual_field_limit, sensory_params.k_visual_field,
        )

    # ── 2. Per-eye gating + sensor saturation ────────────────────────────────
    scene_angular_in   = velocity_saturation(scene_angular_vel * scene_vis, sensory_params.v_max_scene_vel)
    target_vel_in      = velocity_saturation(target_vel * target_vis, sensory_params.v_max_target_vel)
    scene_linear_in    = scene_linear_vel * scene_vis
    target_pos_in      = target_pos * target_vis
    # Retinal defocus (blur, D) is computed HERE from the optics — the dioptric
    # demand (1/target-distance + the eye's refractive error) minus the effective
    # accommodation (accommodation already lens-adjusted upstream). Mirrors how
    # slip/position are derived from the physical eye + world.
    defocus_in         = 1.0 / (jnp.linalg.norm(x_target) + 1e-9) + sensory_params.refractive_error - acc

    # ── 3. Advance sharp cascades (N stages × τ_retina, per signal) ──────────
    tau_retina = sensory_params.tau_vis_sharp
    N          = _N_STAGES_OTHER
    # Luminance afferent (pupillary light reflex): 1-pole light-adaptation LP of
    # this eye's physical retinal illumination — a lit full-field scene dominates,
    # a lone foveal target is a dim point source. Per-eye (one optic nerve each),
    # so a monocular afferent defect (RAPD) stays isolated; the consensual reflex
    # is assembled downstream in the pupil controller.
    L_phys = jnp.clip(sensory_params.lum_scene  * scene_present
                      + sensory_params.lum_target * target_present, 0.0, 1.0)
    dlum   = (L_phys - state.luminance) / sensory_params.tau_lum
    dstate = State(
        scene_angular_vel = delay_cascade_step(state.scene_angular_vel, scene_angular_in, tau_retina, N=N),
        scene_linear_vel  = delay_cascade_step(state.scene_linear_vel,  scene_linear_in,  tau_retina, N=N),
        target_pos        = delay_cascade_step(state.target_pos,        target_pos_in,    tau_retina, N=N),
        target_vel        = delay_cascade_step(state.target_vel,        target_vel_in,    tau_retina, N=N),
        scene_visible     = delay_cascade_step(state.scene_visible,     scene_vis,        tau_retina, N=N),
        target_visible    = delay_cascade_step(state.target_visible,    target_vis,       tau_retina, N=N),
        defocus           = delay_cascade_step(state.defocus,           defocus_in,       tau_retina, N=N),
        luminance         = dlum,
    )

    return dstate


def read_outputs(state):
    """State readout — returns RetinaOut from a per-eye retina.State.

    Last n_axes of each cascade buffer = sharp-cascade output (delayed signal);
    luminance is the current 1-pole afferent register.

    Args:
        state: per-eye retina.State
    """
    return RetinaOut(
        scene_angular_vel = state.scene_angular_vel[-3:],
        scene_linear_vel  = state.scene_linear_vel[-3:],
        target_pos        = state.target_pos[-3:],
        target_vel        = state.target_vel[-3:],
        scene_visible     = state.scene_visible[-1],
        target_visible    = state.target_visible[-1],
        defocus           = state.defocus[-1],
        luminance         = state.luminance[-1],
    )


