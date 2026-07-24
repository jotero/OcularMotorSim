"""Sensory model — thin connector wiring canal, otolith, and per-eye retina.

Imports the canal SSM (canal.py), otolith SSM (otolith.py), and per-eye retina
(retina.py) and aggregates them into a single combined step + read_outputs.
Binocular fusion + brain-LP smoothing live in the brain
(``brain_models.perception_cyclopean``) — they operate on already-delayed
per-eye signals and are cortical computations, not peripheral.

Signal flow:
    w_head            → [Canal array]    → y_canals (6,)   afferent firing rates
    a_head, q_head    → [Otolith array]  → f_gia (3,)      running GIA → gravity estimator
    per-eye stimulus  → [retina.step] L  → RetinaOut_L (delayed per-eye signals + luminance)
                      → [retina.step] R  → RetinaOut_R
    SensoryOutput bundles canal + otolith + retina_L + retina_R for the brain.

State layout (200 states) — a NamedTuple of per-subsystem States (no flat array,
no _IDX_* slice constants; Diffrax handles the PyTree natively):
    State(
        canal    : canal.State    (12)   two-stage bandpass
        otolith  : otolith.State   (6)   bilateral GIA-tracking LP
        retina_L : retina.State   (91)   sharp cascade (90) + luminance (1)
        retina_R : retina.State   (91)
    )
Each subsystem is read as an attribute (state.canal, state.retina_L, …); the
per-subsystem outputs are exposed through read_outputs().
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.sensory_models import canal            as _canal
from oculomotor.models.sensory_models import otolith          as _otolith
from oculomotor.models.sensory_models import retina           as _retina
from oculomotor.models.sensory_models.retina import RetinaOut  # noqa: F401  (re-export)


# ── Sensory parameters ──────────────────────────────────────────────────────────

class SensoryParams(NamedTuple):
    """Sensory parameters — canal mechanics + otolith + visual pathway.

    These are determined by peripheral anatomy/physiology.  Fixed during
    typical patient fitting but freed for known peripheral pathology
    (e.g. canal paresis → canal_gains, drug effects → tau_vis_sharp).
    """
    # Semicircular canals — Steinhausen torsion-pendulum (Fernandez & Goldberg 1971)
    tau_c:              float       = 5.0    # cupula adaptation TC (s); HP corner ≈ 0.03 Hz
    tau_s:              float       = 0.005  # endolymph inertia TC (s); LP corner ≈ 32 Hz
    canal_gains:        jnp.ndarray = jnp.ones(6)  # (6,) per-canal scale; 1=intact, 0=paresis
    canal_floor:        float       = 80.0   # resting discharge (deg/s); inhibitory saturation point
                                             # (Goldberg & Fernandez 1971 J Neurophysiol 34:635)
    canal_v_max:        float       = 400.0  # excitatory canal afferent saturation (deg/s).
                                             # Sensor-side ceiling — applied in canal.step output.
                                             # Inhibitory saturation is the FLOOR (canal_floor); this is
                                             # the symmetric upper clip on afferent firing rate.
                                             # Goldberg & Fernandez 1971: ~300–600 deg/s; 400 conservative.
                                             # At typical stimulus velocities (<200 deg/s) the clip is inert.

    # Otolith — first-order GIA tracking. tau_oto is SHORT so the state is a
    # running estimate of GIA (small lag + light noise smoothing), NOT a slow
    # adaptation; this state is what the gravity estimator reads.
    tau_oto:            float       = 0.02   # otolith GIA-tracking TC (s); short → tracks GIA closely

    # Visual pathway — sensor-side parameters only. The brain-side LP smoothing
    # TCs (tau_vis_smooth_*) and the binocular-fusion-policy parameters (npc /
    # div_max / vert_max / tors_max / eye_dominant) live in BrainParams since
    # they're cortical decisions; perception_cyclopean (in brain_models) reads
    # them from there.
    tau_vis_sharp:      float = 0.05   # sharp cascade mean delay (s) — photo-transduction +
                                       # axonal/synaptic transport (Pugh & Lamb 1993,
                                       # Dunn & Rieke 2006). Used by retina.step.
    visual_field_limit: float = 90.0   # retinal eccentricity limit (deg); ~90° monocular field
    k_visual_field:     float = 1.0    # sigmoid steepness for visual field gate (1/deg)

    # Sensory noise (std in output units; 0 = noiseless). All four sources are
    # Ornstein-Uhlenbeck processes — short τ approaches white noise (band-limited),
    # longer τ produces drift-like fluctuations.
    sigma_canal:        float       = 1.0    # canal afferent noise (deg/s); filtered heavily by VS/NI/plant
    tau_canal_drift:    float       = 0.005  # OU TC for canal noise (s); essentially band-limited white
    sigma_slip:         float       = 0.0    # retinal slip noise (deg/s); drives VS/OKR (off by default)
    tau_slip_drift:     float       = 0.005  # OU TC for slip noise (s)
    sigma_pos:          float       = 0.2    # retinal position drift (deg); triggers microsaccades
    tau_pos_drift:      float       = 0.2    # OU TC for retinal-pos drift (s); inter-microsaccade interval
    sigma_vel:          float       = 1.0    # target velocity noise (deg/s); drives pursuit jitter
    tau_vel_drift:      float       = 0.005  # OU TC for retinal-vel noise (s); essentially band-limited white

    # Binocular geometry
    ipd:                float       = 0.064  # inter-pupillary distance (m); ~64 mm adult

    # Luminance afferent — pupillary light reflex (PLR) input. Physical retinal
    # luminance is assembled from scene / target presence in step() (consensual:
    # both eyes averaged), then low-passed by the luminance sub-SSM.
    tau_lum:            float       = 0.3    # luminance afferent + PLR-latency TC (s) [Ellis 1981; McDougal 2015]
    lum_scene:          float       = 1.0    # luminance weight of a lit full-field scene (scene_present → L)
    lum_target:         float       = 0.15   # luminance weight of a lone foveal target (dim point source)

    # Sensor-side velocity saturation (applied per-eye in retina.step before sharp cascade).
    # Mirrors the speed tuning of MT/MST (target) and NOT/AOS (scene) neurons. Must match
    # the v_max_pursuit / v_max_okr values in BrainParams so that the EC correction (clipped
    # to the same ceiling) exactly cancels what made it through the retina cascade.
    v_max_target_vel:   float       = 40.0   # MT/MST speed ceiling (deg/s)
    v_max_scene_vel:    float       = 80.0   # NOT/AOS speed ceiling (deg/s)

# ── Re-exports for external callers ────────────────────────────────────────────

# Canal
N_CANALS          = _canal.N_CANALS        # 6
ORIENTATIONS      = _canal.ORIENTATIONS    # (6, 3)
PINV_SENS         = _canal.PINV_SENS       # (3, 6)
FLOOR             = _canal.FLOOR           # 80.0
_SOFTNESS         = _canal._SOFTNESS       # 0.5  nonlinearity sharpness
canal_nonlinearity = _canal.nonlinearity   # renamed in canal.py

# Visual delay — the delayed cyclopean signals live on the BRAIN state, not here.
# They are read from the perception_cyclopean sub-state (brain_state.pc) via that
# module's C_* readout matrices (the brain state is a NamedTuple now — no _IDX_*
# slices). External code should import the matrices directly:
#     from oculomotor.models.brain_models.perception_cyclopean import C_slip, C_pos, ...

# ── State layout ───────────────────────────────────────────────────────────────
# Per-eye retina sharp cascades only (90 each). The cyclopean brain LP block
# now lives in brain state (perception_cyclopean is in brain_models).

_N_CANAL_STATES  = _canal.N_STATES                # 12
_N_OTO_STATES    = _otolith.N_STATES              #  6
_N_RETINA_PER_EYE= _retina.N_STATES_PER_EYE       # 91 (90 cascade + 1 luminance)
_N_VIS_STATES    = 2 * _N_RETINA_PER_EYE          # 91+91 = 182
N_STATES         = _N_CANAL_STATES + _N_OTO_STATES + _N_VIS_STATES  # 12+6+182 = 200


# ── State NamedTuple ──────────────────────────────────────────────────────────

class State(NamedTuple):
    """Top-level sensory state — canal + otolith + per-eye retina.

    Per-eye retina now carries its own afferent-luminance register (pupillary
    light reflex), so there is no separate luminance sub-SSM.
    """
    canal:    _canal.State      # (12,)  bandpass two-stage SSM
    otolith:  _otolith.State    #  (6,)  bilateral GIA-tracking LP
    retina_L: _retina.State     # (91,)  per-eye sharp cascade + luminance — left eye
    retina_R: _retina.State     # (91,)  per-eye sharp cascade + luminance — right eye


def rest_state():
    """Initial sensory state (otolith starts settled to gravity, others zero)."""
    return State(
        canal    = _canal.rest_state(),
        otolith  = _otolith.rest_state(),
        retina_L = _retina.rest_state(),
        retina_R = _retina.rest_state(),
    )


# ── Bundled sensory output ──────────────────────────────────────────────────────

class SensoryOutput(NamedTuple):
    """Bundled sensory outputs — passed as a unit to brain_model.

    Visual signals are PER-EYE and already sharp-cascade delayed. Binocular fusion
    + brain-LP smoothing happen downstream in perception_cyclopean; this bundle
    carries the two eyes' RetinaOut untouched. Afferent luminance (pupillary light
    reflex) rides inside each RetinaOut, so a monocular afferent defect (RAPD)
    stays per-eye — the pupil controller assembles the [L, R] pair itself.

    Fields:
        canal:    (6,)      canal afferent rates
        otolith:  (3,)      running GIA estimate in head frame (m/s²) → gravity estimator
        retina_L: RetinaOut delayed per-eye signals (incl. luminance) — left eye
        retina_R: RetinaOut delayed per-eye signals (incl. luminance) — right eye
    """
    canal:     jnp.ndarray           # (6,)  canal afferent rates
    otolith:   jnp.ndarray           # (3,)  running GIA estimate (m/s², head frame)
    retina_L:  RetinaOut             # delayed per-eye signals (incl. luminance) — left eye
    retina_R:  RetinaOut             # delayed per-eye signals (incl. luminance) — right eye


def read_outputs(state, sensory_params):
    """Read all sensory outputs from the current state (pure state readout).

    Args:
        state:          sensory_model.State
        sensory_params: SensoryParams

    Returns:
        SensoryOutput with delayed per-eye signals.
    """
    # Each subsystem exposes a uniform read_outputs(state[, params]) — the canal
    # afferent nonlinearity, the otolith running-GIA estimate, and each eye's
    # delayed retinal signals (which now carry their own afferent luminance).
    return SensoryOutput(
        canal    = _canal.read_outputs(state.canal, sensory_params),
        otolith  = _otolith.read_outputs(state.otolith),
        retina_L = _retina.read_outputs(state.retina_L),
        retina_R = _retina.read_outputs(state.retina_R),
    )


# ── Combined step ───────────────────────────────────────────────────────────────

def step(state,
         # ── Head kinematics ───────────────────────────────────────────────────
         q_head, w_head, x_head, v_head, a_head,
         # ── Eye kinematics (prism-shifted by ODE before this call) ────────────
         q_eye_L, w_eye_L, q_eye_R, w_eye_R,
         # ── Scene stimulus (per eye — L/R split enables stereoscopic displays) ─
         q_scene_L, w_scene_L, x_scene_L, v_scene_L,
         q_scene_R, w_scene_R, x_scene_R, v_scene_R,
         # ── Target stimulus (per eye — L/R split enables stereoscopic displays) ─
         p_target_L, dp_dt_L,
         p_target_R, dp_dt_R,
         # ── Defocus (per eye; = acc_demand + refractive_error − x_acc_plant) ──
         defocus_L, defocus_R,
         # ── Visibility flags ──────────────────────────────────────────────────
         scene_present_L, scene_present_R,
         target_present_L, target_present_R, target_strobed,
         # ── Parameters ───────────────────────────────────────────────────────
         sensory_params):
    """Single ODE step for the sensory subsystem (canal + otolith + per-eye retina).

    Args:
        state: sensory_model.State (canal + otolith + retina_L + retina_R)

    Returns:
        dstate: sensory_model.State  state derivative
    """
    ipd_half  = sensory_params.ipd * 0.5
    eye_off_L = jnp.array([-ipd_half, 0.0, 0.0])
    eye_off_R = jnp.array([ ipd_half, 0.0, 0.0])

    dcanal,   _ = _canal.step(state.canal,     w_head, sensory_params)
    dotolith, _ = _otolith.step(state.otolith, jnp.concatenate([a_head, q_head]), sensory_params)

    # Per-eye retina cascades (cyclopean fusion happens in brain).
    dretina_L, _ = _retina.step(
        state.retina_L, eye_off_L, q_head, w_head, x_head, v_head,
        q_eye_L, w_eye_L, w_scene_L, v_scene_L, p_target_L, dp_dt_L,
        defocus_L, scene_present_L, target_present_L, target_strobed,
        sensory_params)
    dretina_R, _ = _retina.step(
        state.retina_R, eye_off_R, q_head, w_head, x_head, v_head,
        q_eye_R, w_eye_R, w_scene_R, v_scene_R, p_target_R, dp_dt_R,
        defocus_R, scene_present_R, target_present_R, target_strobed,
        sensory_params)

    # Per-eye afferent luminance (pupillary light reflex) is advanced inside each
    # retina.step from that eye's own scene / target presence — no separate
    # luminance sub-SSM here.

    return State(canal=dcanal, otolith=dotolith, retina_L=dretina_L, retina_R=dretina_R)
