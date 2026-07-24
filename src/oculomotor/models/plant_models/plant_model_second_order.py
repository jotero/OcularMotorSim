"""Plant model — extraocular muscles + globe (Level 2: 3-D, two viscoelastic poles).

Second-order viscoelastic plant.  The eye has negligible INERTIA (the globe is
tiny and light — no θ̈ term); the two poles are two cascaded viscoelastic /
force-development low-passes, NOT a mass:

    motor_cmd ──► [muscle LP τ₂] ──► [orbital LP τ₁] ──► eye position

    τ₂  muscle fast pole  (~10-20 ms): muscle force can't appear instantly — it
        builds through the series-elastic element + activation dynamics.
    τ₁  orbital slow pole (~150 ms):  the dominant orbital-tissue viscoelasticity
        (= the old first-order τ_p).

Transfer function:   q_eye / motor_cmd  =  (Tz·s + 1) / [(τ₁·s + 1)(τ₂·s + 1)].

    Tz  muscle series-elastic ZERO (~8 ms): the SEE transmits force *changes* to
        position with a lead (numerator), so the eye responds fast without an
        aggressive neural command.  Robinson; Optican & Miles 1985 (4th-order plant).
        The NI slide (brain.tau_slide) is set = Tz so its pole cancels this zero and
        the pulse-slide-step lands NI_net cleanly (no glissade).

Why this exists (vs first-order): a step in the differential command now gives a
velocity *ramp* (the muscle force develops over τ₂) instead of an instant jump.
So an INO's antagonist relaxation coasts rather than pulses, while the NI's
pulse/acceleration feedforward cancels both poles for a normal, fast saccade.
Drop-in replacement for plant_model_first_order.step (same contract, one extra
per-eye state).

Binocular layout — the State carries POSITIONS in `left`/`right` (so analysis and
benches read `state.plant.left[:, 0]` = eye yaw unchanged) plus the muscle-force
intermediate in `left_musc`/`right_musc`.  `step()` is BINOCULAR: it takes the
plant State + the (12,) nerve vector and returns the State derivative (both eyes);
the per-eye math lives in the private `_step_eye` helper.

Per-eye state: x_musc (3,)  muscle-force intermediate (fast-pole), deg
               x_pos  (3,)  eye rotation vector (deg), bounded within ±orbital_limit
Input:   nerves (12,)  per-muscle activations [L6 | R6] from the FCP
Outputs (read off the state / derivative, NOT returned by step — the common SSM
contract keeps step to the derivative):
         eye position = state.left/right    (C = I: the position IS the state)
         eye velocity = dstate.left/right   (= d position/dt: reuse the derivative)

Parameters (PlantParams, shared with the first-order module):
  τ_p        — orbital slow pole τ₁ (s). 0.15 s.
  tau_muscle — muscle fast pole τ₂ (s). ~0.013 s.
  orbital_limit — mechanical half-range (deg). 50 deg.
"""

from typing import NamedTuple

import jax.numpy as jnp

# Reuse the shared PlantParams (tau_p = τ₁, tau_muscle = τ₂, orbital_limit).
from oculomotor.models.plant_models.plant_model_first_order import PlantParams


# ── State layout ───────────────────────────────────────────────────────────────

N_STATES  = 12          # (muscle 3 + position 3) per eye × 2 eyes
N_INPUTS  = 6           # muscle activation vector from brain_model (6,)
N_OUTPUTS = 3           # q_eye (position, per eye)


class State(NamedTuple):
    """Binocular 2nd-order plant state.

    `left`/`right` are the eye POSITIONS (rotation vectors, deg) — read directly
    by analysis/benches, exactly like the first-order plant.  `left_musc`/
    `right_musc` are the muscle-force intermediates (the fast-pole stage).
    """
    left:       jnp.ndarray   # (3,) left  eye position (rotation vector)
    right:      jnp.ndarray   # (3,) right eye position
    left_musc:  jnp.ndarray   # (3,) left  muscle-force state (fast-pole intermediate)
    right_musc: jnp.ndarray   # (3,) right muscle-force state


def rest_state():
    """Zero state — both eyes at primary position, muscle intermediates at 0."""
    return State(left=jnp.zeros(3), right=jnp.zeros(3),
                 left_musc=jnp.zeros(3), right_musc=jnp.zeros(3))


def _step_eye(x_musc, x_pos, motor_cmd, plant_params, decode_matrix=None):
    """One eye — two cascaded viscoelastic LPs.  Returns (dx_musc, dx_pos).

    dx_pos IS the instantaneous eye velocity (wall-clipped); q_eye = x_pos, so
    the caller reads position off the state and velocity off this derivative.

    Args:
        x_musc:        (3,)   muscle-force state (fast-pole intermediate, deg)
        x_pos:         (3,)   eye rotation vector (deg), ∈ [−L, +L]
        motor_cmd:     (3,) or (6,)  pulse-step-slide motor command (or muscle activations)
        plant_params:  PlantParams  (tau_p = τ₁ orbital, tau_muscle = τ₂ muscle)
        decode_matrix: (3, 6) or None.  motor_cmd_3 = decode_matrix @ motor_cmd_6.
    """
    tau_1 = plant_params.tau_p          # orbital slow pole τ₁
    tau_2 = plant_params.tau_muscle     # muscle fast pole  τ₂
    tau_z = plant_params.tau_see        # muscle series-elastic ZERO Tz (numerator lead)
    L     = plant_params.orbital_limit

    # Decode 6-D muscle activations → 3-D effective motor command
    if decode_matrix is not None:
        motor_cmd = decode_matrix @ motor_cmd

    # Stage 1 (fast): muscle force builds toward the command — this is the pole
    # that turns a command step into a velocity ramp (no inertia; force lag).
    dx_musc = (motor_cmd - x_musc) / tau_2

    # Stage 2 (slow): orbital position follows the developed muscle force, PLUS the
    # series-elastic lead — the SEE transmits force *changes* (dx_musc) straight to
    # position velocity.  This adds the numerator zero:
    #     x_pos/x_musc = (1 + Tz·s)/(1 + τ₁·s)   →   q_eye/motor_cmd = (1+Tz s)/[(1+τ₁ s)(1+τ₂ s)]
    # so the eye responds fast to force changes without an aggressive neural command;
    # the NI's slide (tau_slide = Tz) cancels this zero so the pulse-slide-step lands
    # NI_net with no post-saccadic glissade.  (Tz mismatch ↔ slide ⇒ glissade — the
    # clinical picture.)  Robinson; Optican & Miles 1985.
    w_raw = (x_musc - x_pos) / tau_1 + (tau_z / tau_1) * dx_musc

    # Orbital walls on the POSITION velocity: zero it when at ±L and pushing out.
    w_true = jnp.where(x_pos >= L,  jnp.minimum(w_raw,  0.0), w_raw)
    w_true = jnp.where(x_pos <= -L, jnp.maximum(w_true, 0.0), w_true)

    return dx_musc, w_true       # dx_pos = w_true (clipped eye velocity)


def step(state, nerves, plant_params, decode_L=None, decode_R=None):
    """Binocular ODE step: nerves → plant State derivative (both eyes).

    Common SSM shape — takes the subsystem State + input, returns ONLY the State
    derivative.  The eye-position and eye-velocity outputs are not returned: they
    are trivially available to the caller as the state (position, C = I) and this
    derivative (velocity = d position/dt).

    Args:
        state:        plant.State  binocular positions + muscle-force intermediates
        nerves:       (12,)  per-muscle activations [L6 | R6]  (split in half per eye)
        plant_params: PlantParams
        decode_L/R:   (3, 6) muscle→command decode per eye (M_MUSCLE_ACTION_INV_L/R).
                      None = use the per-eye command directly.

    Returns:
        dstate: plant.State  state derivative
    """
    h = nerves.shape[0] // 2
    dx_m_L, dx_p_L = _step_eye(state.left_musc,  state.left,  nerves[:h], plant_params, decode_L)
    dx_m_R, dx_p_R = _step_eye(state.right_musc, state.right, nerves[h:], plant_params, decode_R)
    return State(left=dx_p_L, right=dx_p_R, left_musc=dx_m_L, right_musc=dx_m_R)
