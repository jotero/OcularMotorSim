"""Plant model — extraocular muscles + globe (Level 2: 3-D, two viscoelastic poles).

Second-order viscoelastic plant.  The eye has negligible INERTIA (the globe is
tiny and light — no θ̈ term); the two poles are two cascaded viscoelastic /
force-development low-passes, NOT a mass:

    motor_cmd ──► [muscle LP τ₂] ──► [orbital LP τ₁] ──► eye position

    τ₂  muscle fast pole  (~10-20 ms): muscle force can't appear instantly — it
        builds through the series-elastic element + activation dynamics.
    τ₁  orbital slow pole (~150 ms):  the dominant orbital-tissue viscoelasticity
        (= the old first-order τ_p).

Transfer function:   q_eye / motor_cmd  =  1 / [(τ₁·s + 1)(τ₂·s + 1)].

Why this exists (vs first-order): a step in the differential command now gives a
velocity *ramp* (the muscle force develops over τ₂) instead of an instant jump.
So an INO's antagonist relaxation coasts rather than pulses, while the NI's
pulse/acceleration feedforward cancels both poles for a normal, fast saccade.
Drop-in replacement for plant_model_first_order.step (same contract, one extra
per-eye state).

Binocular layout — the State carries POSITIONS in `left`/`right` (so analysis and
benches read `state.plant.left[:, 0]` = eye yaw unchanged) plus the muscle-force
intermediate in `left_musc`/`right_musc`.  `step()` operates on a single eye.

State:   x_musc (3,)  muscle-force state (fast-pole intermediate, deg)
         x_pos  (3,)  eye rotation vector (deg), bounded within ±orbital_limit
Input:   motor_cmd  (3,) or (6,)  pulse-step-slide motor command from NI
Outputs: q_eye  (3,)  eye rotation vector (= x_pos)
         w_true (3,)  instantaneous eye angular velocity (deg/s) (= dx_pos)

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


def to_array(state):
    """plant.State → (12,) flat array — legacy adapter (pos L|R then musc L|R)."""
    return jnp.concatenate([state.left, state.right, state.left_musc, state.right_musc])


def step(x_musc, x_pos, motor_cmd, plant_params, decode_matrix=None):
    """Single ODE step for one eye — two cascaded viscoelastic LPs.

    Args:
        x_musc:        (3,)   muscle-force state (fast-pole intermediate, deg)
        x_pos:         (3,)   eye rotation vector (deg), ∈ [−L, +L]
        motor_cmd:     (3,) or (6,)  pulse-step-slide motor command (or muscle activations)
        plant_params:  PlantParams  (tau_p = τ₁ orbital, tau_muscle = τ₂ muscle)
        decode_matrix: (3, 6) or None.  motor_cmd_3 = decode_matrix @ motor_cmd_6.

    Returns:
        dx_musc: (3,)  d(muscle-force)/dt   = (motor_cmd − x_musc)/τ₂
        dx_pos:  (3,)  d(position)/dt        = wall-clipped (x_musc − x_pos)/τ₁
        q_eye:   (3,)  eye rotation vector   (= x_pos)
        w_true:  (3,)  instantaneous eye angular velocity (= dx_pos)
    """
    tau_1 = plant_params.tau_p          # orbital slow pole
    tau_2 = plant_params.tau_muscle     # muscle fast pole
    L     = plant_params.orbital_limit

    # Decode 6-D muscle activations → 3-D effective motor command
    if decode_matrix is not None:
        motor_cmd = decode_matrix @ motor_cmd

    # Stage 1 (fast): muscle force builds toward the command — this is the pole
    # that turns a command step into a velocity ramp (no inertia; force lag).
    dx_musc = (motor_cmd - x_musc) / tau_2

    # Stage 2 (slow): orbital position follows the developed muscle force.
    w_raw = (x_musc - x_pos) / tau_1

    # Orbital walls on the POSITION velocity: zero it when at ±L and pushing out.
    w_true = jnp.where(x_pos >= L,  jnp.minimum(w_raw,  0.0), w_raw)
    w_true = jnp.where(x_pos <= -L, jnp.maximum(w_true, 0.0), w_true)

    dx_pos = w_true
    q_eye  = x_pos
    return dx_musc, dx_pos, q_eye, w_true
