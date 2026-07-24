"""Otolith SSM — bilateral gravitoinertial acceleration sensors.

Models the utricle and saccule as a single lumped 3-D GIA sensor per side
(left / right), following the Laurens & Angelaki (2011, 2017) framework.

──────────────────────────────────────────────────────────────────────────
Anatomy summary
    Utricle  — horizontal macula, primarily sensitive to x/y GIA
    Saccule  — vertical macula, primarily sensitive to z GIA
    Each side lumped into one 3-D sensor (full-axis sensitivity).

    Unlike semicircular canals (clean push-pull pairs), left and right
    otoliths measure the SAME GIA with the SAME sign — they are parallel,
    not opposing.  Their bilateral output supports unilateral-loss detection
    and averaging reduces noise, but there is no differencing operator.

    PINV mixes: f̂ = (y_L + y_R) / 2

──────────────────────────────────────────────────────────────────────────
Physical signal
    GIA in head frame:  f(t) = g_head(t) + a_head(t)
        g_head  — gravity resolved into head frame = R(q_head)ᵀ · g_world
        a_head  — head linear acceleration (m/s²)

    Axis convention (left-handed world frame: x=right, y=up, z=forward):
        specific force is +y when head is upright (y=up)
        g_world = [0, +9.81, 0] m/s²

    At rest, upright: f = g_world = [0, +9.81, 0] m/s²

──────────────────────────────────────────────────────────────────────────
SSM interface (follows canal.py convention)
    State layout (6,): [x_L (3) | x_R (3)]  — LP adaptation states only.
    Head orientation q_head is passed as input (already integrated externally).

    step(x_oto, u, sensory_params) → (dx_oto, f_gia)
        u = [a_head (3) | q_head (3)]  — 6-D input
        f_gia  (3,)  — LP-filtered GIA estimate passed to gravity_estimator

──────────────────────────────────────────────────────────────────────────
Dynamics  (first-order tracking)
    Each side low-passes the raw head-frame GIA into a running estimate:
        dx_oto/dt = (S · f − x_oto) / τ_oto
        y_oto     = mean(x_L, x_R)           (running GIA estimate → brain)

    τ_oto is SHORT (light smoothing) so the estimate tracks the true GIA with a
    small lag and can absorb afferent noise.  This running estimate — the state,
    NOT the raw instantaneous f — is what the gravity estimator consumes; it is a
    faithful GIA tracker, not a slow adaptation that would drop the DC gravity
    component.
"""

from typing import NamedTuple

import jax.numpy as jnp

from oculomotor.models.plant_models.readout import rotation_matrix

# ── Sensor geometry ────────────────────────────────────────────────────────────

G0        = 9.81   # standard gravity (m/s²)
G_WORLD   = jnp.array([0., G0, 0.])   # specific force at rest, world frame (y=up)

# Sensitivity matrices (per side): full 3-D, identity (all axes equally sensitive)
SENS_LEFT  = jnp.eye(3)   # (3, 3)
SENS_RIGHT = jnp.eye(3)   # (3, 3)

# Mixing: GIA estimate = average of left and right LP states
# y shape (6,): [x_L (0:3) | x_R (3:6)]
PINV_SENS = 0.5 * jnp.array([
    [1., 0., 0., 1., 0., 0.],
    [0., 1., 0., 0., 1., 0.],
    [0., 0., 1., 0., 0., 1.],
])   # (3, 6): averages left and right per axis

# ── SSM constants ──────────────────────────────────────────────────────────────

N_STATES  = 6    # [x_L (3) | x_R (3)] — LP adaptation states; q_head is an input
N_INPUTS  = 6    # [a_head (3) | q_head (3)]
N_OUTPUTS = 3    # f_gia (3,) — GIA estimate

# Default initial state: both sides settled to gravity at rest, upright head
X0 = jnp.concatenate([SENS_LEFT @ G_WORLD, SENS_RIGHT @ G_WORLD])   # [9.81,0,0, 9.81,0,0]


# ── State NamedTuple ──────────────────────────────────────────────────────────

class State(NamedTuple):
    """Otolith state — bilateral LP adaptation registers (parallel, not push-pull)."""
    x_L: jnp.ndarray   # (3,) left  utricle/saccule LP state (m/s²)
    x_R: jnp.ndarray   # (3,) right utricle/saccule LP state (m/s²)


def rest_state():
    """Initial state — both sides settled at upright gravity."""
    return State(x_L=SENS_LEFT @ G_WORLD, x_R=SENS_RIGHT @ G_WORLD)


# ── GIA readout (running estimate → brain) ───────────────────────────────────────

def read_outputs(state):
    """Pure state readout — running GIA estimate in head frame (m/s²).

    Each side's state tracks the head-frame GIA; the brain reads their bilateral
    average (= the ``PINV_SENS`` mix, since ``SENS_LEFT = SENS_RIGHT = I``). This
    mirrors ``canal.read_outputs`` / ``retina.read_outputs``: downstream reads the
    GIA from *state*, so the transduction geometry lives only in ``step`` and
    there is no duplicated formula.

    Args:
        state: otolith.State (x_L, x_R) bilateral GIA-tracking states (m/s²)

    Returns:
        f_gia: (3,) running GIA estimate, head frame (m/s²)
    """
    return 0.5 * (state.x_L + state.x_R)


# ── SSM step ───────────────────────────────────────────────────────────────────

def step(state, u, sensory_params):
    """Single ODE step: otolith LP derivative + GIA estimate output.

    Args:
        state:          otolith.State  (x_L, x_R) bilateral adaptation states (m/s²)
        u:              (6,)  [a_head (3) | q_head (3)]
                              a_head — head linear acceleration (m/s²)
                              q_head — head orientation rotation vector (deg)
        sensory_params: SensoryParams  (reads tau_oto)

    Returns:
        dstate: otolith.State  state derivative (m/s³)
        f_gia:  (3,)           running GIA estimate → gravity_estimator (m/s²)
    """
    a_head = u[:3]   # (3,) head linear acceleration (m/s²)
    q_head = u[3:]   # (3,) head orientation rotation vector (deg)

    x_L = state.x_L
    x_R = state.x_R

    # Raw instantaneous GIA in head frame (mechanical transduction). This is the
    # ONLY place the sensor geometry lives — read_outputs() and sensory_model read
    # the running estimate from state, so there is no duplicated formula.
    # ypr_to_xyz convention: yaw→+y, pitch→−x, roll→+z (left-handed world frame).
    q_xyz = jnp.array([-q_head[1], q_head[0], q_head[2]])
    R     = rotation_matrix(q_xyz)          # (3,3) world←head rotation
    f     = R.T @ G_WORLD + R.T @ a_head    # (3,) raw GIA, head frame

    # First-order tracking: each side runs a low-pass estimate of the GIA.
    # tau_oto is SHORT (light smoothing) so the estimate tracks GIA with a small
    # lag and can absorb otolith afferent noise — this is a running estimate, not
    # a slow adaptation. The gravity estimator consumes this state (not raw f).
    tau  = sensory_params.tau_oto
    dx_L = (SENS_LEFT  @ f - x_L) / tau
    dx_R = (SENS_RIGHT @ f - x_R) / tau

    # Output = the running (state) estimate the brain reads — SSM y = C·x.
    f_gia = read_outputs(state)

    return State(x_L=dx_L, x_R=dx_R), f_gia
