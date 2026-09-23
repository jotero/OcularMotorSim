"""Target perception — FEF/dlPFC working memory feeding the saccade generator.

Pursuit drive (post-delay EC subtraction + magnitude/directional gates) used
to live here but moved into `cerebellum.py` (pursuit region — paraflocculus
ventral / vermis VI–VII).  This module now owns only the working-memory layer
that lets brief target flashes drive a saccade after the flash ends, and that
decays slowly when the target is gone.

Component (working memory):
  4-state cognitive layer (3-D last-seen position + trust scalar).  The SG
  uses these to fire a saccade toward the remembered location after the
  flash ends.  Memory drains proportional to |ec_d_target|, so any eye
  movement (saccade, fast pursuit, head-impulse fast-phase) consumes the
  memory and prevents re-triggering on the residual.

State layout (N_STATES = 4):
    x_target_mem = [x_mem_pos (3) | mem_age (1)]

Outputs of step():
    dstate         pt.State derivative
    tgt_pos_eff   (3,)   blended raw + memory target position → SG
    tgt_vis_eff   scalar blended visibility                   → SG
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


# ── Working memory time constants — local module hacks, not in BrainParams ──
# These shape FEF/dlPFC-style gaze memory so brief flashes fire a saccade, then
# the saccade burst consumes the memory so it doesn't re-trigger.
_TAU_TARGET_MEM_UPDATE       = 0.02    # memory lock TC when target is visible (s).
                                       # Lock progress over a flash is ~∫vis dt / TC,
                                       # so a 50 ms TC captures only ~20% of the
                                       # position from a 10 ms pulse. Swept 0.05 /
                                       # 0.02 / 0.01: 0.02 is the best compromise
                                       # between brief-flash capture and long-flash
                                       # accuracy (see the flash-gain table).
_TARGET_MEM_DETECT           = 0.15    # delayed-visibility level that counts as "seen"
_TARGET_MEM_DETECT_WIDTH     = 0.01    # softness of that detection gate. Must be
                                       # sharp enough that `seen` is genuinely ~0 in
                                       # the dark: the age reset divides by
                                       # _TAU_TARGET_MEM_RESET, so a residual tail of
                                       # even 0.007 becomes a ~1.3/s pull that pins
                                       # age near zero and the memory never expires.
_TAU_TARGET_MEM_RESET        = 0.005   # how fast age returns to 0 once seen (s).
                                       # MUST be well under the briefest flash we
                                       # want to register: the reset is exponential
                                       # from the current age, so at 0.05 s a 20 ms
                                       # sighting only drags age from 4.0 to 2.7 —
                                       # still past the hold window, so the memory
                                       # stays invalid and no saccade fires.
_TARGET_MEM_HOLD_S           = 2.0     # PERFECT-memory plateau after last sighting (s)
_TARGET_MEM_FADE_S           = 0.20    # width of the forget edge (s)

# Why age rather than a trust integrator:
# The previous rule drove a single `trust` scalar toward `target_visible` and
# committed above a fixed threshold. That made one scalar do two incompatible
# jobs — commit fast on brief evidence, and release promptly once the target is
# genuinely gone — and it failed both. A 20 ms flash reached trust 0.161 against
# a 0.2 threshold, so no saccade fired at all; meanwhile a steadily-viewed target
# left the memory saturated for 7.7 s after it vanished (the hard sigmoid pinned
# mem_active at 1 until trust decayed to threshold, so the decay TC never set the
# release time). Tightening either behaviour necessarily worsened the other.
# Tracking TIME SINCE LAST EVIDENCE separates them: detection is instantaneous
# (any real sighting resets age, so brief flashes commit immediately), while
# release is governed purely by the clock — a flat plateau for _TARGET_MEM_HOLD_S
# then a fast fade. The two are now tunable independently.

# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """Target perception state — working-memory population + recency."""
    mem_pos: jnp.ndarray   # (3,)   target working memory pop  [dlPFC / FEF]
    mem_age: jnp.ndarray   # scalar seconds since the target was last SEEN [dlPFC].
                           #        0 while visible; grows at 1 s/s in the dark.
                           #        Memory is held perfectly until _TARGET_MEM_HOLD_S
                           #        then fades over _TARGET_MEM_FADE_S.


# State == Activations: working-memory pop is itself a firing-rate signal.
Activations = State


def rest_state():
    """Zero state — used for SimState initialisation.

    mem_age starts well past the hold window: at rest nothing has been seen, so
    the memory must be INVALID. Starting it at 0 would mean "just saw the target".
    """
    return State(mem_pos=jnp.zeros(3),
                 mem_age=jnp.float32(_TARGET_MEM_HOLD_S + 10.0 * _TARGET_MEM_FADE_S))


def read_activations(state):
    """Working-memory IS the activation — identity projection."""
    return state


def step(activations,
         target_visible, target_pos,
         ec_d_target):
    """Single ODE step for target-side perception.

    Activation-driven: working-memory pop firing rates come from `activations`
    (acts.pt).  State == Activations for pt today (identity projection).

    Args:
        activations:     pt.Activations  mem_pos (3,) | mem_age (scalar, s)
        target_visible:  scalar  delayed cyclopean target visibility gate ∈ [0,1]
        target_pos:      (3,)    delayed cyclopean retinal target position (eye frame, deg)
        ec_d_target:     (3,)    delayed EC (cascade-matched to target_vel; eye frame, deg/s)

    Returns:
        dstate:        pt.State  derivative
        tgt_pos_eff:   (3,)    effective target position → SG
        tgt_vis_eff:   scalar  effective target visibility → SG
    """
    x_mem = activations.mem_pos
    age   = activations.mem_age
    # "Seen" is a soft detection on the delayed visibility, NOT an accumulation:
    # a genuine sighting of any duration counts immediately, which is what lets a
    # brief flash drive a saccade.
    seen  = jax.nn.sigmoid((target_visible - _TARGET_MEM_DETECT) / _TARGET_MEM_DETECT_WIDTH)

    # Memory drain proportional to delayed-EC magnitude. Whenever the eye is
    # moving (saccade burst, fast pursuit overshoot, head-impulse fast-phase),
    # |ec_d_target| is large in deg/s → drain rate grows in 1/s, draining the
    # remembered position fast even for small saccades whose cascade-delayed
    # EC peaks at only tens of deg/s. Between saccades the LP cascade has
    # decayed and inv_consume → 0, so the memory holds.
    # Age is the "give up and look home" clock; it is not driven by the EC.
    inv_consume = jnp.linalg.norm(ec_d_target)
    # Lock strength is raw visibility, NOT the saturated `seen` gate. The two do
    # different jobs and must not be conflated: `seen` stays pinned at 1 until
    # visibility falls below the detection level, so locking on it keeps dragging
    # the memory along while the position signal is already DECAYING — the memory
    # ends up holding the faded tail instead of the peak. Weighting by visibility
    # makes the lock fade with the evidence, freezing x_mem near its best estimate.
    dx_mem = target_visible * (target_pos - x_mem) / _TAU_TARGET_MEM_UPDATE \
             - inv_consume * x_mem

    # Age: pinned to 0 while the target is seen, otherwise counts real seconds.
    dx_age = (1.0 - seen) - seen * age / _TAU_TARGET_MEM_RESET
    dstate = State(mem_pos=dx_mem, mem_age=dx_age)

    # Validity is a function of AGE alone: flat ~1 through the hold window, then
    # a fast fade. Nothing here depends on how long or how brightly the target was
    # seen, which is what decouples "commit on a brief flash" from "forget promptly".
    mem_valid   = jax.nn.sigmoid((_TARGET_MEM_HOLD_S - age) / _TARGET_MEM_FADE_S)
    tgt_pos_eff = target_visible * target_pos + (1.0 - target_visible) * mem_valid * x_mem
    tgt_vis_eff = jnp.maximum(target_visible, mem_valid)

    return dstate, tgt_pos_eff, tgt_vis_eff


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

N_STATES = 4   # 3-D last-seen position + 1 age-since-seen scalar
