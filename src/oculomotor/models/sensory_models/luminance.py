"""Luminance afferent SSM — retinal illumination sensor for the pupillary reflex.

A minimal first-order low-pass modelling the afferent limb of the pupillary
light reflex (PLR): retinal photoreceptor light adaptation + the pretectal
olivary nucleus (PON) transport delay.  It converts the *physical* luminance
reaching the retina (a scalar in [0, 1], assembled in ``sensory_model.step``
from scene / target presence) into an *afferent* luminance signal that drives
pupil constriction in ``brain_models.pupil``.

Physiology:
    The PLR has a ~0.2–0.5 s latency dominated by retinal + pretectal
    processing (Ellis 1981; Clarke & Ikeda 1985).  A single first-order pole
    with τ_lum ≈ 0.3 s captures this afferent lag; the slower iris-plant lag
    lives in ``plant_models.pupil_plant``.  This pathway is intentionally
    separate from the image-forming retina (``retina.py``): the PLR runs via
    the retinotectal route (retina → PON → Edinger-Westphal), not via LGN /
    cortex, and responds to whole-field illumination, not retinal geometry.

    The two eyes are SEPARATE afferents (one per retina / optic nerve) so the
    consensual reflex and a monocular afferent defect (RAPD, swinging-flashlight)
    can be modelled: each pupil is driven by the SUM of both eyes' afferents
    (bilateral pretectum → both Edinger-Westphal nuclei), computed downstream in
    ``brain_models.pupil``.  The connector (``sensory_model.step``) feeds each
    register its own eye's physical luminance.

Dynamics (first-order light adaptation, cf. otolith.py — parallel, not push-pull):
    dx_lum/dt = (L_phys − x_lum) / τ_lum      (elementwise over [L, R])
    y_lum     = x_lum                          (afferent luminance → pupil)

SSM interface (follows canal.py / otolith.py convention):
    State (2,):  [x_lum_L, x_lum_R]  — per-eye afferent registers (normalised, ~[0, 1])
    step(x_lum, L_phys, tau_lum) → (dx_lum, y_lum)   with L_phys, y_lum shape (2,)

References:
    Ellis CJK (1981) Br J Ophthalmol 65:754   — PLR latency
    Clarke RJ, Ikeda H (1985) Exp Brain Res 57:224 — pretectal light response
    McDougal & Gamlin (2015) Compr Physiol 5:439   — PLR circuitry review
"""

from typing import NamedTuple

import jax.numpy as jnp

N_STATES  = 2   # [x_lum_L, x_lum_R] — per-eye afferent registers (normalised)
N_INPUTS  = 2   # L_phys  — per-eye physical retinal luminance (normalised, [0, 1])
N_OUTPUTS = 2   # y_lum   — per-eye afferent luminance → pupil light reflex


class State(NamedTuple):
    """Luminance afferent state — per-eye light-adaptation registers."""
    x_lum: jnp.ndarray   # (2,) [L, R] afferent luminance (normalised, ~[0, 1])


def rest_state():
    """Initial state — dark (zero afferent luminance, both eyes)."""
    return State(x_lum=jnp.zeros(2))


def to_array(state):
    """luminance.State → (2,) flat array — legacy adapter; SimState uses the NT directly."""
    return state.x_lum


def read_activations(state):
    """Afferent luminance is the sole field — identity projection."""
    return state


def step(state, L_phys, tau_lum):
    """Single ODE step for the per-eye luminance afferent low-pass.

    Args:
        state:   luminance.State  (x_lum,) per-eye afferent registers (2,)
        L_phys:  (2,)    per-eye physical retinal luminance in [0, 1] (assembled
                 from per-eye scene / target presence in sensory_model.step)
        tau_lum: scalar  afferent + PLR-latency TC (s)

    Returns:
        dstate: luminance.State  state derivative (1/s)
        y_lum:  (2,)             per-eye afferent luminance → pupil light reflex
    """
    x_lum = state.x_lum
    dx    = (L_phys - x_lum) / tau_lum
    return State(x_lum=dx), x_lum
