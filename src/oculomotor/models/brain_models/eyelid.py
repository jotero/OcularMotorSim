"""Eyelid control — levator / orbicularis command (blinks + posture + lesions).

Stateless brain module (cf. pupil.py): computes the commanded upper-lid CLOSURE
(0 = open, 1 = fully closed) per eye from three drives, then the eyelid plant
(``plant_models.eyelid_plant``) low-passes it to the actual lid position.

Drives (per eye):
  1. Resting posture — the lid is held OPEN by the levator palpebrae superioris
     (CN III, somatic) with a small contribution from Müller's muscle (sympathetic).
     Loss of either raises the resting closure = ptosis. The levator is innervated
     in TWO stages, BOTH following the shared CN III gains (no duplicate lid knobs):
     the central caudal nucleus projects to BOTH levators (ipsi + contra) and
     follows the CN III subnucleus gains (g_nucleus), then peripheral CN III nerve
     fibres run to each lid and follow g_nerve. Hence a NUCLEAR lesion gives
     bilateral PARTIAL ptosis, a NERVE lesion unilateral COMPLETE ptosis:
         levator_tone_i = cn3_nerve_i · [(1−c)·cn3_nuc_i + c·cn3_nuc_other]
     where c = eyelid_levator_contra_frac (contra fraction; ~0.3 → ipsi-dominant).
  2. Blinks — a central blink command (conjugate) drives the orbicularis oculi
     (CN VII) to snap the lid shut.  Gated per eye by orbicularis integrity.
  3. Gaze-follow — the upper lid follows the eye vertically; downgaze lowers it
     (levator relaxation), mimicking web/avatar.js.

    ptosis_i  = ptosis_levator·(1 − g_levator_i) + ptosis_muller·(1 − g_symp_i)
    closure_i = clip(ptosis_i + gaze_follow(pitch_i) + blink·g_cn7_i, 0, 1)

Lesions (per eye; helpers in sim.simulator):
  * Ptosis — CN III NERVE (levator): follows the SHARED oculomotor nerve gains
    (g_nerve) — no separate lid-nerve knob. A CN III nerve palsy → unilateral
    COMPLETE ptosis (with_cn3_palsy: ptosis + down-and-out eye + blown pupil).
  * Ptosis — CN III NUCLEAR (CCN): follows the shared CN III subnucleus gains
    (g_nucleus). A one-sided nuclear lesion drops the central caudal nucleus drive
    to BOTH lids → bilateral PARTIAL ptosis (asymmetric, ipsi-dominant per
    eyelid_levator_contra_frac); via with_cn3_nuclear_palsy.
  * Ptosis — Horner (Müller): reuses the oculosympathetic gain g_ocular_symp_{L,R}
    (< 1 → mild ptosis alongside the Horner miosis; via with_horner).
  * Lagophthalmos — CN VII (orbicularis): the facial nerve gain g_cn7_{L,R} → 0
    abolishes that eye's blink / closure (Bell's palsy); via with_facial_palsy.

References:
    Loewenfeld IE (1993) The Pupil (near-triad / lid context)
    Standard neuro-ophthalmology (CN III ptosis; Horner Müller ptosis; CN VII
    orbicularis / lagophthalmos)
"""

import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import (
    cn3_nerve_integrity, cn3_nucleus_integrity,
)

N_STATES  = 0   # stateless — the dynamics live in the eyelid plant (eyelid_plant.py)
N_OUTPUTS = 2   # commanded lid closure per eye [L, R]

# ── Fixed anatomical constants (not patient params) ──────────────────────────
PTOSIS_LEVATOR   = 0.7    # resting closure when the levator drive is fully lost (full ptosis)
PTOSIS_MULLER    = 0.15   # extra resting closure when Müller (sympathetic) tone is fully lost (Horner mild ptosis)
DOWNGAZE_GAIN    = 0.4    # lid-follow: closure per (downgaze deg / DOWNGAZE_DEG)
DOWNGAZE_DEG     = 70.0   # downgaze angle mapping to a full unit of lid-follow closure


def _gaze_follow(pitch):
    """Downgaze (pitch < 0) lowers the lid → closure; upgaze contributes 0."""
    return jnp.maximum(0.0, -pitch) / DOWNGAZE_DEG * DOWNGAZE_GAIN


def command(blink_drive, pitch_L, pitch_R, brain_params):
    """Commanded per-eye lid closure [L, R] from posture + blink + gaze-follow.

    Args:
        blink_drive: scalar  central blink command in [0, 1] (conjugate; a 0→1→0
                             pulse during a blink), from the pre-generated schedule
        pitch_L:     scalar  left  eye pitch (deg, + = up); downgaze lowers the lid
        pitch_R:     scalar  right eye pitch (deg, + = up)
        brain_params: BrainParams (reads g_nerve + g_nucleus (CN III → levator),
                                   eyelid_levator_contra_frac, g_cn7_{L,R} (facial
                                   nerve → orbicularis), g_ocular_symp_{L,R} (Müller))

    Returns:
        u_lid: (2,)  commanded lid closure [L, R] → eyelid plant low-pass
    """
    bp = brain_params

    # Two-stage levator tone per eye, both derived from the SHARED CN III gains (no
    # duplicate lid knobs): the central caudal nucleus stage follows the CN III
    # subnucleus gains (g_nucleus) and projects to BOTH lids (ipsi + contra), then
    # the peripheral nerve stage follows the CN III nerve gains (g_nerve) per lid.
    # So a NERVE palsy → unilateral complete ptosis; a NUCLEAR lesion → bilateral
    # (asymmetric, ipsi-dominant) partial ptosis.
    cn3_nerve_L, cn3_nerve_R = cn3_nerve_integrity(bp.g_nerve)
    nuc3_L, nuc3_R           = cn3_nucleus_integrity(bp.g_nucleus)
    c      = bp.eyelid_levator_contra_frac          # fraction of each levator's nuclear drive from CONTRA
    lev_L  = cn3_nerve_L * ((1.0 - c) * nuc3_L + c * nuc3_R)
    lev_R  = cn3_nerve_R * ((1.0 - c) * nuc3_R + c * nuc3_L)

    # Orbicularis oculi (lid closure / blink) is CN VII → gated by the facial nerve
    # gain g_cn7 (0 = lagophthalmos / Bell's palsy).
    def _eye(lev_tone, g_cn7, g_symp, pitch):
        ptosis = PTOSIS_LEVATOR * (1.0 - lev_tone) + PTOSIS_MULLER * (1.0 - g_symp)
        closure = ptosis + _gaze_follow(pitch) + blink_drive * g_cn7
        return jnp.clip(closure, 0.0, 1.0)

    return jnp.array([
        _eye(lev_L, bp.g_cn7_L, bp.g_ocular_symp_L, pitch_L),
        _eye(lev_R, bp.g_cn7_R, bp.g_ocular_symp_R, pitch_R),
    ])
