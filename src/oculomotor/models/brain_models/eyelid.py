"""Eyelid control — levator / Müller / orbicularis drives (blinks + posture).

Stateless brain module (cf. pupil.py): emits the three RAW upper-lid muscle
drives per eye — WITHOUT lesions — and the FCP (``fcp.eyelid_nerves``) applies the
peripheral nerve lesions, exactly like the extraocular and iris nerves.  The
eyelid plant (``plant_models.eyelid_plant``) then decodes the antagonist balance
into a lid closure (0 = open, 1 = fully closed) and low-passes it.

The lid is an antagonist system with three muscles:
  1. Levator palpebrae superioris (CN III, somatic) — the main OPENER; holds the
     lid up.  Loss → ptosis.  Commanded fully open at rest and RELAXED on downgaze
     so the lid follows the eye down (efference-driven, not proprioceptive).
  2. Müller's muscle (sympathetic) — a minor tonic OPENER; loss → mild (Horner)
     ptosis.  Commanded as a constant tone.
  3. Orbicularis oculi (CN VII, facial) — the CLOSER; a central blink command
     drives it to snap the lid shut.  Loss → lagophthalmos (Bell's palsy).

    levator_i     = clip(1 − levator_relax(pitch_i), 0, 1)   # opener tone ∈ [0,1]
    muller_i      = 1                                          # opener tone ∈ [0,1]
    orbicularis_i = blink_drive                                # closer drive ∈ [0,1]

The two lids (L, R) are computed separately so a unilateral lesion affects only
its side.  All lesion gains — CN III nerve + nucleus (levator, → ptosis),
sympathetic (Müller, → Horner ptosis), CN VII (orbicularis, → lagophthalmos) —
are applied downstream in ``fcp.eyelid_nerves``; NONE live here.  Nuclear vs nerve
CN III lesions (bilateral partial vs unilateral complete ptosis) and the
contra-levator mixing likewise live in the FCP.

References:
    Loewenfeld IE (1993) The Pupil (near-triad / lid context)
    Standard neuro-ophthalmology (CN III ptosis; Horner Müller ptosis; CN VII
    orbicularis / lagophthalmos)
"""

import jax.numpy as jnp

N_STATES  = 0   # stateless — the dynamics live in the eyelid plant (eyelid_plant.py)
N_OUTPUTS = 6   # three per-eye lid muscle drives: levator + Müller + orbicularis

# ── Gaze-follow (levator relaxation on downgaze) — a central command, not a lesion.
# The upper lid tracks the eye ~1:1 in downgaze, keeping its margin near the upper
# limbus: the limbus drops ~R·sin(θ) (R ≈ 12 mm) against ~10 mm of full lid travel,
# i.e. closure ≈ 0.02 per degree. With the plant's PTOSIS_LEVATOR droop-per-loss that
# is closure = PTOSIS_LEVATOR · min(1, θ / DOWNGAZE_DEG · DOWNGAZE_RELAX) → 0.24 at
# 12°, 0.6 at 30°, saturating at 0.7 from 35°. (Was 0.57 / 70° → only 0.07 at 12°,
# leaving a band of sclera above the iris in downgaze, like lid lag.)
DOWNGAZE_RELAX = 1.0    # levator relaxation fraction at a full unit of downgaze
DOWNGAZE_DEG   = 35.0   # downgaze angle mapping to a full unit of levator relaxation


def _levator_relax(pitch):
    """Downgaze (pitch < 0) relaxes the levator → lid lowers; upgaze contributes 0."""
    return jnp.maximum(0.0, -pitch) / DOWNGAZE_DEG * DOWNGAZE_RELAX


def command(blink_drive, pitch_L, pitch_R, brain_params):
    """Raw per-eye lid muscle drives (pre-lesion): levator, Müller, orbicularis.

    Args:
        blink_drive: scalar  central blink command in [0, 1] (conjugate; a 0→1→0
                             pulse during a blink), from the pre-generated schedule
        pitch_L:     scalar  left  eye pitch (deg, + = up); downgaze relaxes the levator
        pitch_R:     scalar  right eye pitch (deg, + = up)
        brain_params: BrainParams (unused here — kept for a uniform command()
                                   signature; all lesion gains live in the FCP)

    Returns:
        levator:     (2,)  levator opener tone ∈ [0,1] [L, R] (1 = fully open)
        muller:      (2,)  Müller opener tone ∈ [0,1]  [L, R] (constant)
        orbicularis: (2,)  orbicularis closer drive ∈ [0,1] [L, R] (central blink)
    """
    del brain_params   # no lesions here — applied in fcp.eyelid_nerves
    # Levator (CN III) commanded opening tone: fully open at rest, relaxed on
    # downgaze so the lid follows the eye down.
    levator = jnp.clip(
        1.0 - jnp.array([_levator_relax(pitch_L), _levator_relax(pitch_R)]),
        0.0, 1.0)
    # Müller (sympathetic) tonic opening — fully commanded (constant).
    muller = jnp.ones(2)
    # Orbicularis (CN VII) commanded closure — central blink (conjugate).
    orbicularis = jnp.full(2, blink_drive)
    return levator, muller, orbicularis
