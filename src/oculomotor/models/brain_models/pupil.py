"""Pupil control — pupillary light reflex + near-response constriction.

Stateless brain module (cf. tvor.py): computes the commanded pupil DIAMETER
(mm) from two efferent drives converging on the Edinger-Westphal (EW) nucleus →
ciliary ganglion → iris sphincter.  The commanded diameter is low-passed by the
iris plant (``plant_models.pupil_plant``) to give the actual pupil size.

The two pupils (L, R) are computed SEPARATELY (one iris plant each), so
anisocoria — the hallmark of an efferent (CN III / iris) lesion — can appear.
Each pupil's constriction subtracts from a per-eye resting (dark) diameter:

  1. Pupillary light reflex (PLR) — CONSENSUAL: the pretectal olivary nucleus
     sums BOTH eyes' afferent luminance and projects to BOTH Edinger-Westphal
     nuclei, so the light DRIVE is identical for the two pupils.  A monocular
     afferent lesion (RAPD) therefore does NOT cause anisocoria; it shows up as
     a weak response when only the affected eye is lit (swinging-flashlight).
     ``light_drive`` is the scalar consensual drive — the pretectal binocular sum
     (with the per-eye afferent gains / RAPD already applied) computed in
     ``perception_cyclopean`` (cyc.light_drive), from each eye's retinal
     luminance afferent (retina.State.luminance).

  2. Pupillary near response — central, symmetric (near triad, accommodation-
     linked; Myers & Stark 1990).  Same drive to both pupils.

    light  = g_light_reflex · K_pupil_light · clip(Σ_eye g_afferent·lum, 0, 1)
    near   = K_pupil_near · accom
    rest_i = pupil_min + g_symp_i · (pupil_baseline − pupil_min)     (per eye)
    diam_i = clip(rest_i − cn3_i · (light + near), pupil_min, pupil_max)

Lesion gains: cn3_{L,R} = CN III nerve integrity (from g_nerve, NOT a separate
knob — parasympathetics travel with the nerve); g_pupil_afferent_{L,R} (afferent
limb / optic nerve → RAPD); g_ocular_symp_{L,R} (oculosympathetic tone → Horner
miosis when < 1); g_pupil_light_reflex is the bilateral pretectal relay (Argyll
Robertson).

Lesions (helpers in sim.simulator):
    * CN III efferent (blown pupil), per eye: a CN III nerve palsy zeroes that
      side's cn3 integrity → abolishes BOTH its light and near constriction →
      an ipsilateral fixed dilated pupil (anisocoria). The parasympathetics travel
      with CN III, so this follows the SHARED oculomotor nerve gains (g_nerve) —
      there is NO separate pupil-nerve knob. Any CN III nerve palsy (however set,
      e.g. with_cn3_palsy) blows that side's pupil automatically.
    * Afferent defect (RAPD / Marcus Gunn), per eye: g_pupil_afferent_{L,R} < 1
      weakens that eye's contribution to the CONSENSUAL light drive. No
      anisocoria at rest — revealed by the swinging-flashlight test.
    * Pretectal (Argyll Robertson / Parinaud): g_pupil_light_reflex = 0 removes
      the LIGHT reflex centrally (both pupils) while sparing near — light-near
      dissociation.
    * Sympathetic (Horner miosis), per eye: g_ocular_symp_{L,R} < 1 lowers that
      pupil's resting (dark) diameter via ``with_horner(side=...)`` — small
      ipsilateral pupil, reactions preserved.

Units:
    All diameters in mm (physiological ~3–8 mm).  ``lum_afferent`` is a
    normalised luminance in ~[0, 1]; ``accom`` is accommodation in diopters (D).

References:
    Loewenfeld IE (1993) The Pupil — comprehensive reference
    McDougal & Gamlin (2015) Compr Physiol 5:439 — PLR + near-response circuitry
    Myers GA, Stark L (1990) Ophthalmic Physiol Opt 10:360 — near-response pupil
"""

import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import cn3_nerve_integrity

N_STATES  = 0   # stateless — the dynamics live in the iris plants (pupil_plant.py)
N_OUTPUTS = 2   # commanded pupil diameter (mm), per eye [L, R]


def command(light_drive, accom_level, brain_params):
    """Commanded per-eye pupil diameter (mm) from light reflex + near response.

    Args:
        light_drive:  scalar  consensual pretectal light drive (~[0,1]) — the
                              binocular afferent sum (with the monocular afferent
                              gains / RAPD already applied), from
                              perception_cyclopean (cyc.light_drive)
        accom_level:  scalar  accommodation level (D) — acts.va.acc_fast + acc_slow
        brain_params: BrainParams (reads pupil_baseline, K_pupil_light,
                                   K_pupil_near, pupil_min, pupil_max, g_nerve
                                   (CN III efferent), g_ocular_symp_{L,R},
                                   g_pupil_light_reflex)

    Returns:
        u_pupil: (2,)  commanded pupil diameter (mm) [L, R] → iris plant low-pass
    """
    bp = brain_params

    # Efferent parasympathetic (pupilloconstrictor) integrity per eye — travels
    # with CN III, so it follows the SHARED oculomotor nerve gains (no separate
    # pupil-nerve knob). A CN III palsy → that side's pupil is blown → anisocoria.
    cn3_L, cn3_R = cn3_nerve_integrity(bp.g_nerve)

    # ── Consensual light drive (shared by both pupils) ────────────────────────
    # The binocular afferent sum (with RAPD gains) is the pretectal combination,
    # now computed in perception_cyclopean (cyc.light_drive). Here we only apply
    # the central pretectal-relay integrity (g_pupil_light_reflex — Argyll Robertson
    # when 0) and the constriction gain; the drive is shared by BOTH pupils.
    light = bp.g_pupil_light_reflex * bp.K_pupil_light * light_drive

    # ── Near drive (central, symmetric) ───────────────────────────────────────
    near = bp.K_pupil_near * accom_level

    # ── Per-eye output: resting (sympathetic) size − ipsilateral parasympathetic
    # (CN III) constriction of the shared light + near drive ──────────────────
    def _eye(cn3, g_symp):
        rest = bp.pupil_min + g_symp * (bp.pupil_baseline - bp.pupil_min)
        diam = rest - cn3 * (light + near)
        return jnp.clip(diam, bp.pupil_min, bp.pupil_max)

    return jnp.array([_eye(cn3_L, bp.g_ocular_symp_L),
                      _eye(cn3_R, bp.g_ocular_symp_R)])
