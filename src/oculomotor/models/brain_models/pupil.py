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
     ``lum_afferent`` is the (2,) per-eye low-passed luminance from
     ``sensory_models.luminance``; per-eye afferent gains model RAPD.

  2. Pupillary near response — central, symmetric (near triad, accommodation-
     linked; Myers & Stark 1990).  Same drive to both pupils.

    light  = g_light_reflex · K_pupil_light · clip(Σ_eye g_afferent·lum, 0, 1)
    near   = K_pupil_near · accom
    rest_i = pupil_min + g_symp_i · (pupil_baseline − pupil_min)     (per eye)
    diam_i = clip(rest_i − g_cn3_i · (light + near), pupil_min, pupil_max)

Per-eye lesion gains: g_pupil_cn3_{L,R} (efferent parasympathetic, ipsilateral
CN III), g_pupil_afferent_{L,R} (afferent limb / optic nerve → RAPD), and
g_pupil_symp_{L,R} (sympathetic dilator tone → Horner miosis when < 1).
g_pupil_light_reflex is the bilateral pretectal relay (Argyll Robertson).

Lesions (all knobs in BrainParams; helpers in sim.simulator):
    * CN III efferent (blown pupil), per eye: g_pupil_cn3_{L,R} → 0 gates that
      pupil's efferent path, abolishing BOTH its light and near constriction →
      an ipsilateral fixed dilated pupil (anisocoria). Shares the CN III lesion
      with the eye muscles via ``with_cn3_palsy(side=...)`` (down-and-out eye +
      blown pupil on that side); an isolated 0 is a tonic (Adie) pupil.
    * Afferent defect (RAPD / Marcus Gunn), per eye: g_pupil_afferent_{L,R} < 1
      weakens that eye's contribution to the CONSENSUAL light drive. No
      anisocoria at rest — revealed by the swinging-flashlight test.
    * Pretectal (Argyll Robertson / Parinaud): g_pupil_light_reflex = 0 removes
      the LIGHT reflex centrally (both pupils) while sparing near — light-near
      dissociation.
    * Sympathetic (Horner miosis), per eye: g_pupil_symp_{L,R} < 1 lowers that
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

N_STATES  = 0   # stateless — the dynamics live in the iris plants (pupil_plant.py)
N_OUTPUTS = 2   # commanded pupil diameter (mm), per eye [L, R]


def command(lum_afferent, accom_level, brain_params):
    """Commanded per-eye pupil diameter (mm) from light reflex + near response.

    Args:
        lum_afferent: (2,)    per-eye afferent luminance [L, R] (normalised) —
                              from sensory_out.luminance (PLR-latency low-passed)
        accom_level:  scalar  accommodation level (D) — acts.va.acc_fast + acc_slow
        brain_params: BrainParams (reads pupil_baseline, K_pupil_light,
                                   K_pupil_near, pupil_min, pupil_max, and the
                                   per-eye lesion gains g_pupil_cn3_{L,R} /
                                   g_pupil_afferent_{L,R} / g_pupil_symp_{L,R}
                                   plus the bilateral g_pupil_light_reflex)

    Returns:
        u_pupil: (2,)  commanded pupil diameter (mm) [L, R] → iris plant low-pass
    """
    bp = brain_params

    # ── Consensual light drive (shared by both pupils) ────────────────────────
    # Bilateral pretectum sums the two eyes' afferents (each scaled by its
    # afferent-limb integrity → RAPD) then projects to BOTH EW nuclei. Clip the
    # summed drive so a single fully-lit intact eye already saturates the reflex
    # (monocular ≈ binocular constriction); a monocular afferent defect only
    # shows when THAT eye is the one being lit (swinging-flashlight).
    lum_sum = (bp.g_pupil_afferent_L * lum_afferent[0]
               + bp.g_pupil_afferent_R * lum_afferent[1])
    light = bp.g_pupil_light_reflex * bp.K_pupil_light * jnp.clip(lum_sum, 0.0, 1.0)

    # ── Near drive (central, symmetric) ───────────────────────────────────────
    near = bp.K_pupil_near * accom_level

    # ── Per-eye output: resting (sympathetic) size − ipsilateral parasympathetic
    # (CN III) constriction of the shared light + near drive ──────────────────
    def _eye(g_cn3, g_symp):
        rest = bp.pupil_min + g_symp * (bp.pupil_baseline - bp.pupil_min)
        diam = rest - g_cn3 * (light + near)
        return jnp.clip(diam, bp.pupil_min, bp.pupil_max)

    return jnp.array([_eye(bp.g_pupil_cn3_L, bp.g_pupil_symp_L),
                      _eye(bp.g_pupil_cn3_R, bp.g_pupil_symp_R)])
