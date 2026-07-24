"""Pupil plant — iris sphincter/dilator biomechanics (rate-asymmetric).

Low-pass mapping the commanded pupil diameter (mm) from the pupil controller
(``brain_models.pupil``) to the actual pupil diameter (mm).  The iris has two
antagonistic muscles with very different dynamics, giving the hallmark
**fast constriction / slow dilation**:

  * Constriction (pupil shrinking) — the sphincter pupillae (parasympathetic) is
    a strong circular muscle actively contracting → fast, τ ≈ 0.2–0.4 s.
  * Dilation (pupil enlarging) — the weaker dilator pupillae (sympathetic) plus
    passive viscoelastic recoil of the relaxing sphincter → slow, τ ≈ 0.5–1.5 s.

Modelled as a first-order pole whose time constant depends on the direction of
motion: τ_constrict when the command is smaller than the current diameter,
τ_dilate when it is larger.  The two pupils (L, R) are carried as one (2,) state
and stepped elementwise, so each iris relaxes independently (an efferent lesion
on one side produces anisocoria).

The plant receives the two iris NERVE drives (already lesioned in the FCP) and
does the push-pull DECODE to a commanded diameter before the low-pass:

    diam_cmd = clip(pupil_min + dilator − sphincter, pupil_min, pupil_max)

so a blown pupil (sphincter drive gone) settles toward pupil_max and Horner
miosis (dilator drive gone) toward pupil_min — both EMERGE from the antagonist
balance rather than a hardcoded resting size.

Dynamics (elementwise over [L, R]):
    diam_cmd = clip(pupil_min + dilator − sphincter, pupil_min, pupil_max)
    τ  = τ_constrict  where diam_cmd < x   (pupil shrinking, fast)
         τ_dilate     where diam_cmd ≥ x   (pupil enlarging, slow)
    dx = (diam_cmd − x) / τ

State:   x  (2,)   actual pupil diameter (mm) [L, R]
Input:   sphincter (2,)  CN III constrictor nerve drive (mm) [L, R]  (fcp.iris_nerves)
         dilator   (2,)  sympathetic dilator nerve drive (mm) [L, R] (fcp.iris_nerves)
Output:  x         (2,)  current pupil diameter (mm) → readout / avatar / plots

Parameters:
    tau_pupil_constrict (s)  fast sphincter constriction TC; ~0.3 s
    tau_pupil_dilate    (s)  slow (dilator + viscoelastic recoil) TC; ~1.0 s
    pupil_min, pupil_max (mm)  push-pull decode clamp (physiological ~3–8 mm)

References:
    Loewenfeld IE (1993) The Pupil: Anatomy, Physiology, and Clinical Applications
    Ellis CJK (1981) Br J Ophthalmol 65:754  — PLR constriction vs redilation asymmetry
    McDougal & Gamlin (2015) Compr Physiol 5:439
"""

import jax.numpy as jnp

N_STATES  = 2   # [x_L, x_R] — actual pupil diameter (mm), per eye
N_INPUTS  = 4   # sphincter (2,) + dilator (2,) — iris nerve drives (mm), per eye
N_OUTPUTS = 2   # x (mm)       — current pupil diameter per eye


def step(x, sphincter, dilator, pupil_min, pupil_max, tau_constrict, tau_dilate):
    """Single ODE step for the (bilateral) rate-asymmetric iris plant.

    Decodes the antagonist nerve pair to a commanded diameter, then low-passes
    toward it with direction-dependent τ (fast constrict / slow dilate).

    Args:
        x:             (2,)   current pupil diameter (mm) [L, R]
        sphincter:     (2,)   CN III constrictor nerve drive (mm) [L, R]
        dilator:       (2,)   sympathetic dilator nerve drive (mm) [L, R]
        pupil_min:     scalar smallest pupil diameter (mm) — decode floor
        pupil_max:     scalar largest pupil diameter (mm)  — decode ceiling
        tau_constrict: scalar fast constriction TC (s) — used where cmd < x
        tau_dilate:    scalar slow dilation TC (s)     — used where cmd ≥ x

    Returns:
        dx: (2,)  state derivative (mm/s).  The observable (current pupil
                  diameter) IS the state (C = I) — the caller reads x directly.
    """
    diam_cmd = jnp.clip(pupil_min + dilator - sphincter, pupil_min, pupil_max)
    tau = jnp.where(diam_cmd < x, tau_constrict, tau_dilate)   # fast in, slow out
    dx  = (diam_cmd - x) / tau
    return dx
