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

Dynamics (elementwise over [L, R]):
    τ  = τ_constrict  where u_pupil < x   (pupil shrinking, fast)
         τ_dilate     where u_pupil ≥ x   (pupil enlarging, slow)
    dx = (u_pupil − x) / τ

State:   x  (2,)   actual pupil diameter (mm) [L, R]
Input:   u_pupil  (2,)  commanded pupil diameter (mm) [L, R] from pupil.command()
Output:  x        (2,)  current pupil diameter (mm) → readout / avatar / plots

Parameters:
    tau_pupil_constrict (s)  fast sphincter constriction TC; ~0.3 s
    tau_pupil_dilate    (s)  slow (dilator + viscoelastic recoil) TC; ~1.0 s

References:
    Loewenfeld IE (1993) The Pupil: Anatomy, Physiology, and Clinical Applications
    Ellis CJK (1981) Br J Ophthalmol 65:754  — PLR constriction vs redilation asymmetry
    McDougal & Gamlin (2015) Compr Physiol 5:439
"""

import jax.numpy as jnp

N_STATES  = 2   # [x_L, x_R] — actual pupil diameter (mm), per eye
N_INPUTS  = 2   # u_pupil (mm) — commanded diameter per eye
N_OUTPUTS = 2   # x (mm)       — current pupil diameter per eye


def step(x, u_pupil, tau_constrict, tau_dilate):
    """Single ODE step for the (bilateral) rate-asymmetric iris plant.

    Args:
        x:             (2,)   current pupil diameter (mm) [L, R]
        u_pupil:       (2,)   commanded pupil diameter (mm) [L, R]
        tau_constrict: scalar fast constriction TC (s) — used where u < x
        tau_dilate:    scalar slow dilation TC (s)     — used where u ≥ x

    Returns:
        dx:      (2,)   state derivative (mm/s)
        x_out:   (2,)   current pupil diameter (mm)
    """
    tau = jnp.where(u_pupil < x, tau_constrict, tau_dilate)   # fast in, slow out
    dx  = (u_pupil - x) / tau
    return dx, x
