"""Pupil plant — iris sphincter/dilator biomechanics.

First-order low-pass mapping the commanded pupil diameter (mm) from the pupil
controller (``brain_models.pupil``) to the actual pupil diameter (mm).  The iris
musculature (sphincter pupillae + dilator pupillae) behaves as a viscoelastic
element; the pupillary light reflex settles over ~0.3–0.8 s, well-approximated
by a single first-order pole with τ_pupil ≈ 0.4 s (Loewenfeld 1993).

This mirrors ``accommodation_plant.py`` — a first-order ocular plant low-passing
a neural command toward its commanded value.  The two pupils (L, R) are carried
as one (2,) state and stepped elementwise, so each iris relaxes independently
(an efferent lesion on one side produces anisocoria).

Dynamics (elementwise over [L, R]):
    dx = (u_pupil − x) / τ_pupil

State:   x  (2,)   actual pupil diameter (mm) [L, R]
Input:   u_pupil  (2,)  commanded pupil diameter (mm) [L, R] from pupil.command()
Output:  x        (2,)  current pupil diameter (mm) → readout / avatar / plots

Parameters:
    tau_pupil (s)  iris sphincter/dilator TC; ~0.4 s [Loewenfeld 1993]

References:
    Loewenfeld IE (1993) The Pupil: Anatomy, Physiology, and Clinical Applications
    McDougal & Gamlin (2015) Compr Physiol 5:439
"""

N_STATES  = 2   # [x_L, x_R] — actual pupil diameter (mm), per eye
N_INPUTS  = 2   # u_pupil (mm) — commanded diameter per eye
N_OUTPUTS = 2   # x (mm)       — current pupil diameter per eye


def step(x, u_pupil, tau_pupil):
    """Single ODE step for the (bilateral) iris plant.

    Args:
        x:         (2,)   current pupil diameter (mm) [L, R]
        u_pupil:   (2,)   commanded pupil diameter (mm) [L, R]
        tau_pupil: scalar iris sphincter/dilator TC (s)

    Returns:
        dx:      (2,)   state derivative (mm/s)
        x_out:   (2,)   current pupil diameter (mm)
    """
    dx = (u_pupil - x) / tau_pupil
    return dx, x
