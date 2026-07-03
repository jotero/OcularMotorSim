"""Eyelid plant — upper-lid (levator / orbicularis) biomechanics.

Low-pass mapping the commanded lid closure (from ``brain_models.eyelid``) to the
actual lid closure, per eye.  Like the iris, the lid has fast-closing / slower-
opening dynamics: the orbicularis snaps the lid shut in a blink (~50 ms) while
levator-driven re-opening is a bit slower (~100–150 ms).  Modelled as a first-
order pole whose time constant depends on the direction of motion.

The two eyelids (L, R) are carried as one (2,) state and stepped elementwise, so
each lid moves independently (a unilateral levator / orbicularis lesion affects
only its side).

Closure convention: 0 = eye fully open, 1 = fully closed.

Dynamics (elementwise over [L, R]):
    τ  = τ_close  where u_lid > x   (lid closing, fast)
         τ_open   where u_lid ≤ x   (lid opening, slower)
    dx = (u_lid − x) / τ

State:   x  (2,)   actual lid closure [L, R], in [0, 1]
Input:   u_lid  (2,)  commanded lid closure [L, R] from eyelid.command()
Output:  x        (2,)  current lid closure → readout / avatar / plots

Parameters:
    tau_lid_close (s)  fast orbicularis closing TC; ~0.02 s
    tau_lid_open  (s)  slower levator opening TC;   ~0.06 s
"""

import jax.numpy as jnp

N_STATES  = 2   # [x_L, x_R] — actual lid closure per eye
N_INPUTS  = 2   # u_lid — commanded closure per eye
N_OUTPUTS = 2   # x — current closure per eye

# Fixed lid mechanics (not patient params): blink snaps shut fast, eases open slower.
TAU_CLOSE = 0.02   # s — fast closing TC (used where the lid is closing)
TAU_OPEN  = 0.06   # s — slower opening TC (used where the lid is opening)


def step(x, u_lid, tau_close=TAU_CLOSE, tau_open=TAU_OPEN):
    """Single ODE step for the (bilateral) rate-asymmetric eyelid plant.

    Args:
        x:         (2,)   current lid closure [L, R]
        u_lid:     (2,)   commanded lid closure [L, R]
        tau_close: scalar fast closing TC (s) — used where u > x
        tau_open:  scalar slower opening TC (s) — used where u ≤ x

    Returns:
        dx:      (2,)   state derivative (1/s)
        x_out:   (2,)   current lid closure
    """
    tau = jnp.where(u_lid > x, tau_close, tau_open)   # snap shut, ease open
    dx  = (u_lid - x) / tau
    return dx, x
