"""Eyelid plant — upper-lid (levator / Müller / orbicularis) biomechanics.

Decodes the three antagonist lid-muscle drives (from ``brain_models.eyelid`` via
``fcp.eyelid_nerves``, already lesioned) into a lid closure, then low-passes it to
the actual closure per eye.  Like the iris, the lid has fast-closing / slower-
opening dynamics: the orbicularis snaps the lid shut in a blink (~50 ms) while
levator-driven re-opening is a bit slower (~100–150 ms).  Modelled as a first-
order pole whose time constant depends on the direction of motion.

Push-pull decode — the lid droops (closes) as the OPENER tones fall and the
CLOSER drive rises:

    closure = clip(  PTOSIS_LEVATOR·(1 − levator)      # opener deficit → ptosis
                   + PTOSIS_MULLER·(1 − muller)        # Müller deficit → mild ptosis
                   + orbicularis,                      # active blink closure
                   0, 1)

so a CN III palsy (levator → 0) settles toward PTOSIS_LEVATOR closure, Horner
(Müller → 0) adds PTOSIS_MULLER, and a CN VII palsy (orbicularis → 0) abolishes
the blink — all EMERGE from the antagonist balance rather than a ptosis formula
in the controller.

The two eyelids (L, R) are carried as one (2,) state and stepped elementwise, so
each lid moves independently.  Closure convention: 0 = open, 1 = fully closed.

State:   x  (2,)   actual lid closure [L, R], in [0, 1]
Input:   levator     (2,)  levator opener tone ∈ [0,1] [L, R]  (fcp.eyelid_nerves)
         muller      (2,)  Müller opener tone ∈ [0,1]  [L, R]  (fcp.eyelid_nerves)
         orbicularis (2,)  orbicularis closer drive ∈ [0,1] [L, R] (fcp.eyelid_nerves)
Output:  x           (2,)  current lid closure → readout / avatar / plots

Parameters:
    tau_lid_close (s)  fast orbicularis closing TC; ~0.02 s
    tau_lid_open  (s)  slower levator opening TC;   ~0.06 s
"""

import jax.numpy as jnp

N_STATES  = 2   # [x_L, x_R] — actual lid closure per eye
N_INPUTS  = 6   # levator (2,) + Müller (2,) + orbicularis (2,) — lid muscle drives
N_OUTPUTS = 2   # x — current closure per eye

# Fixed lid mechanics (not patient params): blink snaps shut fast, eases open slower.
TAU_CLOSE = 0.02   # s — fast closing TC (used where the lid is closing)
TAU_OPEN  = 0.06   # s — slower opening TC (used where the lid is opening)

# Push-pull decode gains — how far the lid droops when an OPENER tone is fully lost.
PTOSIS_LEVATOR = 0.7    # closure from full levator (CN III) loss — full ptosis
PTOSIS_MULLER  = 0.15   # extra closure from full Müller (sympathetic) loss — Horner mild ptosis


def decode(levator, muller, orbicularis):
    """Push-pull decode: antagonist muscle drives → commanded lid closure ∈ [0,1].

    Shared by ``step`` and the simulator's resting-lid initial condition so the
    formula lives in one place.
    """
    return jnp.clip(PTOSIS_LEVATOR * (1.0 - levator)
                    + PTOSIS_MULLER * (1.0 - muller)
                    + orbicularis, 0.0, 1.0)


def step(x, levator, muller, orbicularis, tau_close=TAU_CLOSE, tau_open=TAU_OPEN):
    """Single ODE step for the (bilateral) rate-asymmetric eyelid plant.

    Decodes the antagonist muscle drives to a commanded closure, then low-passes
    toward it with direction-dependent τ (fast close / slow open).

    Args:
        x:           (2,)   current lid closure [L, R]
        levator:     (2,)   levator opener tone ∈ [0,1] [L, R]
        muller:      (2,)   Müller opener tone ∈ [0,1]  [L, R]
        orbicularis: (2,)   orbicularis closer drive ∈ [0,1] [L, R]
        tau_close:   scalar fast closing TC (s) — used where cmd > x
        tau_open:    scalar slower opening TC (s) — used where cmd ≤ x

    Returns:
        dx: (2,)  state derivative (1/s).  The observable (current lid closure)
                  IS the state (C = I) — the caller reads x directly.
    """
    closure_cmd = decode(levator, muller, orbicularis)
    tau = jnp.where(closure_cmd > x, tau_close, tau_open)   # snap shut, ease open
    dx  = (closure_cmd - x) / tau
    return dx
