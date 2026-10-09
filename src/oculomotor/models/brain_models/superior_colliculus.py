"""Superior colliculus — 2-D place-code map of target position (SC v1, SHADOW MODE).

See PLAN_SC.md. v1 receives the PREDICTED current target position — perception's
delayed cyclopean position plus the cerebellum's predicted shift of it during the visual
delay (eye and target motion) —
renders it as a bump of activity on a log-polar map of each colliculus, and reads back a
decoded target position. By default nothing downstream
reads it (shadow mode); BrainParams.sc_drives_sg = 1 makes its readout drive the target
working memory → saccade generator instead of the cyclopean target position.

Why a place code: low-passing a rate-coded position makes a target step glide through
every intermediate position. On a map, filtering only changes the AMPLITUDE of activity
— the bump at A fades while the bump at B rises — so the decoded position can switch
fast. Position (bump location) and visibility (bump amplitude) are also separate, so a
brief flash is a small bump in the right place rather than a full-size signal at a
fraction of the eccentricity.

Map geometry — Ottes, Van Gisbergen & Eggermont (1986)
──────────────────────────────────────────────────────
    w = ln((z + A) / A),     z = x + i·y   (deg; x = contralateral horizontal, y = up)
    u = B_u · Re(w)   (mm, rostro-caudal),    v = B_v · Im(w)   (mm, medio-lateral)
    inverse:  z = A · (exp(w) − 1)
The LEFT colliculus codes rightward targets (x = +yaw); the RIGHT codes leftward
(x = −yaw). Same grid in both, per colliculus 19 (Re w) × 17 (Im w), ~0.35 mm spacing:
    Re(w) ∈ [−1.0, ln(1 + 100/A)]   → u ∈ [−1.40, 4.95] mm   (out to ~100° eccentricity)
    Im(w) ∈ [−π/2, π/2]             → v ∈ [−2.83, 2.83] mm   (the whole hemifield)
Input coordinates: [yaw, pitch] are the ROTATION-VECTOR (axis-angle) components of the
rotation taking the line of sight to the target direction (retina.world_to_retina), used
directly as z = yaw + i·pitch. Its magnitude |z| is the target's true angular eccentricity
and arg z its true direction (meridian), so z is exactly the polar form (eccentricity,
direction) on which Ottes et al. define the map — no approximation on obliques.
The map extends well past the largest target we decode (~50°) because a bump truncated at
the map edge is read back biased inward. Re(w) < 0 is the rostral overlap: a few degrees of
the ipsilateral field, so small (microsaccade-scale) errors are represented in both
colliculi without truncation.

Dynamics (per unit; membrane x, rate r = max(0, x))
───────────────────────────────────────────────────
    τ dx/dt = −x + a·G(u − u_t, v − v_t)                    input bump
                 + g_exc · K_exc r                          local excitation (same colliculus)
                 − g_inh · S_same − g_cross · S_other       inhibition (S = Σr / M_bump)
    input: location = predicted current target position (cyc.target_pos +
           acts.cb.target_shift, summed in brain_model), amplitude a = cyclopean target
           visibility × the cerebellar saccadic-suppression gate
Sub-threshold excitation (g_exc < 1): activity decays when the input goes away — no
persistent bump (memory is not part of v1).

Saccades
────────
The visual position arrives ~65 ms late. The cerebellum's position forward model
predicts how much it shifts during that delay — the eye displacement still in flight
(efference copy) against the target's own motion — and that shift is added to the
perceived position, so the input bump sits at the CURRENT target position: after a
saccade the map reads ~0 at once instead of the stale pre-saccadic error. During the saccade
the predicted position sweeps with the eye, and so does the bump (a "moving hill" — not seen
in the monkey SC, Soetedjo, Kaneko & Fuchs 2002); saccadic suppression (the cerebellar gate
on the input amplitude) attenuates it.

Readout — thresholded population average
────────────────────────────────────────
Only units firing above _R_THRESH ("firing a lot") enter the average, with weights
q = max(0, r − _R_THRESH). A decaying old bump drops out as soon as it falls below
threshold, and a rising new one counts as soon as its peak crosses it, so the readout
is not dragged by weak trailing or leading activity. Per colliculus the weighted mean in
map coordinates is mapped back to visual space; the two colliculi are combined with
weights m⁴ (m = thresholded mass), which favours the colliculus holding the complete bump
over a truncated one near the vertical meridian.
NO UNIT ABOVE THRESHOLD: the map holds no target. `valid` → 0 and `pos` → 0 (the
weighted mean of an empty set is guarded to the map origin, Re w = Im w = 0, which IS
the fovea), so downstream treats it like "no target visible". `strength` = total
(unthresholded) activity in units of one full bump.

Parameters are provisional module constants for v1 (tuning is step 3 of the plan); they
move to BrainParams when the SC becomes the default target path.
"""

from typing import NamedTuple

import numpy as np
import jax.numpy as jnp


# ── Map geometry (Ottes et al. 1986) ─────────────────────────────────────────
_A   = 3.0        # deg   — foveal magnification offset
_B_U = 1.4        # mm    — rostro-caudal scale
_B_V = 1.8        # mm/rad — medio-lateral scale

_RE_W_MIN = -1.0                          # rostral overlap into the ipsilateral field
_RE_W_MAX = float(np.log(1.0 + 100.0 / _A))   # ~100° eccentricity
_NU, _NV  = 19, 17
N_PER_SIDE = _NU * _NV                     # 323

_re = np.linspace(_RE_W_MIN, _RE_W_MAX, _NU)
_im = np.linspace(-np.pi / 2, np.pi / 2, _NV)
_RE, _IM = [g.ravel() for g in np.meshgrid(_re, _im, indexing='ij')]
_U, _V = _B_U * _RE, _B_V * _IM            # unit centres (mm), identical for both colliculi

# ── Dynamics constants (provisional, v1) ─────────────────────────────────────
_TAU       = 0.015   # s   — membrane / visual-layer time constant
_SIGMA_IN  = 0.5     # mm  — input bump width
_SIGMA_EXC = 0.5     # mm  — local excitation kernel width
_G_EXC     = 0.5     # local excitation gain (< 1: no self-sustained bump)
_G_INH     = 0.25    # inhibition from the same colliculus
_G_CROSS   = 0.25    # inhibition from the other colliculus (intercollicular)
_W_POWER   = 4.0     # readout: colliculus weights m^p
_R_THRESH  = 0.1     # readout: only rates above this enter the average (full bump peak ≈ 1;
                     #   0.1 still detects a 10 ms flash, which peaks at ~0.2 after the retina)
_M_VALID   = 0.02    # readout: thresholded mass (in bumps) at which valid = 0.5

N_STATES = 2 * N_PER_SIDE                  # 646 map units

# Local excitation kernel: Gaussian of map distance, row-normalised so every unit
# (edge units included) receives a unit-gain weighted average of its neighbours.
_d2 = (_U[:, None] - _U[None, :]) ** 2 + (_V[:, None] - _V[None, :]) ** 2
_K = np.exp(-_d2 / (2 * _SIGMA_EXC ** 2))
_K_EXC = jnp.asarray(_K / _K.sum(axis=1, keepdims=True), dtype=jnp.float32)

# Activity of one complete unit-amplitude input bump (normaliser for S and strength).
_uc, _vc = 0.5 * (_U.min() + _U.max()), 0.0
_M_BUMP = float(np.exp(-((_U - _uc) ** 2 + (_V - _vc) ** 2) / (2 * _SIGMA_IN ** 2)).sum())

# Horizontal meridian (Im w = 0) of each colliculus, rostral → caudal — for space-time plots.
MERIDIAN_IDX = np.flatnonzero(np.isclose(_IM, 0.0))                 # (19,) unit indices
MERIDIAN_U   = _U[MERIDIAN_IDX]                                      # (19,) map position (mm)
MERIDIAN_ECC = _A * (np.exp(_RE[MERIDIAN_IDX]) - 1.0)                # (19,) contralateral ecc (deg;
                                                                     #   < 0 = ipsilateral overlap)
_U_J, _V_J   = jnp.asarray(_U, jnp.float32), jnp.asarray(_V, jnp.float32)
_RE_J, _IM_J = jnp.asarray(_RE, jnp.float32), jnp.asarray(_IM, jnp.float32)


# ── State + registries ───────────────────────────────────────────────────────

class State(NamedTuple):
    """SC map membranes, one vector per colliculus (unit order: Re(w) major, Im(w) minor)."""
    L: jnp.ndarray   # (323,) left colliculus  — codes RIGHTWARD targets
    R: jnp.ndarray   # (323,) right colliculus — codes LEFTWARD targets


class Activations(NamedTuple):
    """SC firing rates (rectified membranes) — level-1 neurons."""
    L: jnp.ndarray   # (323,)
    R: jnp.ndarray   # (323,)


class Decoded(NamedTuple):
    """Population readout of the map."""
    pos:      jnp.ndarray   # (2,) decoded target position [yaw, pitch] (deg, retinal); 0 if not valid
    valid:    jnp.ndarray   # scalar ∈ [0, 1) — some unit fires above threshold
    strength: jnp.ndarray   # scalar — total activity, in units of one full bump


def rest_state():
    """Silent map."""
    return State(L=jnp.zeros(N_PER_SIDE), R=jnp.zeros(N_PER_SIDE))


def read_activations(state):
    """Rates = max(0, membrane)."""
    return Activations(L=jnp.maximum(state.L, 0.0), R=jnp.maximum(state.R, 0.0))


# ── Geometry helpers ─────────────────────────────────────────────────────────

def _to_map(yaw, pitch, side):
    """Visual position (deg) → map coordinates (mm) of one colliculus (side +1 = left SC).

    yaw, pitch are rotation-vector components (see the module docstring): z = x + i·y
    has |z| = true eccentricity and arg z = true direction (the Ottes polar form).
    """
    qx = side * yaw + _A
    qy = pitch
    w_re = jnp.log(jnp.sqrt(qx ** 2 + qy ** 2 + 1e-12) / _A)
    w_im = jnp.arctan2(qy, qx)
    return _B_U * w_re, _B_V * w_im


def _render(pos, side):
    """Unit-amplitude Gaussian input bump on one colliculus for target position pos=[yaw, pitch]."""
    u_t, v_t = _to_map(pos[0], pos[1], side)
    return jnp.exp(-((_U_J - u_t) ** 2 + (_V_J - v_t) ** 2) / (2 * _SIGMA_IN ** 2))


def _decode_side(r, side):
    """Thresholded population average of one colliculus, mapped back to visual space.

    Returns the decoded position and the thresholded mass m (in bumps). With no unit
    above threshold the guarded mean is the map origin, i.e. position 0.
    """
    q = jnp.maximum(r - _R_THRESH, 0.0)
    m = jnp.sum(q)
    w_re = jnp.sum(q * _RE_J) / (m + 1e-6)
    w_im = jnp.sum(q * _IM_J) / (m + 1e-6)
    x = _A * (jnp.exp(w_re) * jnp.cos(w_im) - 1.0)
    y = _A * jnp.exp(w_re) * jnp.sin(w_im)
    return jnp.array([side * x, y]), m / _M_BUMP


def decode_states(acts):
    """Decoded target position, validity and strength from the two colliculi (m⁴-weighted)."""
    p_L, m_L = _decode_side(acts.L, +1.0)
    p_R, m_R = _decode_side(acts.R, -1.0)
    w_L, w_R = m_L ** _W_POWER, m_R ** _W_POWER
    pos = (w_L * p_L + w_R * p_R) / (w_L + w_R + 1e-12)
    m = m_L + m_R
    return Decoded(pos=pos, valid=m / (m + _M_VALID),
                   strength=(jnp.sum(acts.L) + jnp.sum(acts.R)) / _M_BUMP)


# ── Step ─────────────────────────────────────────────────────────────────────

def step(state, target_pos, target_amp):
    """SC map dynamics.

    Args:
        state:      sc.State  membranes of both colliculi
        target_pos: (2,)  predicted current target position [yaw, pitch] (deg, rotation vector)
        target_amp: scalar  input bump amplitude ∈ [0, 1] (visibility × saccadic suppression)

    Returns:
        dstate: sc.State
    """
    acts = read_activations(state)

    # ── Input drive: a unit Gaussian at the target's site, rendered on both colliculi
    in_L = target_amp * _render(target_pos, +1.0)
    in_R = target_amp * _render(target_pos, -1.0)

    # ── Lateral (recurrent) interactions ─────────────────────────────────────
    # Local excitation (Gaussian kernel, same colliculus) minus global inhibition
    # from the total activity of the same and of the other colliculus
    # (tot = Σ rates, in units of one full bump).
    tot_L = jnp.sum(acts.L) / _M_BUMP
    tot_R = jnp.sum(acts.R) / _M_BUMP
    lat_L = _G_EXC * (_K_EXC @ acts.L) - _G_INH * tot_L - _G_CROSS * tot_R
    lat_R = _G_EXC * (_K_EXC @ acts.R) - _G_INH * tot_R - _G_CROSS * tot_L

    # ── Membranes: τ dx/dt = −x + input + lateral ────────────────────────────
    return State(L=(-state.L + in_L + lat_L) / _TAU,
                 R=(-state.R + in_R + lat_R) / _TAU)
