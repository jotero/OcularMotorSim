"""Extraocular muscle geometry — the eye's biomechanics (plant side).

Per-eye muscle action and its inverse (the plant decode):

    M_MUSCLE_ACTION_{L,R}     (6×3)  MUSCLE ACTION — each row is one muscle's
                                     rotation axis per unit firing [yaw, pitch, roll]
                                     (its primary/secondary/tertiary action), row
                                     order LR, MR, SR, IR, SO, IO.
    M_MUSCLE_ACTION_INV_{L,R} (3×6)  = pinv(M_MUSCLE_ACTION), the plant decode: 6
                                     muscle (nerve) activations → 3-D eye-rotation
                                     command.  M_MUSCLE_ACTION_INV @ M_MUSCLE_ACTION = I₃.

The brain's motor ENCODE — version/vergence → motor nuclei → cranial nerves
(M_NUCLEUS, M_NERVE_PROJ, the nucleus/nerve index constants, the lesion-gain
defaults and the mapping helpers) — lives in ``brain_models.final_common_pathway``.
That's the brain's motor code, calibrated to this geometry but NOT plant
biomechanics, so it belongs with the FCP.

Index vocabulary: the per-eye muscle indices (0–5: LR, MR, SR, IR, SO, IO) — the
row order of M_MUSCLE_ACTION — live here.  The combined 12-D bilateral nerve-output
layout ([L 0–5 | R 6–11], LR_L…IO_R + N_NERVES) lives with the FCP.
"""

import numpy as np
import jax.numpy as jnp


# ── Per-muscle rotation-axis components ───────────────────────────────────────
# Each row of M_MUSCLE_ACTION is the rotation-axis-per-unit-firing for one muscle, in
# [yaw, pitch, roll] coords.  These are NOT the muscle pulling directions —
# they're the eye's rotation axis when that muscle contracts, derived from
# R × F (insertion position × force vector) biomechanics.  Specified
# directly per Robinson 1975 / Tweed & Vilis 1998.
#
# Vertical recti (SR, IR): insert ~7 mm in front of equator, pull posteriorly
# with small nasal tilt (α_VR ≈ 23° from sagittal).  R × F gives a rotation
# axis dominated by the lateral component → strong vertical action with
# small torsion as a secondary effect.  Approx pitch:roll ≈ 0.92:0.39.
_VR_PITCH = float(np.cos(np.radians(23.0)))   # 0.92  primary vertical action
_VR_ROLL  = float(np.sin(np.radians(23.0)))   # 0.39  secondary torsion

# Obliques (SO, IO): insert behind the equator on the superior-temporal /
# inferior-temporal sclera and pull anteriorly toward the trochlea / orbital
# floor.  Different insertion/pull geometry → R × F is dominantly along the
# AP axis (z), so the rotation axis is mostly torsional with small vertical
# action.  Robinson 1975 / Tweed & Vilis 1998: pitch ≈ 0.18, roll ≈ 0.98.
# These cannot be obtained from the same `sin α / cos α` formula as the
# vertical recti — the cross-product geometry is fundamentally different.
_OBL_PITCH = 0.18    # secondary vertical action (depression for SO, elevation for IO)
_OBL_ROLL  = 0.98    # primary torsional action (intorsion for SO, extorsion for IO)


# ── Per-eye muscle geometry  M_NERVE_{L,R}  (6 × 3) ──────────────────────────
# Row order: LR(0), MR(1), SR(2), IR(3), SO(4), IO(5)
# Columns: [yaw, pitch, roll]

_M_MUSCLE_ACTION_R_np = np.array([
    [+1.0,         0.0,         0.0       ],  # LR: pure abduction
    [-1.0,         0.0,         0.0       ],  # MR: pure adduction
    [ 0.0, +_VR_PITCH, -_VR_ROLL ],  # SR: dominant elevation + small intorsion
    [ 0.0, -_VR_PITCH, +_VR_ROLL ],  # IR: dominant depression + small extorsion
    [ 0.0, -_OBL_PITCH, -_OBL_ROLL],  # SO: dominant intorsion + small depression (CN IV)
    [ 0.0, +_OBL_PITCH, +_OBL_ROLL],  # IO: dominant extorsion + small elevation
], dtype=np.float32)

# Left eye: mirror yaw (col 0) and roll (col 2)
_M_MUSCLE_ACTION_L_np = _M_MUSCLE_ACTION_R_np * np.array([-1.0, +1.0, -1.0], dtype=np.float32)

# Per-eye decode: M_MUSCLE_ACTION_INV = pinv(M_MUSCLE_ACTION)  →  M_MUSCLE_ACTION_INV @ M_MUSCLE_ACTION = I₃
#
# For reference: with symmetric 45°/45° angles (Q=0, pitch-roll decouple):
#
#   M_MUSCLE_ACTION_INV_R  (row=axis, col=muscle: LR    MR    SR     IR     SO     IO)
#     yaw:        [+1/2, -1/2,   0,     0,     0,     0   ]
#     pitch:      [  0,    0,  +√2/4, -√2/4, -√2/4, +√2/4]
#     roll:       [  0,    0,  -√2/4, +√2/4, -√2/4, +√2/4]
#
#   M_MUSCLE_ACTION_INV_L  (yaw and roll rows negated vs R):
#     yaw:        [-1/2, +1/2,   0,     0,     0,     0   ]
#     pitch:      [  0,    0,  +√2/4, -√2/4, -√2/4, +√2/4]
#     roll:       [  0,    0,  +√2/4, -√2/4, +√2/4, -√2/4]
_M_MUSCLE_ACTION_INV_R_np = np.linalg.pinv(_M_MUSCLE_ACTION_R_np).astype(np.float32)  # (3, 6)
_M_MUSCLE_ACTION_INV_L_np = np.linalg.pinv(_M_MUSCLE_ACTION_L_np).astype(np.float32)  # (3, 6)


# ── Per-eye muscle indices — the row order of M_MUSCLE_ACTION (6 muscles) ──────
LR = 0   # Lateral  Rectus  (CN VI)
MR = 1   # Medial   Rectus  (CN III)
SR = 2   # Superior Rectus  (CN III)
IR = 3   # Inferior Rectus  (CN III)
SO = 4   # Superior Oblique (CN IV)
IO = 5   # Inferior Oblique (CN III)


# ── JAX arrays (immutable; safe inside jit) ────────────────────────────────────

M_MUSCLE_ACTION_R     = jnp.array(_M_MUSCLE_ACTION_R_np)             # (6, 3)  right-eye muscle geometry
M_MUSCLE_ACTION_L     = jnp.array(_M_MUSCLE_ACTION_L_np)             # (6, 3)  left-eye  muscle geometry
M_MUSCLE_ACTION_INV_R = jnp.array(_M_MUSCLE_ACTION_INV_R_np)        # (3, 6)  right-eye decode (plant)
M_MUSCLE_ACTION_INV_L = jnp.array(_M_MUSCLE_ACTION_INV_L_np)        # (3, 6)  left-eye  decode (plant)
