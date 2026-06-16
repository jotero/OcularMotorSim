"""Visual-flags segment builder — the only stimulus helper still in this module.

All kinematic stimulus construction (head, scene, target) has moved to
``oculomotor.sim.kinematics``.  This module retains only the per-eye
visibility-flag builder used by the LLM pipeline.
"""

import numpy as np
import jax.numpy as jnp


def build_cover_flags(
    total_T: int,
    cover_L: bool = False,
    cover_R: bool = False,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return per-eye visibility arrays for a cover-test configuration.

    A cover zeroes both scene_present and target_present for the covered eye
    for the entire trial.  Pass the returned arrays as scene_present_L/R_array
    and target_present_L/R_array to simulate().

    Args:
        total_T:  number of time steps
        cover_L:  True → left eye covered (dark + no target)
        cover_R:  True → right eye covered (dark + no target)
        dt:       unused; kept for API symmetry with other flag builders

    Returns:
        scene_present_L, scene_present_R, target_present_L, target_present_R
        — each (total_T,) float32 in {0, 1}
    """
    ones  = np.ones(total_T,  dtype=np.float32)
    zeros = np.zeros(total_T, dtype=np.float32)
    sp_L = zeros if cover_L else ones
    sp_R = zeros if cover_R else ones
    tp_L = zeros if cover_L else ones
    tp_R = zeros if cover_R else ones
    return sp_L, sp_R, tp_L, tp_R


def build_visual_flags(
    segments,   # list[VisualFlagsSegment]
    total_T: int,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert VisualFlagsSegments to per-eye visual flag + cover arrays.

    scene_present_L/R and target_present_L/R default to the segment's scene_present /
    target_present value when the per-eye override fields are None — enabling monocular
    occlusion and cover-test scenarios.

    cover_L/cover_R are the high-level cover intent: when set, that eye's scene AND
    target are forced off for the simulation (the eye sees nothing), and a cover
    flag is emitted so the viz can draw a patch without re-inferring it. cover wins
    over the per-eye scene_present_*/target_present_* overrides.

    Returns:
        scene_present_L:  (T,) float32 in [0, 1] — L eye scene visibility
        scene_present_R:  (T,) float32 in [0, 1] — R eye scene visibility
        target_present_L: (T,) float32 in [0, 1] — L eye target visibility
        target_present_R: (T,) float32 in [0, 1] — R eye target visibility
        target_strobed:   (T,) float32 in {0, 1} — 1 = stroboscopic (velocity absent)
        cover_L:          (T,) float32 in {0, 1} — 1 = left eye covered
        cover_R:          (T,) float32 in {0, 1} — 1 = right eye covered
    """
    spL_chunks, spR_chunks, tpL_chunks, tpR_chunks, ts_chunks = [], [], [], [], []
    cvL_chunks, cvR_chunks = [], []
    for seg in segments:
        T   = max(1, round(seg.duration_s / dt))
        sp  = float(seg.scene_present)
        spL = float(seg.scene_present_L)  if seg.scene_present_L  is not None else sp
        spR = float(seg.scene_present_R)  if seg.scene_present_R  is not None else sp
        tp  = float(seg.target_present)
        tpL = float(seg.target_present_L) if seg.target_present_L is not None else tp
        tpR = float(seg.target_present_R) if seg.target_present_R is not None else tp
        # Cover is the high-level intent: occlude that eye entirely (scene+target off).
        cvL = 1.0 if getattr(seg, 'cover_L', False) else 0.0
        cvR = 1.0 if getattr(seg, 'cover_R', False) else 0.0
        if cvL: spL = tpL = 0.0
        if cvR: spR = tpR = 0.0
        ts  = float(getattr(seg, 'target_strobed', False))
        spL_chunks.append(np.full(T, spL, dtype=np.float32))
        spR_chunks.append(np.full(T, spR, dtype=np.float32))
        tpL_chunks.append(np.full(T, tpL, dtype=np.float32))
        tpR_chunks.append(np.full(T, tpR, dtype=np.float32))
        ts_chunks.append(np.full(T, ts,  dtype=np.float32))
        cvL_chunks.append(np.full(T, cvL, dtype=np.float32))
        cvR_chunks.append(np.full(T, cvR, dtype=np.float32))

    spL = np.concatenate(spL_chunks)
    spR = np.concatenate(spR_chunks)
    tpL = np.concatenate(tpL_chunks)
    tpR = np.concatenate(tpR_chunks)
    ts  = np.concatenate(ts_chunks)
    cvL = np.concatenate(cvL_chunks)
    cvR = np.concatenate(cvR_chunks)

    def _fit1d(arr, T):
        if len(arr) >= T: return arr[:T]
        return np.concatenate([arr, np.full(T - len(arr), arr[-1], dtype=np.float32)])

    return (_fit1d(spL, total_T), _fit1d(spR, total_T),
            _fit1d(tpL, total_T), _fit1d(tpR, total_T),
            _fit1d(ts,  total_T),
            _fit1d(cvL, total_T), _fit1d(cvR, total_T))


def build_prisms(
    segments,   # list[VisualFlagsSegment]
    total_T: int,
    dt: float = 0.001,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert VisualFlagsSegments to per-eye prism-deviation arrays.

    Each segment's prism_L/prism_R is a [yaw, pitch, roll] deg deviation (head
    frame), or None for no prism over that span. Pass the results to simulate()
    as prism_L_array / prism_R_array.

    Returns:
        prism_L: (T, 3) float32 — left-eye deviation [yaw, pitch, roll] deg
        prism_R: (T, 3) float32 — right-eye deviation [yaw, pitch, roll] deg
    """
    def _vec3(v):
        a = np.zeros(3, dtype=np.float32)
        if v:
            for i in range(min(3, len(v))):
                a[i] = float(v[i])
        return a

    L_chunks, R_chunks = [], []
    for seg in segments:
        T = max(1, round(seg.duration_s / dt))
        L_chunks.append(np.tile(_vec3(getattr(seg, 'prism_L', None)), (T, 1)))
        R_chunks.append(np.tile(_vec3(getattr(seg, 'prism_R', None)), (T, 1)))
    L = np.concatenate(L_chunks, axis=0)
    R = np.concatenate(R_chunks, axis=0)

    def _fit2d(arr, T):
        if len(arr) >= T: return arr[:T]
        pad = np.tile(arr[-1], (T - len(arr), 1))
        return np.concatenate([arr, pad], axis=0)

    return _fit2d(L, total_T), _fit2d(R, total_T)
