"""Visibility-flag builders — the only stimulus helpers still in this module.

All kinematic stimulus construction (head, scene, target) has moved to
``oculomotor.sim.kinematics``.  This module retains the per-eye visibility-flag
builders used by the LLM pipeline, plus ``strobe_train`` — THE way to strobe a
target or a scene (a genuine flash train on target_present / scene_present;
there is no separate strobe flag).
"""

import numpy as np
import jax.numpy as jnp

STROBE_HZ_DEFAULT = 1.0    # flashes per second
STROBE_MS_DEFAULT = 20.0   # flash duration (ms)


def strobe_train(t, rate_hz: float = STROBE_HZ_DEFAULT, flash_ms: float = STROBE_MS_DEFAULT,
                 t0: float = 0.0) -> np.ndarray:
    """Stroboscopic visibility: (T,) float32 in {0, 1}, 1 during each flash.

    Flashes start at t0 and repeat every 1/rate_hz s, each lasting flash_ms.
    Before t0 the array is 0.  Use as target_present / scene_present (or a
    per-eye override) — between flashes the stimulus is genuinely absent, so
    position is only sampled at the flashes and there is no usable motion signal.
    """
    t     = np.asarray(t, dtype=np.float64)
    rel   = t - t0
    phase = np.mod(np.maximum(rel, 0.0), 1.0 / rate_hz)
    return ((rel >= 0.0) & (phase < flash_ms * 1e-3)).astype(np.float32)


def _visibility(value, t_seg, rate_hz, flash_ms):
    """Segment visibility (True / False / 'strobe') → (T_seg,) float32 array."""
    if value == 'strobe':
        return strobe_train(t_seg, rate_hz, flash_ms)
    return np.full(len(t_seg), float(bool(value)), dtype=np.float32)


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
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert VisualFlagsSegments to per-eye visual flag + cover arrays.

    scene_present_L/R and target_present_L/R default to the segment's scene_present /
    target_present value when the per-eye override fields are None — enabling monocular
    occlusion and cover-test scenarios.

    Each visibility value is True / False / 'strobe'.  'strobe' expands to a flash
    train (strobe_train) at the segment's strobe_hz / strobe_ms, phase-locked to
    the segment start.

    cover_L/cover_R are the high-level cover intent: when set, that eye's scene AND
    target are forced off for the simulation (the eye sees nothing), and a cover
    flag is emitted so the viz can draw a patch without re-inferring it. cover wins
    over the per-eye scene_present_*/target_present_* overrides.

    Returns:
        scene_present_L:  (T,) float32 in [0, 1] — L eye scene visibility
        scene_present_R:  (T,) float32 in [0, 1] — R eye scene visibility
        target_present_L: (T,) float32 in [0, 1] — L eye target visibility
        target_present_R: (T,) float32 in [0, 1] — R eye target visibility
        cover_L:          (T,) float32 in {0, 1} — 1 = left eye covered
        cover_R:          (T,) float32 in {0, 1} — 1 = right eye covered
    """
    spL_chunks, spR_chunks, tpL_chunks, tpR_chunks = [], [], [], []
    cvL_chunks, cvR_chunks = [], []
    for seg in segments:
        T     = max(1, round(seg.duration_s / dt))
        t_seg = np.arange(T) * dt
        hz    = getattr(seg, 'strobe_hz', STROBE_HZ_DEFAULT)
        ms    = getattr(seg, 'strobe_ms', STROBE_MS_DEFAULT)
        def vis(v, fallback):
            return _visibility(fallback if v is None else v, t_seg, hz, ms)
        spL = vis(seg.scene_present_L,  seg.scene_present)
        spR = vis(seg.scene_present_R,  seg.scene_present)
        tpL = vis(seg.target_present_L, seg.target_present)
        tpR = vis(seg.target_present_R, seg.target_present)
        # Cover is the high-level intent: occlude that eye entirely (scene+target off).
        cvL = 1.0 if getattr(seg, 'cover_L', False) else 0.0
        cvR = 1.0 if getattr(seg, 'cover_R', False) else 0.0
        if cvL: spL = tpL = np.zeros(T, dtype=np.float32)
        if cvR: spR = tpR = np.zeros(T, dtype=np.float32)
        spL_chunks.append(spL); spR_chunks.append(spR)
        tpL_chunks.append(tpL); tpR_chunks.append(tpR)
        cvL_chunks.append(np.full(T, cvL, dtype=np.float32))
        cvR_chunks.append(np.full(T, cvR, dtype=np.float32))

    spL = np.concatenate(spL_chunks)
    spR = np.concatenate(spR_chunks)
    tpL = np.concatenate(tpL_chunks)
    tpR = np.concatenate(tpR_chunks)
    cvL = np.concatenate(cvL_chunks)
    cvR = np.concatenate(cvR_chunks)

    def _fit1d(arr, T):
        if len(arr) >= T: return arr[:T]
        return np.concatenate([arr, np.full(T - len(arr), arr[-1], dtype=np.float32)])

    return (_fit1d(spL, total_T), _fit1d(spR, total_T),
            _fit1d(tpL, total_T), _fit1d(tpR, total_T),
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
