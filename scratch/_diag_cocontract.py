"""Test the user's hypothesis: is the saccadic glissade caused by 'muscles can't
push', or by the OLD floor being a DIFFERENTIAL (independent) rectify?

Three pull schemes on the same healthy 20deg saccade:
  signed - antagonist fires negative (current model)
  shift  - co-contraction: lift LR/MR pair by relu(-min) -> both >=0, COMMON-MODE
           (cancels in (LR-MR)/2) -> predict IDENTICAL eye, pull-only output
  indep  - relu(LR), relu(MR) independently (old floor) -> DIFFERENTIAL ->
           predict velocity loss / glissade

If 'shift' == 'signed' for the eye but pull-only at the output, the floor was
misidentified: co-contraction gives physiological muscles for free."""
import jax, jax.numpy as jnp, numpy as np
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT)
from oculomotor.analysis import extract_burst
from oculomotor.models.plant_models.muscle_geometry import LR_L, MR_L, LR_R, MR_R

t = np.arange(0.0, 0.9, DT); i_j = int(0.1 / DT)
NM = float(fcp._NERVE_MAX)
bp = THETA_NOISELESS.brain
g_nuc14  = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
m_proj   = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)


def nerves_of(mn_row):     # recompute output nerves under the current _FLOOR_MODE
    z = m_proj @ (fcp._smooth_clip(mn_row, NM) + g_nuc14 * r_base14)
    return fcp._pullonly(fcp._smooth_clip_sym(z, bp.g_nerve * NM))


def run(mode):
    fcp._FLOOR_MODE = mode
    jax.clear_caches()
    st = _run(t, _pt3(t, 20.0, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=THETA_NOISELESS)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel = np.gradient(eye, DT)
    z = extract_z_opn(st); ub = np.array(extract_burst(st, THETA_NOISELESS))[:, 0]
    fast = np.abs(ub) > 20.0; be = int(np.where(fast)[0][-1]) if fast.any() else 0
    rw = np.zeros_like(t, bool); rw[be + 30:be + 230] = True; rw &= (z >= 50.0)
    ring = float(np.max(np.abs(evel[rw]))) if rw.any() else float('nan')
    pv = float(np.max(np.abs(evel[i_j:i_j + 400])))
    nv = np.array(jax.vmap(nerves_of)(jnp.array(st.brain.fcp.mn)))
    return pv, ring, float(eye[-1]), float(nv[:, MR_R].min()), float(nv[:, LR_L].min())


print(f'{"mode":7} {"peak_vel":>9} {"post-sac ring":>14} {"final":>7} {"min MR_R":>9} {"min LR_L":>9}')
for mode in ('signed', 'shift', 'indep'):
    pv, ring, fin, mr, lr = run(mode)
    tag = '(antagonists, <0 = pushing)' if mode == 'signed' else ''
    print(f'{mode:7} {pv:9.2f} {ring:14.3f} {fin:7.2f} {mr:9.1f} {lr:9.1f}  {tag}')
