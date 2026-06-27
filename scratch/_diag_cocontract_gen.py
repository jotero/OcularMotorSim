"""Generalize co-contraction to all 3 reciprocal pairs (LR/MR, SR/IR, SO/IO).
Test horizontal, vertical, and oblique saccades: shift must be eye-identical to
signed (transparent) while driving ALL 12 nerves >= 0 (pull-only)."""
import jax, jax.numpy as jnp, numpy as np
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT)
from oculomotor.analysis import extract_burst

t = np.arange(0.0, 0.9, DT); i_j = int(0.1 / DT)
NM = float(fcp._NERVE_MAX); bp = THETA_NOISELESS.brain
g_nuc14  = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
m_proj   = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)


def nerves_of(mn):
    z = m_proj @ (fcp._smooth_clip(mn, NM) + g_nuc14 * r_base14)
    return fcp._pullonly(fcp._smooth_clip_sym(z, bp.g_nerve * NM))


def run(mode, h, v):
    fcp._FLOOR_MODE = mode; jax.clear_caches()
    st = _run(t, _pt3(t, h, v, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=THETA_NOISELESS)
    eye = 0.5 * (np.array(st.plant.left) + np.array(st.plant.right))      # (T,3)
    spd = np.sqrt(np.gradient(eye[:, 0], DT) ** 2 + np.gradient(eye[:, 1], DT) ** 2)
    z = extract_z_opn(st); ub = np.array(extract_burst(st, THETA_NOISELESS))
    fast = np.linalg.norm(ub, axis=1) > 20.0; be = int(np.where(fast)[0][-1]) if fast.any() else 0
    rw = np.zeros_like(t, bool); rw[be + 30:be + 230] = True; rw &= (z >= 50.0)
    ring = float(spd[rw].max()) if rw.any() else float('nan')
    pv = float(spd[i_j:i_j + 400].max())
    nv = np.array(jax.vmap(nerves_of)(jnp.array(st.brain.fcp.mn)))
    return eye[-1, 0], eye[-1, 1], pv, ring, float(nv.min())


for name, (h, v) in {'horiz (20,0)': (20., 0.), 'vert (0,20)': (0., 20.),
                     'obliq (20,20)': (20., 20.)}.items():
    print(f'\n{name}:')
    for mode in ('signed', 'shift'):
        yf, pf, pv, ring, mn = run(mode, h, v)
        print(f'  {mode:7} final=({yf:+6.2f},{pf:+6.2f})  peak_spd={pv:7.2f}  '
              f'ring={ring:6.3f}  min_nerve={mn:+7.1f}')
