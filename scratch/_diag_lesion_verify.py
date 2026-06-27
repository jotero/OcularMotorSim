"""Verify the two-sided g_nerve clip (lesion-aware) at the SIM level:
  [1] HEALTHY saccade  -> peak vel must match golden 640.1 (no-op)
  [2] CN6(R) rightward -> R eye abduction deficit PRESERVED (clinical hallmark)
  [3] CN6(R) leftward  -> dead LR_R nerve now SILENT (was pushing negative)
"""
import jax, jax.numpy as jnp, numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain
from oculomotor.models.brain_models.brain_model import G_NERVE_DEFAULT
from oculomotor.models.plant_models.muscle_geometry import LR_R
import oculomotor.models.brain_models.final_common_pathway as fcp

t = np.arange(0.0, 0.9, DT); i_j = int(0.1 / DT); MAXS = int(0.9 / DT) + 200
NM = float(fcp._NERVE_MAX)


def sim(amp, params):
    st = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=MAXS, params=params)
    L = np.array(st.plant.left)[:, 0]; R = np.array(st.plant.right)[:, 0]
    pv = float(np.max(np.abs(np.gradient(0.5 * (L + R), DT)[i_j:i_j + 400])))
    return st, L, R, pv


# [1] healthy
_, L, R, pv = sim(20.0, THETA_NOISELESS)
print(f'[1] HEALTHY  +20: peak_vel={pv:8.3f}  (golden 640.148)  ver_final={0.5*(L[-1]+R[-1]):6.3f}')

# CN6(R) nerve palsy: g_nerve[LR_R]=0
CN6 = with_brain(THETA_NOISELESS, g_nerve=G_NERVE_DEFAULT.at[LR_R].set(0.0))
bp = CN6.brain
g_nuc14  = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
m_proj   = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)


def drive(mn_row):       # (14,) -> (12,) pre-clip nerve drive
    return m_proj @ (fcp._smooth_clip(mn_row, NM) + g_nuc14 * r_base14)


def report_cn6(amp, label):
    st, L, R, pv = sim(amp, CN6)
    z = np.array(jax.vmap(drive)(jnp.array(st.brain.fcp.mn)))[:, LR_R]   # LR_R drive over time
    old = np.array(fcp._smooth_clip(jnp.array(z), 0.0))                   # one-sided @ g=0 (was)
    new = np.array(fcp._smooth_clip_sym(jnp.array(z), 0.0))               # two-sided @ g=0 (now)
    print(f'[{label}] CN6(R) {amp:+.0f}: R_eye={R[-1]:+6.2f}  L_eye={L[-1]:+6.2f}   '
          f'LR_R drive z[{z.min():+6.1f},{z.max():+5.1f}]  '
          f'OLD nerve[{old.min():+6.1f},{old.max():+5.1f}](push)  NEW nerve[{new.min():+.3f},{new.max():+.3f}](silent)')


report_cn6(20.0, '2')    # rightward: R must abduct (paretic -> undershoots)
report_cn6(-20.0, '3')   # leftward:  R adducts, dead LR_R is antagonist (was pushing)
