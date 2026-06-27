"""Concrete look at pull vs push: the 4 horizontal-muscle nerves during a healthy
20deg RIGHTWARD saccade, plus the plant decode yaw = (LR - MR)/2 per eye.

Right eye ABDUCTS (LR_R pulls, MR_R relaxes); left eye ADDUCTS (MR_L pulls,
LR_L relaxes).  Watch whether the relaxing antagonist stays >=0 (real muscle) or
dips below 0 (model 'push')."""
import jax, jax.numpy as jnp, numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.models.plant_models.muscle_geometry import LR_L, MR_L, LR_R, MR_R

NM = float(fcp._NERVE_MAX)
bp = THETA_NOISELESS.brain
g_nuc14  = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
m_proj   = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)


def nerves(mn_row):                       # (14,) MN membrane -> (12,) nerve firing rates
    z = m_proj @ (fcp._smooth_clip(mn_row, NM) + g_nuc14 * r_base14)
    return fcp._smooth_clip_sym(z, bp.g_nerve * NM)


t = np.arange(0.0, 0.9, DT); i_j = int(0.1 / DT)
st = _run(t, _pt3(t, 20.0, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=THETA_NOISELESS)
nv = np.array(jax.vmap(nerves)(jnp.array(st.brain.fcp.mn)))     # (T,12)
R_eye = np.array(st.plant.right)[:, 0]; L_eye = np.array(st.plant.left)[:, 0]

# pick three instants: rest, peak burst (max |LR_R rate|), settled hold
i_rest = i_j - 5
i_peak = i_j + int(np.argmax(nv[i_j:len(t) - 50, LR_R]))   # true agonist burst peak
i_hold = len(t) - 50

print(f'{"":10} {"LR_R":>8} {"MR_R":>8} {"yawR=(LR-MR)/2":>15}   {"MR_L":>8} {"LR_L":>8} {"yawL=(MR-LR)/2":>15}   R_eye')
for name, i in (('rest', i_rest), ('burst peak', i_peak), ('hold 20deg', i_hold)):
    yawR = 0.5 * (nv[i, LR_R] - nv[i, MR_R])
    yawL = 0.5 * (nv[i, MR_L] - nv[i, LR_L])
    print(f'{name:10} {nv[i,LR_R]:8.1f} {nv[i,MR_R]:8.1f} {yawR:15.1f}   '
          f'{nv[i,MR_L]:8.1f} {nv[i,LR_L]:8.1f} {yawL:15.1f}   {R_eye[i]:+5.1f}')

print(f'\nantagonist extremes over the saccade:')
print(f'  MR_R (R-eye antagonist) min = {nv[:,MR_R].min():+7.1f}   (<0 => pushing)')
print(f'  LR_L (L-eye antagonist) min = {nv[:,LR_L].min():+7.1f}   (<0 => pushing)')
print(f'  agonist peaks: LR_R {nv[:,LR_R].max():.1f}  MR_L {nv[:,MR_L].max():.1f}   (tonic=50, NERVE_MAX={NM:.0f})')
