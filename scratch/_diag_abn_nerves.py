"""What actually moves the eyes under g_nucleus[ABN_R]=0? Look at the 4 horizontal
nerves + per-eye decode at the held position (far +20 rightward target)."""
import jax, jax.numpy as jnp, numpy as np
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain
from oculomotor.models.brain_models.brain_model import G_NUCLEUS_DEFAULT
from oculomotor.models.plant_models.muscle_geometry import ABN_R, LR_L, MR_L, LR_R, MR_R

NM = float(fcp._NERVE_MAX); t = np.arange(0.0, 0.9, DT)


def hold(P):
    st = _run(t, _pt3(t, 20.0, 0.0), key=0, max_s=int(0.9 / DT) + 200, params=P)
    bp = P.brain
    g_nuc14 = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
    r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
    m_proj = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)

    def nv(mn):
        z = m_proj @ (fcp._smooth_clip(mn, NM) + g_nuc14 * r_base14)
        return fcp._pullonly(fcp._smooth_clip_sym(z, bp.g_nerve * NM))
    n = np.array(jax.vmap(nv)(jnp.array(st.brain.fcp.mn)))[-1]
    L = np.array(st.plant.left)[-1, 0]; R = np.array(st.plant.right)[-1, 0]
    return n, L, R


for name, P in [('healthy', THETA_NOISELESS),
                ('ABN_R=0', with_brain(THETA_NOISELESS, g_nucleus=G_NUCLEUS_DEFAULT.at[ABN_R].set(0.0)))]:
    n, L, R = hold(P)
    yawL = (n[MR_L] - n[LR_L]) / 2; yawR = (n[LR_R] - n[MR_R]) / 2
    print(f'{name:9}: LEFT  LR={n[LR_L]:5.1f} MR={n[MR_L]:5.1f} -> (MR-LR)/2={yawL:+6.1f}  eye={L:+6.2f}')
    print(f'{"":9}  RIGHT LR={n[LR_R]:5.1f} MR={n[MR_R]:5.1f} -> (LR-MR)/2={yawR:+6.1f}  eye={R:+6.2f}\n')
