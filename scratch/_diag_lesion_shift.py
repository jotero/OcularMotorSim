"""Is the co-contraction shift transparent to LESIONS too? (dead nerve / dead
nucleus). If the eye is identical signed-vs-shift for each lesion, then adopting
shift does NOT change any lesion behaviour -> 'ok already', no lesion rework."""
import jax, numpy as np
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain
from oculomotor.models.brain_models.brain_model import G_NERVE_DEFAULT, G_NUCLEUS_DEFAULT
from oculomotor.models.plant_models.muscle_geometry import LR_R, ABN_R

t = np.arange(0.0, 0.9, DT); MAXS = int(0.9 / DT) + 200

CASES = {
    'healthy':                 THETA_NOISELESS,
    'CN6  g_nerve[LR_R]=0':    with_brain(THETA_NOISELESS, g_nerve=G_NERVE_DEFAULT.at[LR_R].set(0.0)),
    'nucleus g_nuc[ABN_R]=0':  with_brain(THETA_NOISELESS, g_nucleus=G_NUCLEUS_DEFAULT.at[ABN_R].set(0.0)),
}


def run(mode, params):
    fcp._FLOOR_MODE = mode; jax.clear_caches()
    st = _run(t, _pt3(t, 20.0, 0.0, t_jump=0.1), key=0, max_s=MAXS, params=params)
    L = np.array(st.plant.left)[:, 0]; R = np.array(st.plant.right)[:, 0]
    return float(L[-1]), float(R[-1])


print('rightward +20 gaze — final eye yaw (L, R):')
for name, P in CASES.items():
    Ls, Rs = run('signed', P)
    Lf, Rf = run('shift', P)
    dmax = max(abs(Ls - Lf), abs(Rs - Rf))
    print(f'  {name:22}  signed L={Ls:+7.3f} R={Rs:+7.3f}   '
          f'shift L={Lf:+7.3f} R={Rf:+7.3f}   |Δeye|={dmax:.2e}')
