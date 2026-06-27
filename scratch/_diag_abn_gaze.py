"""Does g_nucleus[ABN_R]=0 give a true gaze palsy (BOTH eyes fail to go right)?
Decompose the eyes into version=(L+R)/2 and vergence=L-R, and check whether the
internuclear AIN_R is actually silenced (which would kill the left MR via MLF)."""
import jax, jax.numpy as jnp, numpy as np
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain
from oculomotor.models.brain_models.brain_model import G_NUCLEUS_DEFAULT
from oculomotor.models.plant_models.muscle_geometry import ABN_R, AIN_R

t = np.arange(0.0, 0.9, DT)


def run(P):
    st = _run(t, _pt3(t, 20.0, 0.0), key=0, max_s=int(0.9 / DT) + 200, params=P)
    L = np.array(st.plant.left)[:, 0]; R = np.array(st.plant.right)[:, 0]
    mn = np.array(st.brain.fcp.mn)
    rates = np.array(jax.vmap(lambda m: fcp._smooth_clip(jnp.array(m), fcp._NERVE_MAX))(mn))
    return L, R, rates[:, AIN_R]


_ABN = G_NUCLEUS_DEFAULT.at[ABN_R].set(0.0)
CASES = {
    'healthy':            THETA_NOISELESS,
    'ABN_R=0':            with_brain(THETA_NOISELESS, g_nucleus=_ABN),
    'ABN_R=0 svbn=0':     with_brain(THETA_NOISELESS, g_nucleus=_ABN, g_svbn_conv=0.0),
    'ABN_R=0 svbn+verg=0': with_brain(THETA_NOISELESS, g_nucleus=_ABN, g_svbn_conv=0.0,
                                      K_verg=0.0, K_phasic_verg=0.0),
}

print('+20 far rightward target:')
print(f'{"case":20} {"L_eye":>7} {"R_eye":>7} {"version":>8} {"vergence":>9} {"AIN_R rate":>16}')
for name, P in CASES.items():
    L, R, ain = run(P)
    ver = 0.5 * (L[-1] + R[-1]); vrg = L[-1] - R[-1]
    print(f'{name:20} {L[-1]:+7.2f} {R[-1]:+7.2f} {ver:+8.2f} {vrg:+9.2f}   [{ain.min():+6.1f},{ain.max():+6.1f}]')
