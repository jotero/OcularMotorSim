"""Compare the per-muscle pull-only nonlinearity vs the pair (common-mode) version
on the SAME membrane trajectory, to measure how much they diverge — especially on
the MR (the MLF-driven, non-antisymmetric muscle).

  pair (current):     pullonly(smooth_clip_sym(drive, ceiling))   — reads min(pair)
  per-muscle (cand):  relu(x) + relu(x - 2*tonic), capped          — each muscle alone
"""
import numpy as np, jax, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.models.plant_models.muscle_geometry import M_NERVE_PROJ, MR_L, MR_R, AIN_L, AIN_R

NM  = float(fcp._NERVE_MAX)
bp  = THETA_NOISELESS.brain
g_nuc14 = np.concatenate([np.array(bp.g_nucleus), np.array(bp.g_nucleus)[:2]])
tonic14 = np.concatenate([np.array(bp.r_baseline), np.zeros(2)])
tonic12 = np.array(bp.r_baseline)                                  # per-muscle tonic (12,)
route   = np.array(M_NERVE_PROJ.at[MR_L, AIN_R].set(0.0).at[MR_R, AIN_L].set(0.0))
NRV = ['LR_L', 'MR_L', 'SR_L', 'IR_L', 'SO_L', 'IO_L', 'LR_R', 'MR_R', 'SR_R', 'IR_R', 'SO_R', 'IO_R']


def run_case(amp_h, label):
    t = np.arange(0.0, 1.0, DT)
    pt3 = np.zeros((len(t), 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.where(t >= 0.1, np.tan(np.radians(amp_h)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=THETA_NOISELESS)

    mn    = jnp.array(st.brain.fcp.mn)                              # (T,14) membrane
    v     = np.array(jax.vmap(lambda m: fcp._smooth_clip(m, NM))(mn))   # (T,14) f-I membrane
    drive = (v + g_nuc14 * tonic14) @ route.T                      # (T,12) signed per-muscle drive

    nerve_pair = np.array(jax.vmap(lambda d: fcp._pullonly(fcp._smooth_clip_sym(d, NM)))(jnp.array(drive)))
    nerve_pm   = np.minimum(np.maximum(drive, 0.0) + np.maximum(drive - 2 * tonic12[None, :], 0.0), NM)
    diff = nerve_pair - nerve_pm                                   # (T,12)

    print(f'\n=== {label} (h={amp_h}°) ===')
    print('  max |pair - per-muscle| per muscle [spk/s]:')
    for j in range(12):
        d = np.max(np.abs(diff[:, j]))
        flag = '   <-- MLF (MR)' if NRV[j] in ('MR_L', 'MR_R') and d > 1e-3 else ''
        print(f'    {NRV[j]:6s} {d:8.3f}{flag}')
    # eye-velocity command = (LR - MR)/2 per eye — what the plant actually sees
    dL = (diff[:, 0] - diff[:, 1]) / 2.0
    dR = (diff[:, 6] - diff[:, 7]) / 2.0
    print(f'  left  eye H command max diff: {np.max(np.abs(dL)):.4f}  (deg/s-equiv)')
    print(f'  right eye H command max diff: {np.max(np.abs(dR)):.4f}  (deg/s-equiv)')


run_case(20.0, 'horizontal saccade')
run_case(40.0, 'larger horizontal saccade')
