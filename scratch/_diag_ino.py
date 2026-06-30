"""Left INO (g_mlf_L=0), rightward 20deg saccade. Decompose the LEFT-eye horizontal
drive into the agonist (MR_L) and antagonist (LR_L) nerves + the plant differential
(MR_L - LR_L), and the pre-pull-only raw firings, to see where the surviving pulse
comes from when the MLF is cut."""
import numpy as np, jax, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.models.plant_models.muscle_geometry import ABN_L, CN3_MR_L, LR_L, MR_L

NM = float(fcp._NERVE_MAX)


def decode(m_arr, bp):
    g_nuc14 = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
    tonic14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])

    def per_t(m):
        raw    = fcp._smooth_clip_sym(m + g_nuc14 * tonic14, NM)   # pre-fold (signed drive)
        firing = fcp.read_activations(fcp.State(mn=m), bp).mn      # >=0 nucleus firing (pull-only fold)
        nerve  = fcp._smooth_clip(fcp._ROUTE @ firing, bp.g_nerve * NM)
        return raw, nerve

    raw, nerve = jax.vmap(per_t)(m_arr)
    return np.array(raw), np.array(nerve)


for g in (1.0, 0.0):
    theta = with_brain(THETA_NOISELESS, g_mlf_L=float(g))
    t = np.arange(0.0, 1.0, DT)
    pt3 = np.zeros((len(t), 3)); pt3[:, 2] = 1.0
    pt3[:, 0] = np.where(t >= 0.1, np.tan(np.radians(20.0)), 0.0)   # rightward 20 deg
    st = _run(t, jnp.array(pt3), key=0, params=theta)

    L = np.array(st.plant.left)[:, 0]
    Lvel = np.gradient(L, DT)
    raw, nerve = decode(jnp.array(np.array(st.brain.fcp.mn)), theta.brain)

    mr = nerve[:, MR_L]; lr = nerve[:, LR_L]
    diff = mr - lr                                   # what the horizontal plant sees
    abn_raw = raw[:, ABN_L]; mr_raw = raw[:, CN3_MR_L]

    i0, ip = 80, int(np.argmax(np.abs(Lvel)))
    print(f'\n=== g_mlf_L = {g} ===  (rest idx {i0}, peak-vel idx {ip}, t={ip*DT:.2f}s)')
    print(f'  left eye          : final {L[-1]:6.2f} deg, peak |vel| {np.abs(Lvel).max():5.0f}')
    print(f'  MR_L nerve  (ago) : rest {mr[i0]:6.1f}  peak {mr[ip]:6.1f}  end {mr[-1]:6.1f}')
    print(f'  LR_L nerve  (ant) : rest {lr[i0]:6.1f}  peak {lr[ip]:6.1f}  end {lr[-1]:6.1f}')
    print(f'  diff MR_L - LR_L  : rest {diff[i0]:6.1f}  peak {diff[ip]:6.1f}  end {diff[-1]:6.1f}   <-- plant drive')
    print(f'  raw CN3_MR_L      : rest {mr_raw[i0]:6.1f}  peak {mr_raw[ip]:6.1f}  end {mr_raw[-1]:6.1f}   (agonist own cmd, pre pull-only)')
    print(f'  raw ABN_L         : rest {abn_raw[i0]:6.1f}  peak {abn_raw[ip]:6.1f}  end {abn_raw[-1]:6.1f}   (antagonist cmd, pre pull-only)')
