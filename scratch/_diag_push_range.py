"""Does a healthy nerve's PUSH (negative drive) ever reach -NERVE_MAX?

If max|push| < NERVE_MAX, then making the g_nerve CLIP two-sided (symmetric,
clipping push at -g_nerve*NM the same way pull is capped at +g_nerve*NM) is a
clean no-op for healthy nerves (g=1) AND silences a complete block (g=0) — a
pure-clip fix that respects the clip-vs-gain split (no gain bolted onto g_nerve).

z = m_proj @ (rates + g_nuc14*r_base14)  is the pre-clip nerve drive; for z<0 the
one-sided clip passes it unchanged, so min(z) over a saccade = the max push."""
import jax, jax.numpy as jnp, numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
import oculomotor.models.brain_models.final_common_pathway as fcp

NM = float(fcp._NERVE_MAX)
bp = THETA_NOISELESS.brain
g_nuc14  = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
m_proj   = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)

NERVE_NAMES = ['LR_L', 'MR_L', 'SR_L', 'IR_L', 'SO_L', 'IO_L',
               'LR_R', 'MR_R', 'SR_R', 'IR_R', 'SO_R', 'IO_R']


def drive(mn_row):                       # (14,) MN membrane -> (12,) pre-clip nerve drive
    return m_proj @ (fcp._smooth_clip(mn_row, NM) + g_nuc14 * r_base14)


t = np.arange(0.0, 0.9, DT)
print(f'NERVE_MAX = {NM:.0f}\n{"amp":>5} {"min push (z<0)":>15} {"max pull":>10} {"push>NM?":>9}  worst-push nerve')
for amp in (5.0, 20.0, 40.0):
    st = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=THETA_NOISELESS)
    z = np.array(jax.vmap(drive)(jnp.array(st.brain.fcp.mn)))      # (T,12)
    zmin, zmax = float(z.min()), float(z.max())
    worst = NERVE_NAMES[int(np.unravel_index(np.argmin(z), z.shape)[1])]
    print(f'{amp:5.0f} {zmin:15.1f} {zmax:10.1f} {str(zmin < -NM):>9}  {worst}')
