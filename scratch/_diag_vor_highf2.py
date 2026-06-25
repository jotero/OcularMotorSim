"""Is the VOR high-f drop the SPV-extraction masking, not a real rolloff? For the
dark VOR at high f, compare the gain with fast-phase masking ON (the bench's
extract_spv_states) vs OFF (same gradient, no masking), and report the masked
fraction. If masked-gain rolls off while no-mask stays flat AND the masked
fraction is large, the rolloff is a measurement artifact of the ±50 ms mask +
interpolation, not the VOR."""
import numpy as np, jax.numpy as jnp
from scipy.ndimage import binary_dilation
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import extract_spv, extract_z_opn
from oculomotor.benchmarks import bode

POS_MAX, MARGIN = 20.0, 0.05


def amp(sig, t, f, frac=0.5):
    m = t >= frac * t[-1]
    a, _, _ = bode.fit_sinusoid(t[m], np.asarray(sig)[m], f)
    return a


print(f'{"f(Hz)":>6} {"masked_frac":>11} {"gain_masked":>11} {"gain_nomask":>11}')
for f in (1.0, 2.0, 5.0, 10.0, 20.0):
    w = 2 * np.pi * f
    V = bode.capped_velocity_amp(f, 30.0, POS_MAX)
    T_end = max(10.0, 10.0 / f)
    t = np.arange(0.0, T_end, DT); Tn = len(t)
    hv = np.zeros((Tn, 3)); hv[:, 0] = V * np.sin(w * t)
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.zeros(Tn), target_present=jnp.zeros(Tn), key=0)
    z_opn = extract_z_opn(st)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    eye_vel = np.gradient(eye, DT)
    # masked fraction (z_opn<50 dilated by ±MARGIN), matching extract_spv
    mn = max(1, int(MARGIN / DT))
    is_fast = binary_dilation(np.asarray(z_opn) < 50.0, structure=np.ones(2 * mn + 1))
    spv_on  = extract_spv(t, eye_vel, z_opn=z_opn, margin_s=MARGIN)            # masking ON
    spv_off = extract_spv(t, eye_vel, z_opn=np.full(Tn, 1e3), margin_s=MARGIN)  # masking OFF (raw)
    h = amp(hv[:, 0], t, f)
    print(f'{f:6.1f} {is_fast.mean():11.2%} {amp(spv_on, t, f)/h:11.3f} '
          f'{amp(spv_off, t, f)/h:11.3f}')
