"""Is the post-saccadic vergence transient the NEAR-target geometric vergence
change (vergence system) or the MLF motor asymmetry? Re-run the 40deg saccade at
a near (1 m) vs far (100 m) target. If the vergence ring collapses at far, it's
geometric/vergence-system, not the MLF."""
import numpy as np
import jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, extract_z_opn, THETA_NOISELESS, DT

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
amp = 40.0

for D in [1.0, 100.0]:
    pt3 = np.zeros((len(t), 3)); pt3[:, 2] = D
    pt3[:, 0] = np.where(t >= t_jump, D * np.tan(np.radians(amp)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
    L = np.array(st.plant.left); R = np.array(st.plant.right)
    ver = (L + R) / 2.0; vrg = L - R
    z = extract_z_opn(st)
    win = (t >= t_jump + 0.05) & (t <= 0.45) & (z >= 50.0)
    pk = lambda x: float(np.max(np.abs(np.gradient(x, DT)[win])))
    hold = (t >= 0.6) & (t <= 0.85)
    print(f'D={D:5.0f} m:  version ring={pk(ver[:,0]):5.2f}  vergence ring={pk(vrg[:,0]):5.2f}  '
          f'verg_hold={np.mean(vrg[hold,0]):+.3f} deg')
