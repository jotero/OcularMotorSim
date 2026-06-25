"""Combined per-eye plant inverse: version MN-LP feedforward (mn_ff_yaw) +
monocular MLF lead (mlf_lead). Test the EXACT split (1.0 + 1.0) vs the legacy
1.5-averaged compromise, at far (isolates conjugate + per-eye residual) and near
targets. Gauge version ring / vergence ring / velocity gap / peak velocity."""
import numpy as np
import jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, extract_z_opn, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
amp = 40.0


def pt3_at(D):
    p = np.zeros((len(t), 3)); p[:, 2] = D
    p[:, 0] = np.where(t >= t_jump, D * np.tan(np.radians(amp)), 0.0)
    return jnp.array(p)


combos = [(1.5, 0.0, 'current (1.5, mlf 0)'),
          (1.0, 0.0, '1.0, mlf 0'),
          (1.0, 1.0, 'EXACT split (1.0,1)'),
          (1.5, 1.0, '1.5, mlf 1')]

for D, dlabel in [(100.0, 'FAR'), (1.0, 'near')]:
    print(f'\n===== {dlabel} target (D={D:.0f} m) =====')
    pt3 = pt3_at(D)
    for ff, ml, lbl in combos:
        P = with_brain(THETA_NOISELESS, mn_ff_yaw=ff, mlf_lead=ml)
        st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
        L = np.array(st.plant.left); R = np.array(st.plant.right)
        ver = (L + R) / 2.0; vrg = L - R
        ni = np.array(ni_net(st))
        z = extract_z_opn(st)
        win = (t >= t_jump + 0.05) & (t <= 0.45) & (z >= 50.0)
        pk = lambda x: float(np.max(np.abs(np.gradient(x, DT)[win])))
        wb = (t >= 0.1) & (t <= 0.45)
        gap = float(np.max(np.abs((np.gradient(ver[:, 0], DT) - np.gradient(ni[:, 0], DT))[wb])))
        peakv = float(np.max(np.abs(np.gradient(ver[:, 0], DT))))
        print(f'  {lbl:22s}: ver_ring={pk(ver[:,0]):5.2f}  verg_ring={pk(vrg[:,0]):5.2f}  '
              f'vel_gap={gap:6.1f}  peak={peakv:6.1f}')
