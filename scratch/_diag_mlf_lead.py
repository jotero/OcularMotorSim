"""Sweep mlf_lead (monocular compensation 0->1): does the post-saccadic vergence
ring drop without regressing the version ring / peak velocity / endpoint?"""
import numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, extract_z_opn, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)

for a in [0.0, 0.5, 1.0]:
    P = with_brain(THETA_NOISELESS, mlf_lead=a)
    print(f'\n=== mlf_lead = {a} ===')
    for amp in [5.0, 40.0]:
        pt3 = _pt3(t, amp, t_jump=t_jump)
        st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
        L = np.array(st.plant.left); R = np.array(st.plant.right)
        ver = (L + R) / 2.0; vrg = L - R
        z = extract_z_opn(st)
        win = (t >= t_jump + 0.05) & (t <= 0.45) & (z >= 50.0)
        ver_vel = np.gradient(ver[:, 0], DT)
        vrg_vel = np.gradient(vrg[:, 0], DT)
        pk = lambda x: float(np.max(np.abs(x[win]))) if win.any() else float('nan')
        peak_sac = float(np.max(np.abs(ver_vel)))
        hold = (t >= 0.6) & (t <= 0.85)
        endpoint = float(np.mean(ver[hold, 0]))
        verg_hold = float(np.mean(vrg[hold, 0]))
        tgt = float(np.degrees(np.arctan2(np.array(pt3[:, 0]), np.array(pt3[:, 2])))[-1])
        print(f'  A={amp:>2.0f}: version ring={pk(ver_vel):5.2f}  vergence ring={pk(vrg_vel):5.2f}  '
              f'peak vel={peak_sac:6.1f}  endpoint={endpoint:5.2f} (tgt {tgt:4.1f})  verg_hold={verg_hold:+.3f}')
