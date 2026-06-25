"""Open the pursuit + OKN loops (zero their gains) and measure (a) whether the
post-saccadic ring persists (=> not the visual loops) and (b) the EC mismatch at
the target/scene-velocity inputs (pred_err / scene PE). Also extract fl_drive
(flocculus NI leak-cancellation → u_ni_in) to test whether the FL pulse-step
matching is what's actually moving the NI.
"""
import numpy as np
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT, read_brain_acts)
from oculomotor.analysis import ni_net
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)

P_open = with_brain(THETA_NOISELESS, K_pursuit=0.0, K_phasic_pursuit=0.0,
                    K_pursuit_direct=0.0, K_cereb_pu=0.0,
                    K_vor_direct=0.0, K_cereb_okr=0.0)

for label, P in [('CLOSED (normal)', THETA_NOISELESS), ('OPEN pursuit+OKN loops', P_open)]:
    print(f'\n############ {label} ############')
    for amp in [5.0, 40.0]:
        pt3 = _pt3(t, amp, t_jump=t_jump)
        st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
        acts = read_brain_acts(st, P)
        L = np.array(st.plant.left); R = np.array(st.plant.right)
        ver = (L + R) / 2.0; vrg = L - R
        ni = np.array(ni_net(st))
        z = extract_z_opn(st)
        win = (t >= t_jump + 0.05) & (t <= 0.45) & (z >= 50.0)

        fl_drive = np.array(acts.cb.fl_drive)[:, 0]                 # FL NI drive (deg/s)
        pred_err = np.array(acts.cb.pred_err)[:, 0]                 # target mismatch (pursuit PE)
        sat_sc   = np.array(acts.cb.saccadic_suppression_scene)
        scene_mm = sat_sc * np.array(acts.pc.scene_angular_vel)[:, 0] + np.array(acts.cb.fl_okr_drive)[:, 0]
        pk = lambda x: float(np.max(np.abs(x[win]))) if win.any() else float('nan')
        print(f'  A={amp:>2.0f}:  version ring={pk(np.gradient(ver[:,0],DT)):5.2f}  '
              f'vergence ring={pk(np.gradient(vrg[:,0],DT)):5.2f}  '
              f'NI vel={pk(np.gradient(ni[:,0],DT)):5.2f}  '
              f'|fl_drive|={pk(fl_drive):5.2f}  '
              f'target_mm={pk(pred_err):.3f}  scene_mm={pk(scene_mm):.3f}')
