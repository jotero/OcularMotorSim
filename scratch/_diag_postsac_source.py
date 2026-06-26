"""Small-saccade post-saccadic NI drift: WHAT drives it and WHERE it enters.
Window = post-jump slow phase only (z_opn>=50, fast phases excluded), current params.

(1) EC mismatch + NI drift vs saccade size — is the residual bigger for small saccades?
    target_mm = acts.cb.pred_err (pursuit-side EC residual slip)
    scene_mm  = sat*scene_angular_vel + fl_okr_drive (VS/OKR-side EC residual)
    fl_drive  = flocculus NI leak-cancel drive
(2) 2 deg drift decomposition — zero each visual path in turn to see which carries it."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT, read_brain_acts)
from oculomotor.analysis import ni_net
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)


def measure(P, amp):
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
    acts = read_brain_acts(st, P)
    ver = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    ni  = np.array(ni_net(st))[:, 0]
    z   = extract_z_opn(st)
    win = (t >= t_jump + 0.05) & (t <= 0.55) & (z >= 50.0)
    pk  = lambda x: float(np.max(np.abs(x[win]))) if win.any() else float('nan')
    fl_drive = np.array(acts.cb.fl_drive)[:, 0]
    pred_err = np.array(acts.cb.pred_err)[:, 0]
    sat_sc   = np.array(acts.cb.saccadic_suppression_scene)
    scene_mm = sat_sc * np.array(acts.pc.scene_angular_vel)[:, 0] + np.array(acts.cb.fl_okr_drive)[:, 0]
    return dict(ni=pk(np.gradient(ni, DT)), ring=pk(np.gradient(ver, DT)),
                target_mm=pk(pred_err), scene_mm=pk(scene_mm), fl=pk(fl_drive))


print('=== EC mismatch + NI drift vs saccade size (closed loop) ===')
print(f'{"amp":>4} {"NI_vel":>7} {"ring":>6} {"target_mm":>10} {"scene_mm":>9} {"fl_drive":>9}')
for amp in (2.0, 5.0, 10.0, 20.0):
    m = measure(THETA_NOISELESS, amp)
    print(f'{amp:4.0f} {m["ni"]:7.2f} {m["ring"]:6.2f} {m["target_mm"]:10.3f} {m["scene_mm"]:9.3f} {m["fl"]:9.2f}')

print('\n=== 2 deg drift: which path carries it? ===')
paths = {
    'closed':       THETA_NOISELESS,
    'open pursuit': with_brain(THETA_NOISELESS, K_pursuit=0., K_phasic_pursuit=0.,
                               K_pursuit_direct=0., K_cereb_pu=0.),
    'open OKN/VS':  with_brain(THETA_NOISELESS, K_vor_direct=0., K_cereb_okr=0.),
    'open both':    with_brain(THETA_NOISELESS, K_pursuit=0., K_phasic_pursuit=0.,
                               K_pursuit_direct=0., K_cereb_pu=0., K_vor_direct=0., K_cereb_okr=0.),
}
for label, P in paths.items():
    m = measure(P, 2.0)
    print(f'  {label:14} NI_vel={m["ni"]:6.2f}  ring={m["ring"]:5.2f}  '
          f'target_mm={m["target_mm"]:.3f}  scene_mm={m["scene_mm"]:.3f}  fl={m["fl"]:.2f}')
