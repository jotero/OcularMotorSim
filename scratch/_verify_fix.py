"""Verify: damping the accommodation loop (K_acc_fast) fixes the VERGENCE overshoot
end-to-end (full model), and measure the speed cost on the accommodation step
(which a forward-model/prediction would avoid)."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

def verg_step(params):
    T0, TOTAL = 1.0, 3.5
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    pt = np.tile([0.0, 0.0, 3.0], (T, 1)).astype(np.float32)
    pt[t >= T0] = [0.0, 0.0, 0.4]
    st = simulate(params, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), target_present_array=np.ones(T),
                  return_states=True)
    v = np.array(st.plant.left[:, 0] - st.plant.right[:, 0])
    a = np.array(st.acc_plant[:, 0]); m = t >= T0
    def ov(x):
        ss = float(np.mean(x[t > TOTAL-0.3])); pk = float(np.max(x[m])); s0 = float(x[int(T0/DT)-1])
        return (pk-ss)/(ss-s0+1e-9)*100, ss
    vel = np.gradient(v, DT); pv = float(np.max(np.abs(vel[m])))
    vo, vss = ov(v); ao, ass = ov(a)
    return vo, vss, ao, ass, pv

def acc_step(params):
    """Isolated accommodation step far->near (cross-links off)."""
    p = with_brain(params, AC_A=0.0, CA_C=0.0)
    T0, TOTAL = 1.0, 4.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    pt = np.tile([0.0, 0.0, 6.0], (T, 1)).astype(np.float32)
    pt[(t >= T0) & (t < 3.0)] = [0.0, 0.0, 0.4]
    st = simulate(p, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), return_states=True)
    a = np.array(st.acc_plant[:, 0]); m = (t >= T0) & (t < 3.0)
    start = float(np.mean(a[(t > T0-0.2) & (t < T0)])); ss = float(np.mean(a[(t>2.5)&(t<3.0)]))
    change = ss - start; pk = float(np.max(a[m]))
    over = (pk-ss)/change*100
    av = np.gradient(a, DT); pvel = float(np.max(np.abs(av[m])))
    thr = start + 0.1*change
    cr = np.where((t>=T0) & (a>=thr))[0]; lat = (t[cr[0]]-T0)*1000 if len(cr) else float('nan')
    return over, ss, pvel, lat

print("VERGENCE STEP (full model):")
print(f"  {'config':38s} | {'acc over':>8} | {'verg over':>9} {'verg ss':>7} {'pv':>5}")
for tag, p in [
    ("K_acc_fast=2.5, K_pred=0 (old-ish)", with_brain(CLEAN, K_acc_fast=2.5, K_pred_acc=0.0, K_pred_verg=0.0)),
    ("K_acc_fast=1.5, K_pred=0 (DAMP)",    with_brain(CLEAN, K_acc_fast=1.5, K_pred_acc=0.0, K_pred_verg=0.0)),
    ("K_acc_fast=1.0, K_pred=0 (DAMP)",    with_brain(CLEAN, K_acc_fast=1.0, K_pred_acc=0.0, K_pred_verg=0.0)),
    ("K_acc_fast=1.5, K_pred=1.0",         with_brain(CLEAN, K_acc_fast=1.5)),
]:
    vo, vss, ao, ass, pv = verg_step(p)
    print(f"  {tag:38s} | {ao:7.1f}% | {vo:8.1f}% {vss:7.2f} {pv:5.1f}")

print("\nACCOMMODATION STEP (isolated) — speed cost of damping:")
print(f"  {'K_acc_fast':>10} | {'over':>6} {'ss':>5} {'peak_vel':>8} {'lat_ms':>6}")
for kf in (2.5, 1.5, 1.0):
    over, ss, pvel, lat = acc_step(with_brain(CLEAN, K_acc_fast=kf))
    print(f"  {kf:10.1f} | {over:5.1f}% {ss:5.2f} {pvel:8.2f} {lat:6.0f}")
