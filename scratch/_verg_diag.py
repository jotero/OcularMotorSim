"""Diagnose the actual overshoot mechanism: does the AC/A DRIVE peak (overshoot)
and inject into vergence? Does accommodation itself overshoot? Print time courses."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
DEG_PER_PD = 0.5729
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

def run(params):
    T_BASE, TOTAL = 1.0, 3.5
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    pt = np.tile([0.0, 0.0, 3.0], (T, 1)).astype(np.float32)
    pt[t >= T_BASE] = [0.0, 0.0, 0.4]
    st = simulate(params, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), target_present_array=np.ones(T),
                  return_states=True)
    verg = np.array(st.plant.left[:, 0] - st.plant.right[:, 0])
    acc  = np.array(st.acc_plant[:, 0])
    af   = np.array(st.brain.va.acc_fast)
    aslw = np.array(st.brain.va.acc_slow)
    apred= np.array(st.brain.va.acc_pred)
    aca_state = params.brain.AC_A * DEG_PER_PD * (af + aslw)   # OLD AC/A read
    aca_pred  = params.brain.AC_A * DEG_PER_PD * apred         # NEW AC/A read
    return t, verg, acc, aca_state, aca_pred, T_BASE

def summ(tag, params, use_pred):
    t, verg, acc, aca_s, aca_p, T0 = run(params)
    m = t >= T0
    ss = float(np.mean(verg[t > 3.2]))
    vpk = float(np.max(verg[m]))
    # accommodation overshoot
    acc_ss = float(np.mean(acc[t > 3.2])); acc_pk = float(np.max(acc[m]))
    acc_over = (acc_pk - acc_ss) / (acc_ss - acc[int(T0/DT)-1] + 1e-9)
    aca = aca_p if use_pred else aca_s
    aca_ss = float(np.mean(aca[t > 3.2])); aca_pk = float(np.max(aca[m]))
    # time of vergence peak vs aca peak
    tvpk = t[m][np.argmax(verg[m])]; tapk = t[m][np.argmax(aca[m])]
    print(f"{tag}")
    print(f"   verg: ss={ss:5.2f} peak={vpk:5.2f} over={(vpk-ss)/(ss-verg[int(T0/DT)-1])*100:5.1f}%  t_peak={tvpk-T0:.3f}s")
    print(f"   acc : ss={acc_ss:5.2f} peak={acc_pk:5.2f} over={acc_over*100:5.1f}%")
    print(f"   AC/A drive: ss={aca_ss:5.2f} peak={aca_pk:5.2f} over={(aca_pk-aca_ss)/(aca_ss+1e-9)*100:5.1f}%  t_peak={tapk-T0:.3f}s")
    print()

summ("OLD (pred off, AC/A reads realized state)", with_brain(CLEAN, K_pred_acc=0.0, K_pred_verg=0.0), use_pred=False)
summ("NEW (pred on,  AC/A reads acc_pred)",        CLEAN, use_pred=True)
summ("AC/A OFF",                                   with_brain(CLEAN, AC_A=0.0), use_pred=False)
summ("FAST ACC (tau_acc_fast=0.04, pred off)",     with_brain(CLEAN, K_pred_acc=0.0, K_pred_verg=0.0, tau_acc_fast=0.04, tau_acc_plant=0.04), use_pred=False)
