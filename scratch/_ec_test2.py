"""Test BOTH cerebellar near-response forward models: K_cereb_acc (accommodation)
and K_cereb_verg (vergence/disparity). Does predicting both channels close the
coupled vergence overshoot toward the AC/A-off floor (~4%)?"""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

def verg_step(params, conv=True):
    T0, TOTAL = 1.0, 3.5
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    d0, d1 = (3.0, 0.4) if conv else (0.4, 3.0)
    pt = np.tile([0.0, 0.0, d0], (T, 1)).astype(np.float32)
    pt[t >= T0] = [0.0, 0.0, d1]
    st = simulate(params, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), target_present_array=np.ones(T),
                  return_states=True)
    v = np.array(st.plant.left[:, 0] - st.plant.right[:, 0])
    a = np.array(st.acc_plant[:, 0]); m = t >= T0
    def ov(x):
        ss = float(np.mean(x[t > TOTAL-0.3])); pk = (float(np.max(x[m])) if (np.mean(x[t>TOTAL-0.3])>x[int(T0/DT)-1]) else float(np.min(x[m])))
        s0 = float(x[int(T0/DT)-1]); return (pk-ss)/(ss-s0+1e-9)*100, ss
    vel = np.gradient(v, DT); pv = float(np.max(np.abs(vel[m])))
    vo, vss = ov(v); ao, ass = ov(a)
    return vo, vss, ao, pv

print("CONVERGENCE STEP — sweep both forward-model gains:")
print(f"  {'K_acc':>5} {'K_verg':>6} | {'acc over':>8} | {'verg over':>9} {'verg ss':>7} {'pv':>5}")
for ka, kv in [(0.0,0.0),(1.0,0.0),(0.0,1.0),(1.0,1.0),(1.0,1.5),(1.5,1.5),(1.0,2.0)]:
    vo, vss, ao, pv = verg_step(with_brain(CLEAN, K_cereb_acc=ka, K_cereb_verg=kv), True)
    print(f"  {ka:5.1f} {kv:6.1f} | {ao:7.1f}% | {vo:8.1f}% {vss:7.2f} {pv:5.1f}")

print("\n  (AC/A-off floor for reference):")
vo, vss, ao, pv = verg_step(with_brain(CLEAN, AC_A=0.0), True)
print(f"  AC_A=0           | {ao:7.1f}% | {vo:8.1f}% {vss:7.2f} {pv:5.1f}")

print("\nDIVERGENCE STEP (default both ON):")
vo, vss, ao, pv = verg_step(CLEAN, False)
print(f"  verg over={vo:.1f}%  verg ss={vss:.2f}  acc over={ao:.1f}%")
