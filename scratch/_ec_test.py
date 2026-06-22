"""Test the cerebellar accommodation EC forward model (K_cereb_acc).
Expect: accommodation stops ringing -> vergence overshoot drops toward the
AC/A-off floor (~4%), WITHOUT losing accommodation-step speed."""
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
    thr = float(v[int(T0/DT)-1]) + 0.1*(float(np.mean(v[t>TOTAL-0.3]))-float(v[int(T0/DT)-1]))
    cr = np.where((t>=T0) & (v>=thr))[0]; lat = (t[cr[0]]-T0)*1000 if len(cr) else float('nan')
    vo, vss = ov(v); ao, ass = ov(a)
    return vo, vss, ao, ass, pv, lat

def acc_step(params):
    p = with_brain(params, AC_A=0.0, CA_C=0.0)
    T0, TOTAL = 1.0, 4.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    pt = np.tile([0.0, 0.0, 6.0], (T, 1)).astype(np.float32)
    pt[(t >= T0) & (t < 3.0)] = [0.0, 0.0, 0.4]
    st = simulate(p, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), return_states=True)
    a = np.array(st.acc_plant[:, 0]); m = (t >= T0) & (t < 3.0)
    start = float(np.mean(a[(t > T0-0.2) & (t < T0)])); ss = float(np.mean(a[(t>2.5)&(t<3.0)]))
    change = ss - start; pk = float(np.max(a[m])); over = (pk-ss)/change*100
    av = np.gradient(a, DT); pvel = float(np.max(np.abs(av[m])))
    thr = start + 0.1*change; cr = np.where((t>=T0) & (a>=thr))[0]
    lat = (t[cr[0]]-T0)*1000 if len(cr) else float('nan')
    return over, ss, pvel, lat

print("VERGENCE STEP (full model):")
print(f"  {'config':28s} | {'acc over':>8} | {'verg over':>9} {'verg ss':>7} {'pv':>5} {'lat':>5}")
for tag, kc in [("K_cereb_acc=1.0 (default, ON)", 1.0),
                ("K_cereb_acc=0.0 (lesion/OFF)", 0.0),
                ("K_cereb_acc=0.5", 0.5),
                ("K_cereb_acc=1.5", 1.5)]:
    vo, vss, ao, ass, pv, lat = verg_step(with_brain(CLEAN, K_cereb_acc=kc))
    print(f"  {tag:28s} | {ao:7.1f}% | {vo:8.1f}% {vss:7.2f} {pv:5.1f} {lat:4.0f}")

print("\nACCOMMODATION STEP (isolated) — does the EC keep speed?")
print(f"  {'config':28s} | {'over':>6} {'ss':>5} {'peak_vel':>8} {'lat_ms':>6}")
for tag, kc in [("K_cereb_acc=1.0 (ON)", 1.0), ("K_cereb_acc=0.0 (OFF)", 0.0)]:
    over, ss, pvel, lat = acc_step(with_brain(CLEAN, K_cereb_acc=kc))
    print(f"  {tag:28s} | {over:5.1f}% {ss:5.2f} {pvel:8.2f} {lat:6.0f}")
