"""Part 3: (i) push K_phasic high / K_verg low for min overshoot + peak vel;
(ii) does AC/A cross-coupling add to overshoot? (iii) full trace of best candidate."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)
print("tau_p:", CLEAN.plant.tau_p)

def run_step(params, conv=True):
    T_BASE, TOTAL = 1.0, 3.5
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    d0, d1 = (3.0, 0.4) if conv else (0.4, 3.0)
    pt = np.tile([0.0, 0.0, d0], (T, 1)).astype(np.float32)
    pt[t >= T_BASE] = [0.0, 0.0, d1]
    st = simulate(params, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), target_present_array=np.ones(T),
                  return_states=True)
    verg = np.array(st.plant.left[:, 0] - st.plant.right[:, 0])
    vel  = np.gradient(verg, DT)
    start = float(np.mean(verg[(t > T_BASE - 0.2) & (t < T_BASE)]))
    ss    = float(np.mean(verg[t > TOTAL - 0.3]))
    change = ss - start
    seg = verg[t >= T_BASE]
    peak = float(np.max(seg)) if change > 0 else float(np.min(seg))
    over = float(max(0.0, (peak - ss) / change)) if abs(change) > 0.5 else float('nan')
    pv = float(np.max(np.abs(vel[t >= T_BASE])))
    thr = start + 0.1 * change
    cr = np.where((t >= T_BASE) & ((verg >= thr) if change > 0 else (verg <= thr)))[0]
    lat = float((t[cr[0]] - T_BASE) * 1000.0) if len(cr) else float('nan')
    return dict(over=over, pv=pv, ss=ss, change=change, lat=lat)

print("\n(i) aggressive K_phasic high / K_verg low:")
print(f"  {'Kph':>4} {'Kv':>4} | {'CONV over':>9} {'pv':>5} {'ss':>6} {'lat':>5}")
for kp, kv in [(4,0.8),(6,0.8),(6,0.5),(8,0.5),(8,0.3),(10,0.3),(6,0.0),(10,0.0)]:
    p = with_brain(CLEAN, K_phasic_verg=kp, K_verg=kv, tau_verg=2.0)
    rc = run_step(p, True)
    print(f"  {kp:4.1f} {kv:4.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f} {rc['lat']:5.0f}")

print("\n(ii) AC/A & CA/C cross-coupling effect on overshoot (baseline gains):")
for tag, p in [('full (AC/A on)', CLEAN),
               ('AC_A=0', with_brain(CLEAN, AC_A=0.0)),
               ('AC_A=0,CA_C=0', with_brain(CLEAN, AC_A=0.0, CA_C=0.0))]:
    rc = run_step(p, True)
    print(f"  {tag:16s}: over={rc['over']*100:5.1f}%  pv={rc['pv']:5.1f}  ss={rc['ss']:5.2f}")

print("\n(iii) AC_A=0 + aggressive tune (decouple, then min overshoot):")
print(f"  {'Kph':>4} {'Kv':>4} | {'CONV over':>9} {'pv':>5} {'ss':>6} | {'DIV over':>8} {'pv':>5}")
for kp, kv in [(6,0.5),(8,0.3),(6,0.0),(8,0.0),(10,0.0)]:
    p = with_brain(CLEAN, AC_A=0.0, K_phasic_verg=kp, K_verg=kv, tau_verg=2.0)
    rc = run_step(p, True); rd = run_step(p, False)
    print(f"  {kp:4.1f} {kv:4.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f} | {rd['over']*100:7.1f}% {rd['pv']:5.1f}")
