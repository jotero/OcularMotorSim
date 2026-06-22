"""Tune vergence smooth-path gains to match Hung 1997 Fig 1: smooth, non-overshooting
convergence + divergence steps. ~9 deg step (far 3 m -> near 0.4 m). NOISELESS.

Hung Fig 1 targets (~8-9 deg step):
  convergence: latency ~150-180 ms, peak vel ~30-32 deg/s, NO overshoot, settle ~0.7 s
  divergence : latency ~200 ms,     peak vel ~16-18 deg/s, NO overshoot, settle ~1.3 s
"""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

def run_step(params, conv=True):
    T_BASE, T_HOLD, TOTAL = 1.0, 2.5, 3.5
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
    geo = abs(change)  # we report endpoint vs geometric below
    return dict(start=start, ss=ss, change=change, over=over, pv=pv, lat=lat)

# geometric target for 0.4 m, IPD 0.064:
geo_near = 2 * np.degrees(np.arctan(0.032 / 0.4))
print(f"geometric near vergence (0.4 m) = {geo_near:.2f} deg\n")

print("baseline (K_verg=2.5, K_phasic=3.0, tau_verg=3.0):")
for conv in (True, False):
    r = run_step(CLEAN, conv)
    lbl = 'CONV' if conv else 'DIV '
    print(f"  {lbl}: over={r['over']*100:5.1f}%  pv={r['pv']:5.1f} deg/s  lat={r['lat']:5.0f} ms  ss={r['ss']:6.2f}  change={r['change']:6.2f}")

print("\nsweep K_verg x tau_verg (K_phasic=3.0):")
print(f"  {'K_verg':>6} {'tau_v':>5} | {'CONV over':>9} {'pv':>5} {'ss':>6} | {'DIV over':>8} {'pv':>5} {'ss':>6}")
for kv in (0.8, 1.2, 1.6, 2.0, 2.5):
    for tv in (1.0, 1.5, 2.0, 3.0):
        p = with_brain(CLEAN, K_verg=kv, tau_verg=tv)
        rc = run_step(p, True); rd = run_step(p, False)
        print(f"  {kv:6.1f} {tv:5.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f} | {rd['over']*100:7.1f}% {rd['pv']:5.1f} {rd['ss']:6.2f}")
