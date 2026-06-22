"""Part 2: is the residual overshoot floor delay-driven (direct path on stale disparity)?
Sweep K_phasic; then test sensitivity to the disparity smoothing delay."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

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
    return dict(over=over, pv=pv, ss=ss, change=change)

print("disparity smoothing TC default:", CLEAN.brain.tau_vis_smooth_disparity)

print("\n(A) K_phasic sweep at K_verg=1.2, tau_verg=2.0:")
print(f"  {'K_phasic':>8} | {'CONV over':>9} {'pv':>5} {'ss':>6}")
for kp in (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0):
    p = with_brain(CLEAN, K_verg=1.2, tau_verg=2.0, K_phasic_verg=kp)
    rc = run_step(p, True)
    print(f"  {kp:8.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f}")

print("\n(B) delay sensitivity (K_verg=1.2, K_phasic=3.0, tau_verg=2.0):")
print(f"  {'tau_disp':>8} | {'CONV over':>9} {'pv':>5} {'lat?':>5}")
for td in (0.001, 0.02, 0.04, 0.06, 0.08):
    p = with_brain(CLEAN, K_verg=1.2, tau_verg=2.0, tau_vis_smooth_disparity=td)
    rc = run_step(p, True)
    print(f"  {td:8.3f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f}")

print("\n(C) low-gain non-overshoot candidates (K_phasic, K_verg, tau_verg):")
print(f"  {'Kph':>4} {'Kv':>4} {'tv':>4} | {'CONV over':>9} {'pv':>5} {'ss':>6} | {'DIV over':>8} {'pv':>5} {'ss':>6}")
for kp, kv, tv in [(1.5,0.8,2.0),(1.0,0.8,2.0),(1.5,1.0,1.5),(2.0,0.6,2.0),(1.0,1.0,1.5),(1.2,0.6,2.0)]:
    p = with_brain(CLEAN, K_phasic_verg=kp, K_verg=kv, tau_verg=tv)
    rc = run_step(p, True); rd = run_step(p, False)
    print(f"  {kp:4.1f} {kv:4.1f} {tv:4.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f} | {rd['over']*100:7.1f}% {rd['pv']:5.1f} {rd['ss']:6.2f}")
