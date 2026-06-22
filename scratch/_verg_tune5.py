"""Part 5 — decisive test of the user's insight:
Is the vergence overshoot caused by ACCOMMODATION'S LAG fed through AC/A?
If yes: speeding up accommodation (small tau_acc_fast / tau_acc_plant) should
remove the vergence overshoot WITHOUT changing AC_A. That is exactly what a
forward-model/PREDICTION on accommodation would buy: AC/A reads a de-lagged
accommodation, so the cross-coupling stops overshooting vergence."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)
print("tau_acc_fast:", CLEAN.brain.tau_acc_fast, " tau_acc_plant:", CLEAN.brain.tau_acc_plant)

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
    start = float(np.mean(verg[(t > T_BASE - 0.2) & (t < T_BASE)]))
    ss    = float(np.mean(verg[t > TOTAL - 0.3]))
    change = ss - start
    seg = verg[t >= T_BASE]
    peak = float(np.max(seg)) if change > 0 else float(np.min(seg))
    over = float(max(0.0, (peak - ss) / change)) if abs(change) > 0.5 else float('nan')
    return dict(over=over, ss=ss)

print("\nAccommodation speed vs VERGENCE overshoot (AC_A=5 fixed, baseline verg gains):")
print(f"  {'tau_af':>6} {'tau_ap':>6} | {'CONV verg over':>14} {'ss':>6}")
for taf, tap in [(0.30,0.156),(0.15,0.156),(0.08,0.080),(0.04,0.040),(0.01,0.010),(0.002,0.002)]:
    p = with_brain(CLEAN, tau_acc_fast=taf, tau_acc_plant=tap)
    rc = run_step(p, True)
    print(f"  {taf:6.3f} {tap:6.3f} | {rc['over']*100:13.1f}% {rc['ss']:6.2f}")

print("\n  control: AC_A=0 (no coupling at all):",
      f"{run_step(with_brain(CLEAN, AC_A=0.0), True)['over']*100:.1f}% overshoot")
