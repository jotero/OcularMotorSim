"""Part 4: is the overshoot fixable while keeping a CLINICALLY VALID AC/A ratio (~3-5 pd/D)?
Map overshoot vs AC_A. Geometric near (0.4 m) = 9.15 deg; small steady lag is physiological."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)
print("AC_A default:", CLEAN.brain.AC_A)

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
    return dict(over=over, pv=pv, ss=ss)

print("\nAC_A sweep at baseline disparity gains (K_verg=2.5,K_phasic=3.0):")
print(f"  {'AC_A':>4} | {'CONV over':>9} {'pv':>5} {'ss':>6}")
for aca in (0,1,2,3,4,5):
    rc = run_step(with_brain(CLEAN, AC_A=float(aca)), True)
    print(f"  {aca:4.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f}")

print("\nAC_A sweep with detuned integrator (K_verg=1.0,K_phasic=5.0,tau_verg=2.0):")
print(f"  {'AC_A':>4} | {'CONV over':>9} {'pv':>5} {'ss':>6}")
for aca in (0,1,2,3,4,5):
    p = with_brain(CLEAN, AC_A=float(aca), K_verg=1.0, K_phasic_verg=5.0, tau_verg=2.0)
    rc = run_step(p, True)
    print(f"  {aca:4.1f} | {rc['over']*100:8.1f}% {rc['pv']:5.1f} {rc['ss']:6.2f}")
