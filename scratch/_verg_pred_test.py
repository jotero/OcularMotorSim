"""Test the forward-predicted cross-links (K_pred_acc / K_pred_verg).
Expect: vergence step overshoot 32% -> ~7% with steady state + AC/A ratio intact."""
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
    thr = start + 0.1 * change
    cr = np.where((t >= T_BASE) & ((verg >= thr) if change > 0 else (verg <= thr)))[0]
    lat = float((t[cr[0]] - T_BASE) * 1000.0) if len(cr) else float('nan')
    return dict(over=over, pv=pv, ss=ss, lat=lat)

print("Vergence step (8 deg, 3m<->0.4m), geometric near = 9.15 deg:\n")
configs = [
    ("prediction ON  (default K_pred=1.0)", CLEAN),
    ("prediction OFF (K_pred=0, old model)", with_brain(CLEAN, K_pred_acc=0.0, K_pred_verg=0.0)),
    ("K_pred_acc=0.5", with_brain(CLEAN, K_pred_acc=0.5)),
]
for tag, p in configs:
    rc = run_step(p, True); rd = run_step(p, False)
    print(f"{tag}")
    print(f"   CONV: over={rc['over']*100:5.1f}%  pv={rc['pv']:5.1f}  ss={rc['ss']:5.2f}  lat={rc['lat']:4.0f}ms")
    print(f"   DIV : over={rd['over']*100:5.1f}%  pv={rd['pv']:5.1f}  ss={rd['ss']:5.2f}  lat={rd['lat']:4.0f}ms\n")

# AC/A ratio check: lens-driven convergence (open accommodation via lens, AC/A drives vergence).
# Simplest proxy: steady-state vergence change for a near step should be AC/A-consistent.
# Compare prediction on/off steady states — should match (error->0 at SS).
print("Steady-state AC/A preserved? conv ss  pred-ON vs pred-OFF:")
print(f"   ON ={run_step(CLEAN, True)['ss']:.3f}   OFF={run_step(with_brain(CLEAN, K_pred_acc=0.0, K_pred_verg=0.0), True)['ss']:.3f}")
