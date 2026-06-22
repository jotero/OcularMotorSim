"""With BOTH forward models active, can we crank the vergence loop gain to recover
Hung Fig 1 peak velocity (~30 deg/s, latency ~150 ms) while KEEPING overshoot low?
That's the prediction payoff: fast AND clean (vs damping = slow)."""
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
    v = np.array(st.plant.left[:, 0] - st.plant.right[:, 0]); m = t >= T0
    ss = float(np.mean(v[t > TOTAL-0.3])); s0 = float(v[int(T0/DT)-1])
    over = (float(np.max(v[m]))-ss)/(ss-s0+1e-9)*100
    vel = np.gradient(v, DT); pv = float(np.max(np.abs(vel[m])))
    thr = s0 + 0.1*(ss-s0); cr = np.where((t>=T0)&(v>=thr))[0]
    lat = (t[cr[0]]-T0)*1000 if len(cr) else float('nan')
    return over, ss, pv, lat

print("Hung Fig 1 conv target: pv ~30 deg/s, lat ~150 ms, overshoot ~0")
print("ECs at K_cereb_acc=1.5, K_cereb_verg=1.5; sweep loop gain:\n")
print(f"  {'K_phasic':>8} {'K_verg':>6} | {'over':>6} {'ss':>5} {'pv':>5} {'lat':>5}")
base = with_brain(CLEAN, K_cereb_acc=1.5, K_cereb_verg=1.5)
for kph, kv in [(3.0,2.5),(5.0,2.5),(7.0,2.5),(9.0,3.0),(9.0,4.0),(12.0,4.0),(12.0,2.5)]:
    over, ss, pv, lat = verg_step(with_brain(base, K_phasic_verg=kph, K_verg=kv))
    print(f"  {kph:8.1f} {kv:6.1f} | {over:5.1f}% {ss:5.2f} {pv:5.1f} {lat:4.0f}")
