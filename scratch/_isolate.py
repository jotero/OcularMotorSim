"""Is my near-response change isolated? Run deterministic non-near sims with the
FULL new params vs params NEUTRALIZED (K_cereb_acc=K_cereb_verg=0, K_phasic_verg=3).
Both have the new cerebellum states, so identical output => my change touches only
the near response; non-near behavior (VOR, OVAR) is unaffected."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
# noiseless so the comparison is exact
BASE = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)
OLDVAL = with_brain(BASE, K_cereb_acc=0.0, K_cereb_verg=0.0, K_phasic_verg=3.0)

def vor_dark(params):
    head = km.head_rotation_step(60.0, 6.0, coast_dur=2.0)
    t = head.t; N = len(t)
    st = simulate(params, t, head=head,
                  scene_present_array=np.zeros(N), target_present_array=np.zeros(N),
                  return_states=True)
    return np.array(st.plant.left[:,0]), np.array(st.acc_plant[:,0])

def vor_light(params):
    head = km.head_rotation_step(45.0, 8.0, coast_dur=2.0)
    t = head.t; N = len(t)
    st = simulate(params, t, head=head,
                  scene_present_array=np.ones(N), target_present_array=np.zeros(N),
                  return_states=True)
    return np.array(st.plant.left[:,0])

for name, fn in [("VOR-dark eye-pos", lambda p: vor_dark(p)[0]),
                 ("VOR-dark accom",   lambda p: vor_dark(p)[1]),
                 ("VVOR eye-pos",     vor_light)]:
    a = fn(BASE); b = fn(OLDVAL)
    d = np.abs(a - b); imax = int(np.argmax(d))
    print(f"  {name:18s}: max={float(d.max()):.3e} @t={imax*DT:.2f}s | "
          f"diff@1s={float(d[min(1000,len(d)-1)]):.3e} | diff@end={float(d[-1]):.3e}")
