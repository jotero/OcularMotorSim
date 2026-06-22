"""Diagnose the ACCOMMODATION overshoot (the real root cause).
Is it the accommodation loop itself, or CA/C from vergence? And does removing
it also clean up vergence (via AC/A)?"""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

def run(params):
    T0, TOTAL = 1.0, 4.0
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    pt = np.tile([0.0, 0.0, 3.0], (T, 1)).astype(np.float32)
    pt[t >= T0] = [0.0, 0.0, 0.4]
    st = simulate(params, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), target_present_array=np.ones(T),
                  return_states=True)
    verg = np.array(st.plant.left[:, 0] - st.plant.right[:, 0])
    acc  = np.array(st.acc_plant[:, 0])
    def over(x):
        m = t >= T0; ss = float(np.mean(x[t > TOTAL-0.3])); pk = float(np.max(x[m]))
        st0 = float(x[int(T0/DT)-1])
        return (pk - ss)/(ss - st0 + 1e-9)*100, ss
    vo, vss = over(verg); ao, ass = over(acc)
    return vo, vss, ao, ass

print(f"{'condition':46s} | {'acc over':>8} {'acc ss':>6} | {'verg over':>9} {'verg ss':>7}")
def show(tag, p):
    vo, vss, ao, ass = run(p)
    print(f"{tag:46s} | {ao:7.1f}% {ass:6.2f} | {vo:8.1f}% {vss:7.2f}")

show("default (pred on)", CLEAN)
show("CA_C=0 (no verg->acc)", with_brain(CLEAN, CA_C=0.0))
show("AC_A=0,CA_C=0 (isolated acc loop)", with_brain(CLEAN, AC_A=0.0, CA_C=0.0))
print("  -> if acc still overshoots with CA_C=0, it's the accommodation loop itself\n")

print("What damps the accommodation loop? (AC_A=0,CA_C=0 isolated)")
base = with_brain(CLEAN, AC_A=0.0, CA_C=0.0)
for tag, p in [
    ("K_acc_fast=2.5 (default)", base),
    ("K_acc_fast=1.5", with_brain(base, K_acc_fast=1.5)),
    ("K_acc_fast=1.0", with_brain(base, K_acc_fast=1.0)),
    ("K_acc_fast=0.5", with_brain(base, K_acc_fast=0.5)),
    ("tau_acc_plant=0.30 (slower plant)", with_brain(base, tau_acc_plant=0.30)),
    ("tau_acc_plant=0.08 (faster plant)", with_brain(base, tau_acc_plant=0.08)),
]:
    vo, vss, ao, ass = run(p)
    print(f"  {tag:40s} acc over={ao:6.1f}%  acc ss={ass:5.2f}")
