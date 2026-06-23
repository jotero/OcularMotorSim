"""Diagnose the residual: (1) is 50s settled? run long. (2) does accommodation rest
at tonic_acc in the dark with cross-links OFF, or is a dark-defocus signal pulling it?"""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)
DT=0.001
def dark_trace(params, T=150.0):
    t=np.arange(0,T,DT); N=len(t)
    st=simulate(params,t,scene_present_array=np.zeros(N),target_present_array=np.zeros(N),return_states=True)
    v=np.array(st.plant.left[:,0]-st.plant.right[:,0]); a=np.array(st.acc_plant[:,0])
    return t, v, a

print("tonic_verg=3.67, tonic_acc=1.0\n")
print("PHASIC cross-links (current) — dark state vs time:")
t,v,a = dark_trace(CLEAN)
for tt in (20,50,100,149):
    i=int(tt/DT); print(f"  t={tt:4d}s : verg {v[i]:6.2f}deg  focus {a[i]:5.2f}D")

print("\nCROSS-LINKS OFF (AC_A=0,CA_C=0) — does accommodation rest at tonic_acc=1.0?")
t,v,a = dark_trace(with_brain(CLEAN, AC_A=0.0, CA_C=0.0))
for tt in (20,50,100,149):
    i=int(tt/DT); print(f"  t={tt:4d}s : verg {v[i]:6.2f}deg  focus {a[i]:5.2f}D")
