"""Lower tonic_verg / tonic_acc to reduce dark over-convergence. Show:
 - dark vergence (no target, ~50s settle) drops,
 - near/far TARGET vergence unaffected (disparity loop dominates),
 - dark focus stays physiological."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001; IPD=0.064
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)
def geo(d): return 2*np.degrees(np.arctan(0.032/d))

def dark_rest(params):
    T=50.0; t=np.arange(0,T,DT); N=len(t)
    # dark: no scene, no target
    st=simulate(params,t,scene_present_array=np.zeros(N),target_present_array=np.zeros(N),return_states=True)
    v=float(np.mean((st.plant.left[:,0]-st.plant.right[:,0])[t>T-1.0]))
    a=float(np.mean(st.acc_plant[:,0][t>T-1.0]))
    return v, a

def target_verg(params, d):
    T=4.0; t=np.arange(0,T,DT); N=len(t)
    pt=np.tile([0,0,d],(N,1)).astype(np.float32)
    st=simulate(params,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),
                target_present_array=np.ones(N),return_states=True)
    return float(np.mean((st.plant.left[:,0]-st.plant.right[:,0])[t>T-0.5]))

print(f"geometric: near 0.4m={geo(0.4):.1f}deg  far 3m={geo(3):.1f}deg\n")
print(f"  {'tonic_verg':>10} {'tonic_acc':>9} | {'DARK verg':>9} {'dark foc':>8} | {'near verg':>9} {'far verg':>8}")
for tv, ta in [(3.67,1.0),(2.5,1.0),(2.5,0.8),(1.8,0.8),(1.5,0.7),(1.0,0.7)]:
    p=with_brain(CLEAN, tonic_verg=float(tv), tonic_acc=float(ta))
    dv,df = dark_rest(p)
    nv = target_verg(p,0.4); fv = target_verg(p,3.0)
    print(f"  {tv:10.2f} {ta:9.2f} | {dv:8.2f}° {df:7.2f}D | {nv:8.2f}° {fv:7.2f}°")
