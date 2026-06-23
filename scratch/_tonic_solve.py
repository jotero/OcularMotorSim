"""Verify the analytic inversion: tonic_verg=-1.13, tonic_acc=0.68 should give a
dark state of ~1 m (verg 3.67deg, focus 1.0 D). Confirm target-driven vergence and
AC/A unaffected. Also test a positive-constrained alternative."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)

def dark_rest(params):
    T=50.0; t=np.arange(0,T,DT); N=len(t)
    st=simulate(params,t,scene_present_array=np.zeros(N),target_present_array=np.zeros(N),return_states=True)
    return (float(np.mean((st.plant.left[:,0]-st.plant.right[:,0])[t>T-1.0])),
            float(np.mean(st.acc_plant[:,0][t>T-1.0])))
def tverg(params,d):
    T=4.0; t=np.arange(0,T,DT); N=len(t)
    pt=np.tile([0,0,d],(N,1)).astype(np.float32)
    st=simulate(params,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),
                target_present_array=np.ones(N),return_states=True)
    return float(np.mean((st.plant.left[:,0]-st.plant.right[:,0])[t>T-0.5]))

print(f"  {'tonic_verg':>10} {'tonic_acc':>9} | {'DARK verg':>9} {'dark foc':>8} | {'near 0.4m':>9} {'far 3m':>7}")
for tv,ta,lbl in [(3.67,1.0,'current'),(-1.13,0.68,'->1m exact'),(0.0,0.5,'tv>=0 alt'),(-0.5,0.6,'mild neg')]:
    p=with_brain(CLEAN, tonic_verg=float(tv), tonic_acc=float(ta))
    dv,df=dark_rest(p); nv=tverg(p,0.4); fv=tverg(p,3.0)
    print(f"  {tv:10.2f} {ta:9.2f} | {dv:8.2f}deg {df:7.2f}D | {nv:8.2f}deg {fv:6.2f}deg  ({lbl})")
