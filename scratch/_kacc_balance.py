"""Balance K_cereb_acc: 1.5 over-damps accommodation (bw halved). Find a value that
keeps accommodation fast (peak vel ~3.8, low overshoot) AND vergence clean.
K_cereb_verg=1.5, K_phasic_verg=12 fixed."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)

def acc_step(params):
    p = with_brain(params, AC_A=0.0, CA_C=0.0)
    T0,TOT=1.0,4.0; t=np.arange(0,TOT,DT); N=len(t)
    pt=np.tile([0,0,6.0],(N,1)).astype(np.float32); pt[(t>=T0)&(t<3.0)]=[0,0,0.4]
    st=simulate(p,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),return_states=True)
    a=np.array(st.acc_plant[:,0]); m=(t>=T0)&(t<3.0)
    s=float(np.mean(a[(t>T0-0.2)&(t<T0)])); ss=float(np.mean(a[(t>2.5)&(t<3.0)])); ch=ss-s
    over=(float(np.max(a[m]))-ss)/ch*100; pv=float(np.max(np.abs(np.gradient(a,DT)[m])))
    thr=s+0.1*ch; cr=np.where((t>=T0)&(a>=thr))[0]; lat=(t[cr[0]]-T0)*1000 if len(cr) else np.nan
    return over, ss, pv, lat

def verg_step(params):
    T0,TOT=1.0,3.5; t=np.arange(0,TOT,DT); N=len(t)
    pt=np.tile([0,0,3.0],(N,1)).astype(np.float32); pt[t>=T0]=[0,0,0.4]
    st=simulate(params,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),target_present_array=np.ones(N),return_states=True)
    v=np.array(st.plant.left[:,0]-st.plant.right[:,0]); m=t>=T0
    ss=float(np.mean(v[t>TOT-0.3])); s0=float(v[int(T0/DT)-1]); over=(float(np.max(v[m]))-ss)/(ss-s0+1e-9)*100
    pv=float(np.max(np.abs(np.gradient(v,DT)[m])))
    return over, ss, pv

print("baseline accommodation step (no forward model): peak_vel ~3.8, over ~12%")
print(f"  {'K_acc':>5} | {'acc over':>8} {'acc pv':>6} {'acc lat':>7} | {'verg over':>9} {'verg pv':>7}")
for ka in (0.0, 0.6, 0.8, 1.0, 1.5):
    p = with_brain(CLEAN, K_cereb_acc=ka, K_cereb_verg=1.5, K_phasic_verg=12.0)
    ao, ass, apv, alat = acc_step(p); vo, vss, vpv = verg_step(p)
    print(f"  {ka:5.1f} | {ao:7.1f}% {apv:6.2f} {alat:6.0f}ms | {vo:8.1f}% {vpv:7.1f}")
