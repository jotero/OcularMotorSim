"""Does CA/C-off in the isolated accommodation step matter? Compare the accommodation
step (far->near) isolated (benchmark: AC_A=0, CA_C=0) vs the COUPLED near-triad
(AC_A=5, CA_C=0.08). Also check the coupled accommodation overshoot now that the
vergence loop is fixed (it re-injected the ring before)."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)

def acc_step(params):
    T0,TOT=1.0,4.0; t=np.arange(0,TOT,DT); N=len(t)
    pt=np.tile([0,0,6.0],(N,1)).astype(np.float32); pt[(t>=T0)&(t<3.0)]=[0,0,0.4]
    st=simulate(params,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),
                target_present_array=np.ones(N),return_states=True)
    a=np.array(st.acc_plant[:,0]); m=(t>=T0)&(t<3.0)
    s=float(np.mean(a[(t>T0-0.2)&(t<T0)])); ss=float(np.mean(a[(t>2.5)&(t<3.0)])); ch=ss-s
    over=(float(np.max(a[m]))-ss)/ch*100; pv=float(np.max(np.abs(np.gradient(a,DT)[m])))
    thr=s+0.1*ch; cr=np.where((t>=T0)&(a>=thr))[0]; lat=(t[cr[0]]-T0)*1000 if len(cr) else np.nan
    # vergence too (is the coupled triad clean?)
    vg=np.array(st.plant.left[:,0]-st.plant.right[:,0])
    vss=float(np.mean(vg[(t>2.5)&(t<3.0)])); v0=float(vg[int(T0/DT)-1])
    vover=(float(np.max(vg[m]))-vss)/(vss-v0+1e-9)*100 if abs(vss-v0)>0.5 else 0.0
    return over, ss, pv, lat, vover

print("Accommodation step (far 6m -> near 0.4m):")
print(f"  {'config':34s} | {'acc over':>8} {'acc ss':>6} {'acc pv':>6} {'acc lat':>7} | {'verg over':>9}")
for tag,p in [
    ("isolated  (AC_A=0, CA_C=0)  [bench]", with_brain(CLEAN, AC_A=0.0, CA_C=0.0)),
    ("CA/C only (AC_A=0, CA_C=0.08)",       with_brain(CLEAN, AC_A=0.0, CA_C=0.08)),
    ("coupled   (AC_A=5, CA_C=0.08) [real]",CLEAN),
    ("CA/C high (AC_A=5, CA_C=0.16)",       with_brain(CLEAN, CA_C=0.16)),
]:
    o,ss,pv,lat,vo = acc_step(p)
    print(f"  {tag:34s} | {o:7.1f}% {ss:6.2f} {pv:6.2f} {lat:6.0f}ms | {vo:8.1f}%")
