"""Test a MONOCULAR accommodation step (full model, cross-links ON) as the
physiological way to isolate accommodation — vs the current artificial CA_C=0,AC_A=0.
Occlude the right eye (no disparity -> no fusional vergence). Check:
 (1) accommodation response clean (latency/TC/gain/peak_vel),
 (2) the viewing (left) eye stays ~on the midline target (AC/A shouldn't drag it off)."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)
ACC_CLEAN = with_brain(CLEAN, AC_A=0.0, CA_C=0.0)

def acc_metrics(a, t, T0):
    m=(t>=T0)&(t<3.0)
    s=float(np.mean(a[(t>T0-0.2)&(t<T0)])); ss=float(np.mean(a[(t>2.5)&(t<3.0)])); ch=ss-s
    over=(float(np.max(a[m]))-ss)/ch*100 if abs(ch)>0.1 else 0.0
    pv=float(np.max(np.abs(np.gradient(a,DT)[m])))
    thr=s+0.1*ch; cr=np.where((t>=T0)&(a>=thr))[0]; lat=(t[cr[0]]-T0)*1000 if len(cr) else np.nan
    return over, ss, pv, lat

def run(params, monocular):
    T0,TOT=1.0,4.0; t=np.arange(0,TOT,DT); N=len(t)
    pt=np.tile([0,0,6.0],(N,1)).astype(np.float32); pt[(t>=T0)&(t<3.0)]=[0,0,0.4]
    kw=dict(target=km.build_target(t,lin_pos=pt), scene_present_array=np.ones(N),
            target_present_array=np.ones(N), return_states=True)
    if monocular:
        # occlude RIGHT eye: no scene/target to R -> no binocular disparity
        kw['scene_present_R_array']  = np.zeros(N)
        kw['target_present_R_array'] = np.zeros(N)
    st=simulate(params,t,**kw)
    a=np.array(st.acc_plant[:,0]); eL=np.array(st.plant.left[:,0]); eR=np.array(st.plant.right[:,0])
    return t,a,eL,eR,T0

print("Accommodation step (far 6m->near 0.4m), midline target:\n")
print(f"  {'config':38s} | {'over':>5} {'ss':>5} {'pv':>5} {'lat':>5} | {'Leye drift':>10} {'verg':>6}")
for tag, p, mono in [
    ("isolated binoc (AC_A=0,CA_C=0) [current]", ACC_CLEAN, False),
    ("MONOCULAR (full model, R occluded)",       CLEAN,     True),
    ("binocular full model (for reference)",     CLEAN,     False),
]:
    t,a,eL,eR,T0 = run(p, mono)
    over,ss,pv,lat = acc_metrics(a,t,T0)
    m=(t>=T0)&(t<3.0)
    Ldrift = float(np.max(np.abs(eL[m])))   # viewing eye should stay near 0 (midline)
    verg = float(np.mean((eL-eR)[(t>2.5)&(t<3.0)]))
    print(f"  {tag:38s} | {over:4.1f}% {ss:5.2f} {pv:5.2f} {lat:4.0f}ms | {Ldrift:9.2f}° {verg:5.1f}°")
