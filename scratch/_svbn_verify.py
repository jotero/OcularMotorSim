"""Verify unity-gain SVBN: midline (no saccade) unaffected + asym conv/div sane."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001; IPD=0.064
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)
def depth(v): return IPD/2.0/np.tan(np.radians(v)/2.0)

def run(amp_end, amp_start, asym, ver_sign=1):
    T0,TOT=0.5,4.0; t=np.arange(0,TOT,DT); N=len(t)
    d0,d1=depth(amp_start),depth(amp_end)
    ver = ver_sign*10.0 if asym else 0.0
    x1=np.tan(np.radians(ver))*d1
    p0=np.array([0.,0.,d0]); p1=np.array([x1,0.,d1])
    pt=np.where((t>=T0)[:,None],p1,p0).astype(np.float32)
    st=simulate(CLEAN,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),return_states=True)
    v=np.array(st.plant.left[:,0]-st.plant.right[:,0]); idx=(t>=T0)&(t<=T0+1.5)
    pv=float(np.max(np.abs(np.gradient(v,DT)[idx])))
    ss=float(np.mean(v[t>TOT-0.3])); s0=float(v[int(T0/DT)-1]); ch=ss-s0
    pk = float(np.max(v[idx])) if ch>0 else float(np.min(v[idx]))
    over=(pk-ss)/ch*100 if abs(ch)>0.5 else 0.0
    return pv, over, ss

print("                       peak_vel  overshoot   ss")
pv,ov,ss=run(12,2,False);            print(f"  midline CONV (no sac): {pv:6.1f}    {ov:6.1f}%   {ss:5.2f}")
pv,ov,ss=run(2,12,False);            print(f"  midline DIV  (no sac): {pv:6.1f}    {ov:6.1f}%   {ss:5.2f}")
pv,ov,ss=run(12,2,True,+1);          print(f"  asym CONV (+10 sac):   {pv:6.1f}    {ov:6.1f}%   {ss:5.2f}")
pv,ov,ss=run(2,12,True,-1);          print(f"  asym DIV  (-10 sac):   {pv:6.1f}    {ov:6.1f}%   {ss:5.2f}")
