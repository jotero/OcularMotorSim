"""SVBN saccade-vergence burst tuning for the faster smooth loop (K_phasic_verg=12).
Measure peak vergence velocity: symmetric (no saccade, SVBN off) vs asymmetric
(concurrent 10 deg version saccade, SVBN fires). Zee 1992: asym ~2-3x sym; conv
peaks ~41-58 deg/s @ 10deg. Also watch vergence OVERSHOOT (burst over-driving)."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001; IPD=0.064
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)

def depth(verg_deg): return IPD/2.0/np.tan(np.radians(verg_deg)/2.0)

def run(params, amp_deg, asym):
    T0,TOT=0.5,4.0; t=np.arange(0,TOT,DT); N=len(t)
    d0,d1 = depth(2.0), depth(amp_deg)
    ver = 10.0 if asym else 0.0
    x1 = np.tan(np.radians(ver))*d1
    p0=np.array([0.,0.,d0]); p1=np.array([x1,0.,d1])
    pt=np.where((t>=T0)[:,None], p1, p0).astype(np.float32)
    st=simulate(params,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),return_states=True)
    eL=np.array(st.plant.left[:,0]); eR=np.array(st.plant.right[:,0])
    verg=eL-eR; vvel=np.gradient(verg,DT)
    idx=(t>=T0)&(t<=T0+1.5)
    pv=float(np.max(np.abs(vvel[idx])))
    ss=float(np.mean(verg[t>TOT-0.3])); s0=float(verg[int(T0/DT)-1])
    over=(float(np.max(verg[idx]))-ss)/(ss-s0+1e-9)*100
    return pv, over, ss

print("10 deg convergence step, K_phasic_verg=12 (current loop):")
print(f"  {'g_svbn':>6} | {'sym pv':>6} {'sym over':>8} | {'asym pv':>7} {'asym over':>9} | {'facil':>5}")
for g in (0, 10, 20, 30, 40):
    p = with_brain(CLEAN, g_svbn_conv=float(g))
    spv, sov, sss = run(p, 12.0, asym=False)   # amp 12 => ~10deg change from 2deg
    apv, aov, ass = run(p, 12.0, asym=True)
    facil = apv/spv if spv>1e-6 else float('nan')
    print(f"  {g:6.0f} | {spv:6.1f} {sov:7.1f}% | {apv:7.1f} {aov:8.1f}% | {facil:5.2f}")
print("\nZee target: sym slower, asym ~2-3x sym, conv peaks ~41-58 deg/s; minimal overshoot")
