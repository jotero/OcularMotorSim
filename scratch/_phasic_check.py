"""Phasic cross-links: dark state should = (tonic_verg, tonic_acc) for ANY AC/A,CA/C.
Verify dark state, target-driven vergence, AC/A ratio, accommodation all OK."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km
DT=0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0,sigma_slip=0,sigma_pos=0,sigma_vel=0), sigma_acc=0)
print('tonic_verg=%.2f tonic_acc=%.2f AC_A=%.1f CA_C=%.3f'%(
    CLEAN.brain.tonic_verg, CLEAN.brain.tonic_acc, CLEAN.brain.AC_A, CLEAN.brain.CA_C))

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

print("\nDARK resting state (should equal tonic_verg=3.67, tonic_acc=1.0):")
for tag,p in [('default (AC_A=5,CA_C=.08)', CLEAN),
              ('AC_A=8', with_brain(CLEAN, AC_A=8.0)),
              ('CA_C=0.16', with_brain(CLEAN, CA_C=0.16)),
              ('AC_A=3,CA_C=0.05', with_brain(CLEAN, AC_A=3.0, CA_C=0.05))]:
    dv,df=dark_rest(p); print(f"  {tag:26s}: dark verg {dv:5.2f}deg  dark focus {df:5.2f}D")

print("\nTarget-driven vergence (geometric near 9.1, far 1.2):")
print(f"  near 0.4m = {tverg(CLEAN,0.4):.2f}deg   far 3m = {tverg(CLEAN,3.0):.2f}deg")

# AC/A ratio: drive accommodation with a lens (monocular so disparity doesn't drive vergence)
def aca_ratio(params):
    T=6.0; t=np.arange(0,T,DT); N=len(t)
    pt=np.tile([0,0,2.0],(N,1)).astype(np.float32)   # fixed far-ish target 2m
    lens=np.where(t>2.0, -2.0, 0.0).astype(np.float32) # -2D lens forces +2D accommodation
    st=simulate(params,t,target=km.build_target(t,lin_pos=pt),scene_present_array=np.ones(N),
                target_present_array=np.ones(N),
                scene_present_R_array=np.zeros(N), target_present_R_array=np.zeros(N),  # monocular
                lens_L_array=lens, return_states=True)
    v=np.array(st.plant.left[:,0]-st.plant.right[:,0]); a=np.array(st.acc_plant[:,0])
    dv=float(np.mean(v[t>T-0.5])-np.mean(v[(t>1.5)&(t<2.0)]))
    da=float(np.mean(a[t>T-0.5])-np.mean(a[(t>1.5)&(t<2.0)]))
    return dv, da, (dv/da if abs(da)>0.1 else float('nan'))
dv,da,r = aca_ratio(CLEAN)
print(f"\nAC/A (lens -2D): dverg={dv:.2f}deg  dacc={da:.2f}D  ratio={r:.2f} deg/D (= {r/0.573:.1f} pd/D)")
