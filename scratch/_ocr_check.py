"""Does the dark-tonic change affect OCR torsion? Replicate the OCR sim (roll tilt,
no target = dark) for NEW tonics vs OLD (tonic_verg=3.67, tonic_acc=1.0)."""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, SimConfig
from oculomotor.sim import kinematics as km
DT=0.001
def _pad3(a, ax='roll'):
    out=np.zeros((len(a),3),np.float32); out[:, {'yaw':0,'pitch':1,'roll':2}[ax]]=a; return out
G_OCR = PARAMS_DEFAULT.brain.g_ocr; G0=9.81
print('g_ocr=%.3f, new tonics: tv=%.3f ta=%.3f'%(G_OCR, PARAMS_DEFAULT.brain.tonic_verg, PARAMS_DEFAULT.brain.tonic_acc))

def ocr_ss(params, tilt_deg):
    TILT_VEL=60.0; HOLD_T=10.0; tilt_dur=tilt_deg/TILT_VEL; total=tilt_dur+HOLD_T
    t=np.arange(0,total,DT); hv=np.where(t<tilt_dur, TILT_VEL, 0.0)
    head=km.build_kinematics(t, rot_vel=_pad3(hv,'roll'))
    st=simulate(params, t, head=head, target_present_array=np.zeros(len(t)),
                sim_config=SimConfig(warmup_s=0.0), return_states=True)
    er=(np.array(st.plant.left[:,2])+np.array(st.plant.right[:,2]))/2.0
    # also vergence to confirm dark state
    vg=float(np.mean((st.plant.left[:,0]-st.plant.right[:,0])[-500:]))
    return float(er[-1]), vg

OLD = with_brain(PARAMS_DEFAULT, tonic_verg=3.67, tonic_acc=1.0)
print(f"\n  {'tilt':>5} | {'expected':>9} | {'NEW torsion':>11} {'(verg)':>7} | {'OLD torsion':>11} {'(verg)':>7}")
for tilt in (15,30,45,90):
    exp = -G_OCR*G0*np.sin(np.radians(tilt))
    tn,vn = ocr_ss(PARAMS_DEFAULT, tilt)
    to,vo = ocr_ss(OLD, tilt)
    print(f"  {tilt:5d} | {exp:9.2f} | {tn:11.2f} {vn:7.1f} | {to:11.2f} {vo:7.1f}")
