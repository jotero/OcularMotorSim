"""With the accommodation EC ON, is the residual full-model accommodation overshoot
the AC/A<->CA/C COUPLING re-injecting the ring? And is the vergence overshoot now
just AC/A reading a fast (non-ringing) accommodation transient?"""
import numpy as np
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

DT = 0.001
CLEAN = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0,
                                sigma_pos=0.0, sigma_vel=0.0), sigma_acc=0.0)

def run(params):
    T0, TOTAL = 1.0, 3.5
    t = np.arange(0.0, TOTAL, DT); T = len(t)
    pt = np.tile([0.0, 0.0, 3.0], (T, 1)).astype(np.float32)
    pt[t >= T0] = [0.0, 0.0, 0.4]
    st = simulate(params, t, target=km.build_target(t, lin_pos=pt),
                  scene_present_array=np.ones(T), target_present_array=np.ones(T),
                  return_states=True)
    v = np.array(st.plant.left[:, 0] - st.plant.right[:, 0])
    a = np.array(st.acc_plant[:, 0]); m = t >= T0
    af = np.array(st.brain.va.acc_fast); aslw = np.array(st.brain.va.acc_slow)
    def ov(x):
        ss = float(np.mean(x[t > TOTAL-0.3])); pk = float(np.max(x[m])); s0 = float(x[int(T0/DT)-1])
        return (pk-ss)/(ss-s0+1e-9)*100, ss
    vo, vss = ov(v); ao, ass = ov(a)
    # AC/A drive = AC_A * 0.5729 * (acc_fast+acc_slow); does the NEURAL signal AC/A reads overshoot?
    aca_sig = af + aslw; m2 = t >= T0
    aca_ss = float(np.mean(aca_sig[t>TOTAL-0.3])); aca_pk = float(np.max(aca_sig[m2]))
    aca_over = (aca_pk-aca_ss)/(aca_ss-float(aca_sig[int(T0/DT)-1])+1e-9)*100
    return vo, vss, ao, ass, aca_over

print(f"{'config (EC on)':34s} | {'acc over':>8} | {'AC/A-sig over':>13} | {'verg over':>9} {'vss':>5}")
for tag, p in [
    ("full (AC_A=5, CA_C=0.08)", CLEAN),
    ("CA_C=0  (no verg->acc)",   with_brain(CLEAN, CA_C=0.0)),
    ("AC_A=0  (no acc->verg)",   with_brain(CLEAN, AC_A=0.0)),
    ("AC_A=0, CA_C=0 (isolated)",with_brain(CLEAN, AC_A=0.0, CA_C=0.0)),
]:
    vo, vss, ao, ass, acao = run(p)
    print(f"  {tag:32s} | {ao:7.1f}% | {acao:12.1f}% | {vo:8.1f}% {vss:5.2f}")
