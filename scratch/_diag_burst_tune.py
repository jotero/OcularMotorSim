"""Tune the saccade burst to trim the overshoot WITHOUT breaking the stop.
Sweep alpha_fac (BN facilitation). Gauge: command/eye overshoot, settle accuracy,
peak velocity, and #saccades (must stay 1-2)."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT, _saccade_onset_times
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)

for af in [1.0, 0.7, 0.5, 0.3, 0.0]:
    P = with_brain(THETA_NOISELESS, alpha_fac=af)
    print(f'\n=== alpha_fac = {af} ===')
    for amp in [10.0, 40.0]:
        pt3 = _pt3(t, amp, t_jump=t_jump)
        st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
        eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
        ni = np.array(ni_net(st))[:, 0]
        ub = np.array(extract_burst(st, P))[:, 0]
        win = (t >= t_jump) & (t <= 0.6)
        hold = (t >= 0.6) & (t <= 0.85)
        ncount = len(_saccade_onset_times(ub, t_jump))
        peakv = float(np.max(np.abs(np.gradient(eye, DT))))
        print(f'  A={amp:>2.0f}: NI_overshoot={ni[win].max()-amp:+.3f}  eye_overshoot={eye[win].max()-amp:+.3f}  '
              f'settle={eye[hold].mean():6.3f}  peakvel={peakv:6.1f}  #sac={ncount}')
