"""Where is the saccade overshoot? Compare burst (u_burst), command (NI_net) and
eye for a 40deg saccade. If NI_net overshoots 40 -> burst generator/NI; if
eye overshoots NI_net -> plant. If u_burst goes negative -> braking (IBN) burst."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net, extract_burst

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
pt3 = _pt3(t, 40.0, t_jump=t_jump)
st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
ni = np.array(ni_net(st))[:, 0]
ub = np.array(extract_burst(st, THETA_NOISELESS))[:, 0]
win = (t >= t_jump) & (t <= 0.6)

print('target           = 40.00 deg')
print('peak NI_net pos  = {:.3f}  -> command overshoot = {:+.3f} deg'.format(ni[win].max(), ni[win].max() - 40))
print('peak eye pos     = {:.3f}  -> eye overshoot      = {:+.3f} deg'.format(eye[win].max(), eye[win].max() - 40))
print('max |eye - NI|   = {:.3f} deg  (plant tracking error)'.format(np.max(np.abs((eye - ni)[win]))))
print('u_burst:  max = {:.1f}   min = {:.1f}   (min<0 => braking/IBN burst)'.format(ub[win].max(), ub[win].min()))
# settle: does eye/NI come back down after overshoot?
hold = (t >= 0.5) & (t <= 0.85)
print('settled NI_net   = {:.3f}   settled eye = {:.3f}'.format(ni[hold].mean(), eye[hold].mean()))
