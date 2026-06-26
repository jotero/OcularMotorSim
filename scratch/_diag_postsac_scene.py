"""Why is the SCENE-side EC residual (scene_mm) 5-6x worse for small saccades?
Split it into the suppressed sensed slip vs the EC prediction, and read the
saccadic-suppression gate, for a small (2 deg) vs large (20 deg) saccade.

scene_mm = sat * scene_angular_vel + fl_okr_drive
  sat               = saccadic-suppression gate (1 = no suppression, 0 = full)
  scene_angular_vel = raw reafferent scene slip on the retina
  fl_okr_drive      = cerebellar OKR EC prediction (should cancel the slip)
If the gate (sat) stays near 1 for small saccades -> suppression too weak/short
-> the reafferent slip leaks. If fl_okr_drive mispredicts -> EC cascade issue."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT, read_brain_acts)
from oculomotor.analysis import ni_net

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)

for amp in (2.0, 20.0):
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
    acts = read_brain_acts(st, THETA_NOISELESS)
    z = extract_z_opn(st)
    sat   = np.array(acts.cb.saccadic_suppression_scene)
    sslip = np.array(acts.pc.scene_angular_vel)[:, 0]          # raw reafferent scene slip
    ecp   = np.array(acts.cb.fl_okr_drive)[:, 0]               # OKR EC prediction
    smm   = sat * sslip + ecp                                  # scene mismatch
    # during-saccade window (z<50) and post-saccade slow window (z>=50)
    sac   = (t >= t_jump) & (z < 50.0)
    post  = (t >= t_jump + 0.05) & (t <= 0.55) & (z >= 50.0)
    pk = lambda x, w: float(np.max(np.abs(x[w]))) if w.any() else float('nan')
    print(f'\n=== {amp:.0f} deg ===')
    print(f'  during saccade : sat_min={float(np.min(sat[sac])):.3f}  '
          f'raw_slip={pk(sslip,sac):6.1f}  suppressed={pk(sat*sslip,sac):6.1f}  ec_pred={pk(ecp,sac):6.1f}')
    print(f'  post saccade   : raw_slip={pk(sslip,post):6.2f}  suppressed={pk(sat*sslip,post):6.2f}  '
          f'ec_pred={pk(ecp,post):6.2f}  -> scene_mm={pk(smm,post):.2f}  (sat={float(np.min(sat[post])):.2f})')
