"""What drives the NI after a small saccade? Decompose the post-burst NI velocity
into its candidate drives: burst tail vs pursuit (EC leak) vs VS (EC leak)."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT,
    read_brain_acts, read_brain_decoded, _omega_tvor_traj)
from oculomotor.analysis import ni_net, vs_net, extract_burst

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)

for amp in [5.0, 40.0]:
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
    acts = read_brain_acts(st, THETA_NOISELESS)
    decoded = read_brain_decoded(st, THETA_NOISELESS)

    ub = np.array(extract_burst(st, THETA_NOISELESS))[:, 0]              # u_burst yaw
    xp = np.array(decoded.pu.net)[:, 0]                                   # pursuit net yaw
    vs = np.array(vs_net(st))[:, 0]                                       # VS net yaw
    ot = np.array(_omega_tvor_traj(st, THETA_NOISELESS.brain))[:, 0]      # omega_tvor yaw
    ni = np.array(ni_net(st))[:, 0]
    ni_v = np.gradient(ni, DT)
    slip = np.array(acts.pc.scene_angular_vel)[:, 0]                     # delayed scene slip (reafference)
    tvel = np.array(acts.pc.target_vel)[:, 0]                            # delayed target vel

    z = extract_z_opn(st)
    win = (t >= t_jump + 0.05) & (t <= 0.45) & (z >= 50.0)
    pk = lambda x: float(np.max(np.abs(x[win]))) if win.any() else float('nan')

    print(f'\n=== {amp:.0f} deg, post-burst slow window peaks (deg/s) ===')
    print(f'  NI net vel       = {pk(ni_v):6.3f}   <- what we are explaining')
    print(f'  u_burst          = {pk(ub):6.3f}   <- burst terminated?')
    print(f'  pursuit net      = {pk(xp):6.3f}   <- pursuit engaged (EC leak)?')
    print(f'  VS net (w_est)   = {pk(vs):6.3f}   <- VS engaged (EC leak)?')
    print(f'  omega_tvor       = {pk(ot):6.3f}')
    print(f'  slip_del (scene) = {pk(slip):6.3f}   <- residual reafferent slip')
    print(f'  tvel_del (targ)  = {pk(tvel):6.3f}')
