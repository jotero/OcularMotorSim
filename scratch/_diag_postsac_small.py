"""Re-ground the SMALL-saccade post-saccadic drift at the current params
(mn_ff_yaw=1.0, alpha_fac=0.5), focusing on the PURSUIT contamination.

For a 5 deg saccade to a stationary lit target, measure in the post-burst window:
  - version eye velocity (the ring)
  - NI_net velocity (is the command still moving?)
  - u_pursuit net (the pursuit command — should be ~0 for a stationary target;
    any residual is the EC mismatch leaking into pursuit via the phasic path)
Closed loop (current) vs OPEN loop (pursuit gains zeroed). If the pursuit
contamination is large closed and ~0 open, the small-saccade drift is the EC/
visual-loop mismatch feeding pursuit."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)


def measure(P, label, amp=5.0):
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
    eye   = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    evel  = np.gradient(eye, DT)
    ni    = np.array(ni_net(st))[:, 0]
    nivel = np.gradient(ni, DT)
    pun   = np.array(st.brain.pu.R - st.brain.pu.L)[:, 0]    # pursuit net command
    ub    = np.array(extract_burst(st, P))[:, 0]
    fast  = np.abs(ub) > 20.0
    if not fast.any():
        print(f'{label}: no burst'); return
    be = int(np.where(fast)[0][-1])                          # burst end index
    s0, s1 = be + 10, be + 60                                # short: +10..+60 ms
    l0, l1 = be + 60, be + 210                               # long:  +60..+210 ms
    pk = lambda x, a, b: float(np.max(np.abs(x[a:b])))
    print(f'{label}  (burst ends {be*DT:.3f}s, landed {eye[be]:.2f} deg)')
    print(f'   eye_vel ring : short={pk(evel,s0,s1):5.2f}  long={pk(evel,l0,l1):5.2f} deg/s')
    print(f'   NI_net vel   : short={pk(nivel,s0,s1):5.2f}  long={pk(nivel,l0,l1):5.2f} deg/s')
    print(f'   u_pursuit    : short={pk(pun,s0,s1):5.2f}  long={pk(pun,l0,l1):5.2f} deg/s  <-- pursuit contamination')


measure(THETA_NOISELESS, '5deg CLOSED (current)')
P_open = with_brain(THETA_NOISELESS, K_pursuit=0.0, K_phasic_pursuit=0.0,
                    K_pursuit_direct=0.0, K_cereb_pu=0.0)
measure(P_open, '5deg OPEN  (pursuit loop zeroed)')
print()
measure(THETA_NOISELESS, '2deg CLOSED (current)', amp=2.0)
