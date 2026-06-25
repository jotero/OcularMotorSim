"""alpha_fac tradeoff table: overshoot vs peak velocity vs main-sequence shape vs
saccade count. Lets the user pick the burst facilitation that trims post-saccadic
overshoot WITHOUT slowing/bending the main sequence out of band.

Columns:
  overshoot_40 : ni_net peak - 40 deg  (command overshoot of a 40 deg saccade)
  #sac_40      : saccades generated for the 40 deg step (want 1)
  peak_v_20    : peak velocity of a 20 deg saccade   (band [550, 750])
  resid_max    : max |peak - 700(1-e^-A/7)|/ref over A>=5 deg (band <= 0.20)
"""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, _pt3, THETA_NOISELESS, DT, _saccade_onset_times
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
i_j = int(t_jump / DT)
AMPS = [5.0, 7.5, 10.0, 15.0, 20.0]


def peak_vel(P, amp):
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    vel = np.gradient(eye, DT)
    return float(np.max(np.abs(vel[i_j:i_j + 400])))


print(f'{"alpha_fac":>9} {"overshoot_40":>13} {"#sac_40":>8} {"peak_v_20":>10} {"resid_max":>10}')
for af in (1.0, 0.85, 0.7, 0.6, 0.5):
    P = with_brain(THETA_NOISELESS, alpha_fac=af)
    # 40 deg overshoot + count
    pt3 = _pt3(t, 40.0, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=P)
    ni = np.array(ni_net(st))[:, 0]
    ub = np.array(extract_burst(st, P))[:, 0]
    over = float(ni[(t >= t_jump) & (t <= 0.6)].max() - 40.0)
    nc = len(_saccade_onset_times(ub, t_jump))
    # main sequence
    pv = {a: peak_vel(P, a) for a in AMPS}
    ref = {a: 700.0 * (1.0 - np.exp(-a / 7.0)) for a in AMPS}
    resid = max(abs(pv[a] - ref[a]) / ref[a] for a in AMPS)
    worst = max(AMPS, key=lambda a: abs(pv[a] - ref[a]) / ref[a])
    print(f'{af:9.2f} {over:+13.3f} {nc:8d} {pv[20.0]:10.1f} {resid:10.3f}'
          f'   (worst@{worst:.0f}deg: {pv[worst]:.0f} vs {ref[worst]:.0f})')
