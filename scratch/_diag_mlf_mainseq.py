"""Verify the saccade main sequence + clean stop at mlf_lead=0.5 vs 0 before
adopting. Expect peak velocity UP (toward the 700-curve, we were ~5% under),
resid DOWN, #saccades still 1, overshoot small."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, THETA_NOISELESS, DT, _saccade_onset_times)
from oculomotor.analysis import ni_net, extract_burst
from oculomotor.sim.simulator import with_brain

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT); i_j = int(t_jump / DT)
AMPS = [5.0, 7.5, 10.0, 15.0, 20.0]


def peak_vel(P, amp):
    st = _run(t, _pt3(t, amp, t_jump=t_jump), key=0, max_s=int(T_end / DT) + 200, params=P)
    eye = ((np.array(st.plant.left) + np.array(st.plant.right)) / 2.0)[:, 0]
    return float(np.max(np.abs(np.gradient(eye, DT)[i_j:i_j + 400])))


print(f'{"mlf":>5} {"pk@20":>7} {"resid":>7} {"#sac@40":>8} {"over@40":>8}')
for mlf in (0.0, 0.5):
    P = with_brain(THETA_NOISELESS, mlf_lead=mlf)
    pv = {a: peak_vel(P, a) for a in AMPS}
    ref = {a: 700.0 * (1.0 - np.exp(-a / 7.0)) for a in AMPS}
    resid = max(abs(pv[a] - ref[a]) / ref[a] for a in AMPS)
    st = _run(t, _pt3(t, 40.0, t_jump=t_jump), key=0, max_s=int(T_end / DT) + 200, params=P)
    ni = np.array(ni_net(st))[:, 0]; ub = np.array(extract_burst(st, P))[:, 0]
    over = float(ni[(t >= t_jump) & (t <= 0.6)].max() - 40.0)
    nc = len(_saccade_onset_times(ub, t_jump))
    print(f'{mlf:5.2f} {pv[20.0]:7.1f} {resid:7.3f} {nc:8d} {over:+8.3f}')
