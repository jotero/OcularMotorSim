"""Can the existing _strengthen knobs (saccadic_suppression_threshold/steepness)
close the half-open suppression gate for small saccades and cut the post-sac
drift? The 2deg gate sits at raw ~0.85 (1-sat_d); a high threshold + steepness
pushes that toward 0 (more suppression). Cost: throttles pursuit between catch-up
saccades. So measure BOTH the 2deg NI drift AND the steady-state pursuit gain."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import (
    _run, _pt3, extract_z_opn, THETA_NOISELESS, DT, read_brain_acts)
from oculomotor.benchmarks import bench_pursuit as bp
from oculomotor.analysis import ni_net, extract_spv_states
from oculomotor.sim.simulator import with_brain

t = np.arange(0.0, 0.9, DT)


def sac_drift(P, amp=2.0):
    pt3 = _pt3(t, amp, t_jump=0.1)
    st = _run(t, pt3, key=0, max_s=int(0.9 / DT) + 200, params=P)
    acts = read_brain_acts(st, P)
    ni = np.array(ni_net(st))[:, 0]; z = extract_z_opn(st)
    win = (t >= 0.15) & (t <= 0.55) & (z >= 50.0)
    sat = np.array(acts.cb.saccadic_suppression_scene)
    smm = sat * np.array(acts.pc.scene_angular_vel)[:, 0] + np.array(acts.cb.fl_okr_drive)[:, 0]
    pk = lambda x: float(np.max(np.abs(x[win]))) if win.any() else float('nan')
    return pk(np.gradient(ni, DT)), pk(smm), float(np.min(sat[win]))


def pursuit_gain(P, vel=10.0):
    tp = np.arange(0.0, 2.5, DT)
    tgt, pt3, vt3 = bp._ramp(tp, vel, 0.3)
    st = bp._run(P, tp, jnp.array(pt3), jnp.array(vt3), key=0)
    spv = extract_spv_states(st, tp)[:, 0]
    ss = (tp >= 1.5) & (tp <= 2.4)
    return float(np.mean(spv[ss]) / vel)


print(f'{"thr":>4} {"steep":>6} {"2deg NI_drift":>13} {"scene_mm":>9} {"gate_min":>9} {"pursuit_gain":>13}')
for thr, k in [(0.0, 1.0), (0.5, 1.0), (0.7, 2.0), (0.85, 3.0)]:
    P = with_brain(THETA_NOISELESS, saccadic_suppression_threshold=thr,
                   saccadic_suppression_steepness=k)
    d, smm, gmin = sac_drift(P)
    pg = pursuit_gain(P)
    print(f'{thr:4.2f} {k:6.1f} {d:13.2f} {smm:9.2f} {gmin:9.2f} {pg:13.3f}')
