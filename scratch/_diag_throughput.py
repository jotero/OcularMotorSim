"""Warm-run throughput: seconds of wall time per second of simulated time.

Runs warm (post-compile) sims at several integrated durations and fits
    warm_wall = fixed_overhead + marginal * integrated_seconds
so we can separate the per-call overhead (interpolant build, noise scan, output
slice) from the true per-step integration cost. warmup_s=0 so integrated time =
window exactly (we only care about timing, not settling).
"""
import time
import numpy as np
import jax, jax.numpy as jnp

from oculomotor.sim.simulator import simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.benchmarks.bench_saccades import THETA_NOISELESS, _pt3

P     = THETA_NOISELESS
DT    = 0.001
TENDS = [1.0, 4.0, 8.0]
rows  = []

for T_end in TENDS:
    t_np   = np.arange(0.0, T_end, DT); Tn = len(t_np)
    target = km.build_target(t_np, lin_pos=np.array(_pt3(t_np, 20.0, t_jump=0.1)),
                             lin_vel=np.zeros((Tn, 3)))
    cfg = SimConfig(dt_solve=DT, warmup_s=0.0)

    def one():
        st = simulate(P, jnp.array(t_np), target=target, scene_present_array=jnp.ones(Tn),
                      sim_config=cfg, return_states=True, key=jax.random.PRNGKey(0))
        return jax.block_until_ready(st)

    one()                                    # compile
    ts = []
    for _ in range(3):
        t0 = time.perf_counter(); one(); ts.append(time.perf_counter() - t0)
    warm = min(ts)
    rows.append((T_end, Tn, warm))
    print(f'T_end={T_end:4.1f}s  steps={Tn:5d}  warm={warm:6.3f}s   '
          f'{warm / T_end:5.3f} s/s   {1e6 * warm / Tn:5.1f} us/step')

xs = np.array([r[0] for r in rows]); ys = np.array([r[2] for r in rows])
b, a = np.polyfit(xs, ys, 1)             # warm = a + b*integrated_seconds
print(f'\nfit:  warm = {a * 1000:5.1f} ms fixed  +  {b:.3f} s per simulated second')
print(f'marginal integration: {b:.3f} s wall / simulated s   '
      f'({1e6 * b * DT:.1f} us/step,  {1.0 / b:.2f}x real-time)')
