"""Measure compile vs warm-run, and test JAX's persistent (on-disk) compilation
cache across fresh processes.

Set OM_CACHE_DIR to enable the persistent cache (must be BEFORE jax compiles):
    OM_CACHE_DIR unset      -> every fresh process recompiles from scratch
    OM_CACHE_DIR=<dir> set  -> first process compiles + writes; later processes
                               with the SAME dir load the executable from disk.

Prints:
    first  simulate  = compile (+ run)   [what a cold process pays]
    second simulate  = warm run          [steady-state throughput]
"""
import os, time
import jax

cache_dir = os.environ.get('OM_CACHE_DIR')
if cache_dir:
    jax.config.update('jax_compilation_cache_dir', cache_dir)
    jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0)
    jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
    print(f'[cache] persistent compilation cache ENABLED  -> {cache_dir}')
else:
    print('[cache] persistent compilation cache DISABLED')

import numpy as np
import jax.numpy as jnp
from oculomotor.sim.simulator import simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.benchmarks.bench_saccades import THETA_NOISELESS, _pt3

P             = THETA_NOISELESS
T_end, t_jump = 0.8, 0.1
t_np          = np.arange(0.0, T_end, 0.001)
Tn            = len(t_np)
SIM_SPAN      = 3.0 + T_end                       # warmup + window (integrated seconds)
target = km.build_target(t_np, lin_pos=np.array(_pt3(t_np, 20.0, t_jump=t_jump)),
                         lin_vel=np.zeros((Tn, 3)))
cfg = SimConfig(dt_solve=0.001, warmup_s=3.0)


def one():
    st = simulate(P, jnp.array(t_np), target=target, scene_present_array=jnp.ones(Tn),
                  sim_config=cfg, return_states=True, key=jax.random.PRNGKey(0))
    return jax.block_until_ready(st)


t0 = time.perf_counter(); one(); t_first  = time.perf_counter() - t0
t0 = time.perf_counter(); one(); t_second = time.perf_counter() - t0

print(f'first  simulate (compile+run): {t_first:7.2f} s')
print(f'second simulate (warm run)   : {t_second:7.2f} s   '
      f'({t_second / SIM_SPAN:.3f} s wall / simulated s)')
