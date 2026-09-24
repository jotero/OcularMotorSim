"""Cross-machine performance baseline. Run the same file on each machine and diff.

Compile and solve are timed separately on purpose: JAX traces this 400-state ODE
once per shape, and that one-off cost dominates a short run. Only the solve time
tells you how fast a real sweep will be.

    python -X utf8 scratch/perf_baseline.py
"""
import platform, sys, time
import numpy as np, jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, SimConfig
import oculomotor

DT, TEND, REPEATS = 0.001, 2.0, 6
t = np.arange(0.0, TEND, DT); T = len(t)
kw = dict(scene_present_array=np.ones(T, np.float32),
          target_present_array=np.ones(T, np.float32),
          max_steps=int(T * 1.1) + 1000, sim_config=SimConfig(warmup_s=0.0),
          return_states=True, key=jax.random.PRNGKey(0))

print(f'machine   : {platform.machine()}  {platform.system()} {platform.release()}')
print(f'processor : {platform.processor() or "n/a"}')
print(f'python    : {sys.version.split()[0]}   jax {jax.__version__}')
print(f'devices   : {jax.devices()}')
print(f'oculomotor: {oculomotor.__version__}')
print(f'workload  : {TEND:g} s @ dt={DT*1e3:g} ms = {T} steps, Heun fixed-step\n')

t0 = time.perf_counter()
jax.block_until_ready(simulate(PARAMS_DEFAULT, t, **kw))
compile_s = time.perf_counter() - t0

runs = []
for _ in range(REPEATS):
    t0 = time.perf_counter()
    jax.block_until_ready(simulate(PARAMS_DEFAULT, t, **kw))
    runs.append(time.perf_counter() - t0)

solve = min(runs)
print(f'compile+first run : {compile_s:8.2f} s')
print(f'solve (best of {REPEATS}) : {solve:8.3f} s   ({T/solve:,.0f} ODE steps/s)')
print(f'solve runs        : {", ".join(f"{r:.3f}" for r in runs)} s')
