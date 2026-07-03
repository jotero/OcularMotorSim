"""dt sweep — speed + saccade main-sequence metrics at dt = 0.001 / 0.002 / 0.005.

For each solver dt:
  * time the compile (first call) and the warm run (subsequent calls)
  * extract saccade amplitude / peak velocity / duration for a few amplitudes
  * flag instability (NaN, or absurd velocity/amplitude)

`tau_fast` (NI lead filter) tracks config.DT_SOLVE, which we set per dt so that
filter stays Heun-stable. The saccade-generator fast states (tau_sac ~1 ms,
tau_bn ~3 ms) are NOT scaled — so a large dt may still destabilise the burst.
That is the point of the sweep: find where it breaks.

Output sampling is fixed at 1 ms regardless of solver dt (diffrax interpolates
the coarse-step solution onto the 1 ms grid), so the metrics are comparable.
"""
import time
import numpy as np
import jax, jax.numpy as jnp

from oculomotor import config as _config
from oculomotor.sim.simulator import SimConfig, simulate
from oculomotor.sim import kinematics as km
from oculomotor.benchmarks.bench_saccades import THETA_NOISELESS, _pt3, _primary_saccade
from oculomotor.analysis import extract_burst

P             = THETA_NOISELESS
DTS           = [0.001, 0.002, 0.005]
AMPS          = [5.0, 10.0, 20.0]
T_end, t_jump = 0.8, 0.1
WARMUP        = 3.0
t_np          = np.arange(0.0, T_end, 0.001)     # OUTPUT grid fixed at 1 ms
Tn            = len(t_np)


def run_one(amp, cfg):
    pt3    = _pt3(t_np, amp, t_jump=t_jump)
    target = km.build_target(t_np, lin_pos=np.array(pt3), lin_vel=np.zeros((Tn, 3)))
    st = simulate(P, jnp.array(t_np), target=target,
                  scene_present_array=jnp.ones(Tn),
                  sim_config=cfg, return_states=True, key=jax.random.PRNGKey(0))
    return jax.block_until_ready(st)


results = {}
for dt in DTS:
    jax.clear_caches()                 # force a fresh compile that re-reads config.DT_SOLVE
    _config.DT_SOLVE = dt              # -> tau_fast tracks dt
    cfg    = SimConfig(dt_solve=dt, warmup_s=WARMUP)
    nsteps = int(round((WARMUP + T_end) / dt))

    t0 = time.perf_counter(); run_one(AMPS[1], cfg); t_compile = time.perf_counter() - t0

    per_amp, warm_times = [], []
    for amp in AMPS:
        t0 = time.perf_counter(); st = run_one(amp, cfg); warm_times.append(time.perf_counter() - t0)
        eye   = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
        burst = np.array(extract_burst(st, P))[:, 0]
        if not np.all(np.isfinite(eye)):
            per_amp.append((amp, np.nan, np.nan, np.nan, 'NaN')); continue
        a_out, v_peak, dur = _primary_saccade(burst, eye, t_np, t_jump)
        status = 'ok' if (abs(v_peak) < 2000 and abs(a_out) < 60) else 'UNSTABLE'
        per_amp.append((amp, a_out, v_peak, dur, status))

    results[dt] = dict(nsteps=nsteps, t_compile=t_compile,
                       warm=float(np.mean(warm_times)), per_amp=per_amp)

_config.DT_SOLVE = 0.001               # restore default

# ── Report ──────────────────────────────────────────────────────────────────────
base = results[DTS[0]]['warm']
print('\n================= dt sweep: speed =================')
print(f'{"dt":>7} {"steps":>7} {"compile_s":>10} {"warm_run_s":>11} {"speedup":>8}')
for dt in DTS:
    r = results[dt]
    print(f'{dt:>7.3f} {r["nsteps"]:>7d} {r["t_compile"]:>10.2f} {r["warm"]:>11.3f} {base/r["warm"]:>7.2f}x')

print('\n=========== saccade main-sequence metrics ===========')
for dt in DTS:
    print(f'\n  dt = {dt:.3f}')
    print(f'    {"cmd":>5} {"amp":>8} {"vpeak":>9} {"dur_ms":>8}  status')
    for (amp, a_out, v_peak, dur, status) in results[dt]['per_amp']:
        dur_ms = dur * 1000 if np.isfinite(dur) else np.nan
        print(f'    {amp:>5.0f} {a_out:>8.2f} {v_peak:>9.1f} {dur_ms:>8.1f}  {status}')

print('\n  reference peak velocity  700*(1-e^{-A/7}):')
for amp in AMPS:
    print(f'    {amp:>5.0f} -> {700 * (1 - np.exp(-amp / 7)):.1f} deg/s')
