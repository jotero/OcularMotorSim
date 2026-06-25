"""Does the oblique fan accumulate saccade lag, and is alpha_fac=0.5 the cause?
For each outward direction, report the latency (saccade onset - target onset) and
whether the eye actually reached the commanded amplitude within the dwell."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

AMP, HOLD, T0 = 12.0, 0.35, 0.2
DIRS = np.arange(0, 360, 45)
targets = []
for d in DIRS:
    targets.append((AMP * np.cos(np.radians(d)), AMP * np.sin(np.radians(d))))
    targets.append((0.0, 0.0))
T_end = T0 + len(targets) * HOLD + 0.2
t = np.arange(0.0, T_end, DT); T = len(t)
tgt_h = np.zeros(T); tgt_v = np.zeros(T); out_events = []
for i, (h, v) in enumerate(targets):
    ts = T0 + i * HOLD
    tgt_h[t >= ts] = h; tgt_v[t >= ts] = v
    if i % 2 == 0:
        out_events.append((int(ts / DT), float(DIRS[i // 2])))
pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
pt3[:, 0] = np.tan(np.radians(tgt_h)); pt3[:, 1] = np.tan(np.radians(tgt_v))

for af in (1.0, 0.5):
    P = with_brain(THETA_NOISELESS, alpha_fac=af)
    st = _run(t, jnp.array(pt3), key=0, max_s=int(T_end / DT) + 500, params=P)
    eye = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
    eh, ev = eye[:, 0], eye[:, 1]
    speed = np.hypot(np.gradient(eh, DT), np.gradient(ev, DT))
    print(f'\n=== alpha_fac={af} ===')
    print(' dir   targ-onset   sac-latency   reached(deg, want 12)')
    for i0, d in out_events:
        win = slice(i0, i0 + int(2 * HOLD / DT))
        sp = speed[win]
        above = sp > 20.0
        lat = (np.argmax(above) * DT * 1000) if above.any() else float('nan')
        # furthest displacement from center reached anywhere in the 2*HOLD window
        rad = np.hypot(eh[win] - eh[i0], ev[win] - ev[i0])
        reached = float(np.max(np.hypot(eh[win], ev[win])))
        print(f' {d:4.0f}   {i0*DT:7.2f}s    {lat:6.0f}ms      {reached:5.1f}')
