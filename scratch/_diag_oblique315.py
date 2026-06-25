"""Why does the 315 deg fan direction read 26.7% curvature while looking straight?
Extract that one window and locate where the max perpendicular deviation sits
(onset glitch / mid-flight real curve / hold drift)."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT

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
st = _run(t, jnp.array(pt3), key=0, max_s=int(T_end / DT) + 500, params=THETA_NOISELESS)
eye = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
eh, ev = eye[:, 0], eye[:, 1]
n_hold = int(HOLD / DT)

for i0, d in out_events:
    if d not in (270.0, 315.0):
        continue
    h = eh[i0:i0 + n_hold] - eh[i0]; v = ev[i0:i0 + n_hold] - ev[i0]
    speed = np.hypot(np.gradient(h, DT), np.gradient(v, DT))
    on = int(np.argmax(speed > 20.0))
    pts = np.column_stack([h[on:], v[on:]]); c = pts - pts.mean(0)
    axis = np.linalg.svd(c, full_matrices=False)[2][0]
    along = c @ axis; perp = c @ np.array([-axis[1], axis[0]])
    extent = float(np.ptp(along))
    k = int(np.argmax(np.abs(perp)))
    print(f'\n=== d={d:.0f}  onset_pos=({eh[i0]:+.2f},{ev[i0]:+.2f})  '
          f'window_end=({h[-1]:+.2f},{v[-1]:+.2f}) ===')
    print(f'  on={on}  extent={extent:.2f}  max|perp|={np.abs(perp).max():.3f}  '
          f'curv={np.abs(perp).max()/extent:.3f}')
    print(f'  max-perp at sample {on+k} (t+{(on+k)*DT*1000:.0f}ms)  '
          f'along={along[k]:+.2f}/{extent:.1f}  ({100*along[k]/max(extent,1e-9):+.0f}% of flight)')
    # trajectory snapshots
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        j = on + int(frac * (len(h) - 1 - on))
        print(f'    t+{(j)*DT*1000:4.0f}ms  pos=({h[j]:+.2f},{v[j]:+.2f})  '
              f'speed={speed[j]:6.1f}  perp={(c @ np.array([-axis[1], axis[0]]))[j-on]:+.3f}')
