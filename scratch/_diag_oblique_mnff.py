"""Verify the oblique curvature = horizontal MN under-compensation. Sweep
mn_ff_yaw and mlf_lead vs the max oblique curvature (robust PCA calc, same as the
bench). Prediction: curvature -> 0 as mn_ff_yaw -> 1.5 (matches the ~1.5-stage
horizontal version), and mlf_lead -> 1 cuts it at mn_ff_yaw=1.0 (per-eye exact)."""
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
    ts = T0 + i * HOLD; tgt_h[t >= ts] = h; tgt_v[t >= ts] = v
    if i % 2 == 0:
        out_events.append((int(ts / DT), float(DIRS[i // 2])))
pt3 = np.zeros((T, 3)); pt3[:, 2] = 1.0
pt3[:, 0] = np.tan(np.radians(tgt_h)); pt3[:, 1] = np.tan(np.radians(tgt_v))


def oblique_curv(params):
    st = _run(t, jnp.array(pt3), key=0, max_s=int(T_end / DT) + 500, params=params)
    eye = (np.array(st.plant.left) + np.array(st.plant.right)) / 2.0
    eh, ev = eye[:, 0], eye[:, 1]
    search_n = int(2 * HOLD / DT); THR = 20.0; merge_n = int(0.03 / DT)
    curvs = {}
    for i0, d in out_events:
        if d % 90 == 0:
            continue                                   # obliques only
        h = eh[i0:i0 + search_n] - eh[i0]; v = ev[i0:i0 + search_n] - ev[i0]
        a = (np.hypot(np.gradient(h, DT), np.gradient(v, DT)) > THR).astype(np.int8)
        if not a.any():
            continue
        edges = np.diff(np.concatenate([[0], a, [0]]))
        starts = list(np.where(edges == 1)[0]); ends = list(np.where(edges == -1)[0])
        runs = []
        for s, e in zip(starts, ends):
            if runs and s - runs[-1][1] <= merge_n:
                runs[-1] = (runs[-1][0], e)
            else:
                runs.append((s, e))
        ux, uy = np.cos(np.radians(d)), np.sin(np.radians(d)); best, bp = None, 5.0
        for s, e in runs:
            proj = (h[e - 1] - h[s]) * ux + (v[e - 1] - v[s]) * uy
            if proj > bp:
                bp, best = proj, (s, e)
        if best is None:
            continue
        s, e = best; c = np.column_stack([h[s:e], v[s:e]]); c = c - c.mean(0)
        ax = np.linalg.svd(c, full_matrices=False)[2][0]
        ext = float(np.ptp(c @ ax))
        if ext < 1.0:
            continue
        perp = c @ np.array([-ax[1], ax[0]])
        curvs[d] = float(np.max(np.abs(perp)) / ext)
    return curvs


print(f'{"mn_ff_yaw":>9} {"mlf_lead":>8}  {"max_curv":>9}   per-direction (45/135/225/315)')
for mnff, mlf in [(1.0, 0.0), (1.25, 0.0), (1.5, 0.0), (1.0, 0.5), (1.0, 1.0)]:
    P = with_brain(THETA_NOISELESS, mn_ff_yaw=mnff, mlf_lead=mlf)
    cv = oblique_curv(P)
    mx = max(cv.values()) if cv else float('nan')
    perdir = '  '.join(f'{int(d)}:{c:.3f}' for d, c in sorted(cv.items()))
    print(f'{mnff:9.2f} {mlf:8.1f}  {mx:9.3f}   {perdir}')
