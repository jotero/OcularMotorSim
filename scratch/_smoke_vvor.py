"""Isolate VVOR/OKN/VOR gain regression for the canal-VS change."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA, DT
import oculomotor.sim.kinematics as km
from oculomotor.analysis import extract_spv_states, vs_net

V, ON = 30.0, 30.0
head = km.head_rotation_step(V, rotate_dur=ON, coast_dur=5.0, dt=DT)
t = np.array(head.t); hv = np.array(head.rot_vel); T = len(t); tj = jnp.array(t)
scene_on = jnp.where((tj >= 0) & (tj < ON), 1.0, 0.0)
mask = (t > 10) & (t < 25)
hd = np.mean(np.abs(hv[mask, 0])) + 1e-9

st_v = _simulate(THETA, tj, head_vel=jnp.array(hv), scene_present=jnp.zeros(T),
                 target_present=jnp.zeros(T), key=0)
gain_v = np.mean(np.abs(-extract_spv_states(st_v, t)[:, 0][mask])) / hd

st_vv = _simulate(THETA, tj, head_vel=jnp.array(hv), scene_present=scene_on,
                  target_present=jnp.zeros(T), key=2)
gain_vv = np.mean(np.abs(-extract_spv_states(st_vv, t)[:, 0][mask])) / hd

sv = jnp.zeros((T, 3)).at[:, 0].set(jnp.where((tj >= 0) & (tj < ON), V, 0.0))
st_o = _simulate(THETA, tj, scene_vel=sv, scene_present=scene_on,
                 target_present=jnp.zeros(T), key=1)
gain_o = np.mean(np.abs(extract_spv_states(st_o, t)[:, 0][mask])) / V

print(f'VOR_dark_gain={gain_v:.4f}  VVOR_gain={gain_vv:.4f}  OKN_gain={gain_o:.4f}')
print(f'vs_net_vvor: absmax={np.abs(vs_net(st_vv)).max():.4f} '
      f'sum={vs_net(st_vv).sum():.3f}')
print(f'vs_net_okn:  absmax={np.abs(vs_net(st_o)).max():.4f} '
      f'sum={vs_net(st_o).sum():.3f}')
