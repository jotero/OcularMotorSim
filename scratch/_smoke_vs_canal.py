"""Behavior-preservation smoke test for the canal-plane VS refactor.
Runs VOR-in-dark + OKN and prints high-precision invariants so the pre/post
change outputs can be diffed exactly (should be bit-identical)."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA, DT
import oculomotor.sim.kinematics as km
from oculomotor.analysis import vs_net, ni_net, extract_spv_states

V, ON, OFF = 30.0, 20.0, 10.0
head = km.head_rotation_step(V, rotate_dur=ON, coast_dur=OFF, dt=DT)
t = np.array(head.t); hv = np.array(head.rot_vel); T = len(t)

# VOR in dark
st = _simulate(THETA, jnp.array(t), head_vel=jnp.array(hv),
               scene_present=jnp.zeros(T), target_present=jnp.zeros(T), key=0)
eye = np.array(st.plant.left)          # (T,3)
vsn = vs_net(st); nin = ni_net(st)

# OKN
tk = jnp.array(t)
sv = jnp.zeros((T, 3)).at[:, 0].set(jnp.where((tk >= 0.0) & (tk < ON), V, 0.0))
sp = jnp.where((tk >= 0.0) & (tk < ON), 1.0, 0.0)
stk = _simulate(THETA, tk, scene_vel=sv, scene_present=sp,
                target_present=jnp.zeros(T), key=1)
eyek = np.array(stk.plant.left)

def sig(name, a):
    a = np.asarray(a)
    print(f'{name:16s} sum={a.sum():.6f}  absmax={np.abs(a).max():.6f}  '
          f'std={a.std():.6f}')

print('=== VOR in dark ===')
sig('eye', eye); sig('vs_net', vsn); sig('ni_net', nin)
print('=== OKN ===')
sig('eye_okn', eyek)
print('checksum eye+okn:', float(np.array(st.plant.left).sum() + eyek.sum()))
