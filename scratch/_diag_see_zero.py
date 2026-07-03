"""Validate the series-elastic plant zero + pulse-slide-step.
Matched Tz=Ts should give eye = NI_net exactly: full peak velocity, low ring,
undistorted main sequence. Compare to slide-without-zero and the sharp baseline."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain, with_plant
from oculomotor.analysis import ni_net

Z = 100.0
def sac(theta, deg=20.0, T=0.7):
    t = np.arange(0, T, DT); pt = np.zeros((len(t), 3)); pt[:, 2] = Z
    pt[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt), key=0, params=theta, max_s=int(T / DT) + 200)
    return t, np.array(st.plant.left)[:, 0], ni_net(st)[:, 0]

def metrics(theta, deg=20.0):
    t, eye, nin = sac(theta, deg)
    ev = np.gradient(eye, DT); pkv = np.abs(ev).max()
    pi = int(np.argmax(np.abs(ev)))
    below = np.where(np.abs(ev[pi:]) < 30.0)[0]; ei = pi + (int(below[0]) if len(below) else 0)
    win = (t > t[ei] + 0.02) & (t < t[ei] + 0.18)
    ring = eye[win].max() - eye[win].min()
    tw = (t > t[ei]) & (t < t[ei] + 0.15)
    trackerr = np.abs(eye[tw] - nin[tw]).max()      # |eye − NI_net| after saccade
    return pkv, ring, eye[-1], trackerr

def cfg(ts, tz):
    return with_plant(with_brain(THETA_NOISELESS, tau_slide=float(ts)), tau_see=float(tz))

print(f'{"config":32} {"peakvel":>7} {"ring":>6} {"final":>6} {"|eye-ni|":>8}')
for name, th in [
        ('NEW matched  (Tz=Ts=8ms)',  cfg(0.008, 0.008)),
        ('slide, NO zero (Ts=8,Tz=0)', cfg(0.008, 0.000)),
        ('sharp baseline (Ts=1,Tz=0)', cfg(0.001, 0.000))]:
    pkv, ring, fin, te = metrics(th)
    print(f'{name:32} {pkv:>7.0f} {ring:>6.3f} {fin:>6.2f} {te:>8.3f}')

print('\nmain sequence (peak vel), NEW matched vs target 700·(1−e^−A/7):')
th = cfg(0.008, 0.008)
print(f'{"deg":>4} {"target":>7} {"NEW":>7} {"ring":>6}')
for deg in [5, 10, 20, 30, 40]:
    ms = 700 * (1 - np.exp(-deg / 7.0))
    pkv, ring, _, _ = metrics(th, deg)
    print(f'{deg:>4} {ms:>7.0f} {pkv:>7.0f} {ring:>6.3f}')
