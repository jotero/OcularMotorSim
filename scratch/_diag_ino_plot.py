"""INO adduction with the 2nd-order plant: left (adducting) eye position + velocity
for a 20deg rightward saccade — healthy vs partial INO vs complete INO."""
import numpy as np, jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

Z = 100.0
def saccade(theta, deg=20.0):
    t = np.arange(0.0, 1.0, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=theta, max_s=int(1.0 / DT) + 200)
    return t, np.array(st.plant.left)[:, 0]

conds = [('healthy (g_mlf=1)', 1.0, 'C0'),
         ('INO partial (g_mlf=0.3)', 0.3, 'C1'),
         ('INO complete (g_mlf=0)', 0.0, 'C3')]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
for label, g, c in conds:
    t, L = saccade(with_brain(THETA_NOISELESS, g_mlf_L=float(g)))
    pv = np.abs(np.gradient(L, DT)).max()
    ax1.plot(t, L, c, lw=1.6, label=f'{label}   (peak {pv:.0f}°/s, end {L[-1]:.1f}°)')
    ax2.plot(t, np.gradient(L, DT), c, lw=1.6)

ax1.axhline(20, color='k', ls=':', lw=0.8)
ax1.set_ylabel('left (adducting) eye position (deg)')
ax1.set_title('INO adduction — 2nd-order plant (left eye, 20° rightward saccade)')
ax1.legend(fontsize=8, loc='lower right'); ax1.grid(alpha=0.3)
ax2.set_ylabel('eye velocity (deg/s)'); ax2.set_xlabel('time (s)'); ax2.grid(alpha=0.3)
fig.tight_layout()
fig.savefig('scratch/_ino_adduction.png', dpi=110)
print('saved scratch/_ino_adduction.png')
