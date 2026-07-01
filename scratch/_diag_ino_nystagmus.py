"""INO + compensatory increased pulse (burst) gain → dissociated (abducting)
nystagmus?  Left INO (g_mlf_L=0.3), rightward 20deg saccade + hold, g_burst 2x.
Quantify the post-saccade oscillation amplitude per eye (abducting vs adducting)."""
import numpy as np, jax.numpy as jnp
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain

Z = 100.0
def saccade(theta, deg=20.0, T=1.5):
    t = np.arange(0.0, T, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=theta, max_s=int(T / DT) + 200)
    return t, np.array(st.plant.left)[:, 0], np.array(st.plant.right)[:, 0]

fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
for col, (gb, tag) in enumerate([(700, 'normal gain'), (1400, 'compensatory ↑gain (2×)')]):
    theta = with_brain(THETA_NOISELESS, g_mlf_L=0.3, g_burst=float(gb))
    t, L, R = saccade(theta)
    # oscillation amplitude = peak-to-peak of position after the primary saccade (t>0.45s)
    win = t > 0.45
    ppL = L[win].max() - L[win].min(); ppR = R[win].max() - R[win].min()
    axes[0, col].plot(t, R, 'C0', lw=1.4, label=f'right (abducting)  osc {ppR:.1f}°')
    axes[0, col].plot(t, L, 'C1', lw=1.4, label=f'left (adducting)   osc {ppL:.1f}°')
    axes[0, col].axhline(20, color='k', ls=':', lw=0.8)
    axes[0, col].set_title(f'left INO g_mlf=0.3 — {tag}  (g_burst={gb})')
    axes[0, col].set_ylabel('eye position (deg)'); axes[0, col].grid(alpha=0.3)
    axes[0, col].legend(fontsize=8, loc='lower right')
    axes[1, col].plot(t, np.gradient(R, DT), 'C0', lw=0.9)
    axes[1, col].plot(t, np.gradient(L, DT), 'C1', lw=0.9)
    axes[1, col].set_ylabel('eye velocity (deg/s)'); axes[1, col].set_xlabel('time (s)'); axes[1, col].grid(alpha=0.3)
    print(f"g_burst={gb}: post-saccade oscillation  abducting {ppR:.2f}°  adducting {ppL:.2f}°  (ratio {ppR/max(ppL,0.01):.1f}×)")
fig.suptitle('INO + increased pulse gain → dissociated (abducting-predominant) oscillation', y=1.0)
fig.tight_layout(); fig.savefig('scratch/_ino_nystagmus.png', dpi=115)
print('saved')
