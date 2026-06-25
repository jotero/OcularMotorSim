"""Is the VOR-Bode low-frequency phase jump a fit artifact? Reproduce the exact
bench run for light+target at 0.05 Hz: the capped 50 s record (what the bench
uses) vs an extended, settled record. Overlay the single-sinusoid fit so the
un-settled / distorted waveform is visible. If capped != extended, the low-f
points are artifacts of too-few cycles + un-settled velocity-storage transient."""
import numpy as np, jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_vor_okr import _simulate, THETA_NOISELESS, DT
from oculomotor.analysis import extract_spv_states
from oculomotor.benchmarks import bode

AMP, SETTLE, f = 30.0, 2.0, 0.05
w = 2 * np.pi * f
PERIOD = 1.0 / f                       # 20 s


def run(T_end, scene_p, target_p):
    t = np.arange(0.0, T_end, DT); Tn = len(t)
    on = t >= SETTLE
    hv = np.zeros((Tn, 3)); hv[:, 0] = np.where(on, AMP * np.sin(w * (t - SETTLE)), 0.0)
    st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                   scene_present=jnp.full(Tn, scene_p),
                   target_present=jnp.full(Tn, target_p), key=0)
    spv = np.array(extract_spv_states(st, t, eye='version'))[:, 0]
    return t, hv[:, 0], -spv            # -spv: compensatory eye in phase with head


def gp(t, drive, out, settle_frac=0.45):
    g, p = bode.bode_point(t, drive, out, f, settle_frac)
    # also return the fitted-sinusoid params on the analysis window for plotting
    t0, t1 = t[0], t[-1]
    mask = t >= (t0 + settle_frac * (t1 - t0))
    a, ph, c = bode.fit_sinusoid(t[mask], out[mask], f)
    return g, p, mask, (a, ph, c)


print(f'f = {f} Hz  (period {PERIOD:.0f} s,  velocity-storage TC ~34.6 s)')
print(f'{"condition":26} {"T_end":>7} {"cycles":>7} {"gain":>7} {"phase":>8}')
rows = {}
for lab, scn, tgt, T_end in [
        ('dark, capped 50s',         0.0, 0.0,  50.0),
        ('light+target, capped 50s', 1.0, 1.0,  50.0),
        ('light+target, extended',   1.0, 1.0, 250.0)]:
    t, drive, out = run(T_end, scn, tgt)
    g, p, mask, fit = gp(t, drive, out)
    cyc = (T_end - SETTLE) / PERIOD
    print(f'{lab:26} {T_end:7.0f} {cyc:7.1f} {g:7.3f} {p:+8.1f}')
    rows[lab] = (t, drive, out, mask, fit)

# ── plot: capped (artifact) vs extended (truth) for light+target ──────────────
fig, axes = plt.subplots(2, 1, figsize=(11, 7))
for ax, key in zip(axes, ('light+target, capped 50s', 'light+target, extended')):
    t, drive, out, mask, (a, ph, c) = rows[key]
    ax.plot(t, drive / AMP, color='#888', lw=0.8, label='head vel (norm)')
    ax.plot(t, out, color='#c0392b', lw=1.2, label='eye SPV (−)')
    fitw = a * np.sin(w * t[mask] + ph) + c
    ax.plot(t[mask], fitw, '--', color='k', lw=1.4, label='single-sinusoid fit (analysis window)')
    ax.axvspan(t[mask][0], t[-1], color='gold', alpha=0.12, label='fit window')
    ax.axhline(0, color='k', lw=0.3)
    ax.set_title(key); ax.set_ylabel('deg/s (eye) · norm (head)')
    ax.legend(fontsize=8, loc='upper right'); ax.grid(True, alpha=0.2)
axes[1].set_xlabel('time (s)')
fig.suptitle('VOR Bode 0.05 Hz light+target: capped record (artifact) vs settled', fontsize=12)
fig.tight_layout()
out_png = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_vor_bode_lowf.png'
fig.savefig(out_png, dpi=110)
print('saved', out_png)
