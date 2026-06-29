"""FCP cascade over a 9-position gaze sequence + near/far vergence shifts.
Stacked timecourse of the whole final common pathway, decomposed pulse/step:
  eye(version+vergence) -> version step(NI L/R) -> version pulse(tau_p*burst)
  -> vergence {step = verg_fast+verg_tonic | pulse = direct path | AC/A}
  -> MN(14, split by eye) -> nerves/muscles(12, split by eye).
Prototype for bench_fcp; helps visualise the [H,V,T]<->muscle mapping + vergence."""
import numpy as np, jax, jax.numpy as jnp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net, extract_burst
import oculomotor.models.brain_models.final_common_pathway as fcp
import oculomotor.models.brain_models.saccade_generator as sg
import oculomotor.models.brain_models.vergence_accommodation as va

NM = float(fcp._NERVE_MAX)
bp = THETA_NOISELESS.brain
TAU_P = float(THETA_NOISELESS.plant.tau_p)
g_nuc14  = jnp.concatenate([bp.g_nucleus, bp.g_nucleus[:2]])
r_base14 = jnp.concatenate([bp.r_baseline, jnp.zeros(2)])
m_proj   = fcp.M_NERVE_PROJ.at[fcp.MR_L, fcp.AIN_R].set(0.0).at[fcp.MR_R, fcp.AIN_L].set(0.0)

MN_LABELS  = ['LR_L', 'LR_R', 'CN4_L', 'CN4_R', 'MR_L', 'MR_R', 'SR_L', 'SR_R',
              'IR_L', 'IR_R', 'IO_L', 'IO_R', 'AIN_L', 'AIN_R']
NRV_LABELS = ['LR_L', 'MR_L', 'SR_L', 'IR_L', 'SO_L', 'IO_L',
              'LR_R', 'MR_R', 'SR_R', 'IR_R', 'SO_R', 'IO_R']
AXES3 = ['horizontal', 'vertical', 'torsion']


def rates_of(mn):    return fcp._smooth_clip(mn, NM)
def nerves_of(mn):
    z = m_proj @ (fcp._smooth_clip(mn, NM) + g_nuc14 * r_base14)
    return fcp._pullonly(fcp._smooth_clip_sym(z, bp.g_nerve * NM))


# ── sequence: gaze (1 m) center-out-center + near/far depth shifts ──────────────
#   (label, h_deg, v_deg, depth_m, hold_s)
GAZE = [('L', -30, 0), ('R', 30, 0), ('U', 0, 30), ('D', 0, -30),
        ('UR', 30, 30), ('UL', -30, 30), ('DL', -30, -30), ('DR', 30, -30)]
seq = [('c', 0, 0, 1.0, 1.0)]                              # settle at 1 m
for name, h, v in GAZE:
    seq += [(name, h, v, 1.0, 1.0), ('c', 0, 0, 1.0, 1.0)]
seq += [('near', 0, 0, 0.25, 2.0), ('c', 0, 0, 1.0, 2.0),  # depth shifts (2 s holds)
        ('far',  0, 0, 4.0, 2.0),  ('c', 0, 0, 1.0, 2.0)]

holds  = np.array([s[4] for s in seq])
starts = np.concatenate([[0.0], np.cumsum(holds)])
T_END  = starts[-1] + 0.5
t      = np.arange(0.0, T_END, DT)
seg    = np.clip(np.searchsorted(starts, t, side='right') - 1, 0, len(seq) - 1)
hh = np.array([s[1] for s in seq])[seg]; vv = np.array([s[2] for s in seq])[seg]
zz = np.array([s[3] for s in seq])[seg]
pt3 = np.zeros((len(t), 3))
pt3[:, 0] = zz * np.tan(np.radians(hh)); pt3[:, 1] = zz * np.tan(np.radians(vv)); pt3[:, 2] = zz
trans   = starts[1:len(seq)]
out_lab = [(seq[i][0], starts[i]) for i in range(1, len(seq)) if seq[i][0] != 'c']

st = _run(t, jnp.array(pt3), key=0, max_s=len(t) + 300, params=THETA_NOISELESS)

# ── extract the cascade ─────────────────────────────────────────────────────────
L = np.array(st.plant.left); R = np.array(st.plant.right)
eye      = 0.5 * (L + R)                                   # (T,3) version
eye_verg = L[:, 0] - R[:, 0]                               # (T,) ocular vergence
ni_L = np.array(st.brain.ni.L); ni_R = np.array(st.brain.ni.R)
burst = np.array(extract_burst(st, THETA_NOISELESS))      # (T,3) version velocity cmd
v_pulse = TAU_P * burst                                   # (T,3) version pulse (feed-through)
mn   = jnp.array(st.brain.fcp.mn)
rates = np.array(jax.vmap(rates_of)(mn))                  # (T,14)
nrv   = np.array(jax.vmap(nerves_of)(mn))                 # (T,12)

# vergence decomposition (H axis): step = verg_fast + verg_tonic;
# pulse = direct path = tau_p*(K_phasic*disparity + SVBN); AC/A = cross-link
vf = np.array(st.brain.va.verg_fast)          # (T,3)
vt = np.array(st.brain.va.verg_tonic)         # (T,3)
vc = np.array(st.brain.va.verg_copy)          # (T,3)
verg_step = vf + vt                           # (T,3) step = both integrators
disp = np.array(st.brain.pc.target_disparity)[:, -3:]     # (T,3) current disparity H/V/T
gate_opn = np.array(st.brain.sg.z_opn) / sg._OPN_TONIC
z_act    = np.clip(1.0 - gate_opn, 0.0, 1.0)[:, None]     # (T,1)
br = disp - vc
u_svbn = z_act * np.sign(br) * bp.g_svbn_conv * (1.0 - np.exp(-np.abs(br) / bp.X_svbn_conv))
verg_pulse = TAU_P * (bp.K_phasic_verg * disp + u_svbn)   # (T,3) direct path
aca = bp.AC_A * va._DEG_PER_PD * (np.array(st.brain.va.acc_fast) + np.array(st.brain.va.acc_slow))  # (T,) H-only

# ── plot ─────────────────────────────────────────────────────────────────────────
c3 = ['#c0392b', '#2980b9', '#27ae60']        # horizontal / vertical / torsion
c3_dk = ['#922b21', '#1a5276', '#1d8348']     # NI left pop  (dark)
c3_lt = ['#e6b0aa', '#a9cce3', '#a9dfbf']     # NI right pop (light)
PAIR = {'LR': '#1f4e79', 'MR': '#7fb3e6', 'SR': '#2e7d32', 'IR': '#9ccc65',
        'SO': '#6a1b9a', 'IO': '#ce93d8', 'CN4': '#6a1b9a', 'AIN': '#9e9e9e'}
def mcol(lbl): return PAIR.get(lbl.split('_')[0], '#333')

fig, ax = plt.subplots(9, 1, figsize=(14.5, 20.5), sharex=True)
for a in ax:
    for tt in trans:
        a.axvline(tt, color='#ececec', lw=0.5, zorder=0)
    a.axhline(0, color='k', lw=0.3); a.grid(True, alpha=0.12)

for k in range(3):
    ax[0].plot(t, eye[:, k],        color=c3[k],    lw=1.2, label=AXES3[k])
    ax[1].plot(t, ni_L[:, k],       color=c3_dk[k], lw=1.1, label=f'{AXES3[k]} L')
    ax[1].plot(t, ni_R[:, k],       color=c3_lt[k], lw=1.1, label=f'{AXES3[k]} R')
    ax[2].plot(t, v_pulse[:, k],    color=c3[k],    lw=1.0, label=AXES3[k])
    ax[3].plot(t, verg_step[:, k],  color=c3[k],    lw=1.2, label=AXES3[k])
    ax[4].plot(t, verg_pulse[:, k], color=c3[k],    lw=1.0, label=AXES3[k])
ax[0].plot(t, eye_verg, color='#8e44ad', lw=1.3, label='vergence (L−R)')
ax[3].plot(t, aca, color='#7f8c8d', lw=1.2, ls='--', label='AC/A (H)')

mn_L, mn_R = [0, 2, 4, 6, 8, 10, 12], [1, 3, 5, 7, 9, 11, 13]
for j in mn_L: ax[5].plot(t, rates[:, j], color=mcol(MN_LABELS[j]), lw=1.0, label=MN_LABELS[j].split('_')[0])
for j in mn_R: ax[6].plot(t, rates[:, j], color=mcol(MN_LABELS[j]), lw=1.0, label=MN_LABELS[j].split('_')[0])
for j in range(6):     ax[7].plot(t, nrv[:, j], color=mcol(NRV_LABELS[j]), lw=1.1, label=NRV_LABELS[j].split('_')[0])
for j in range(6, 12): ax[8].plot(t, nrv[:, j], color=mcol(NRV_LABELS[j]), lw=1.1, label=NRV_LABELS[j].split('_')[0])

titles = ['eye position (version) + ocular vergence  [deg]',
          'version STEP — NI bilateral pops, L/R push-pull  [deg]',
          'version PULSE — τp·burst (feed-through)  [deg]',
          'vergence STEP — verg_fast+tonic, H/V/T  (+ AC/A, H)  [deg]',
          'vergence PULSE — direct path, H/V/T  [deg]',
          'MN firing — LEFT nucleus  (CN4_L→SO_R, AIN_L→MR_R; signed)  [spk/s]',
          'MN firing — RIGHT nucleus  (CN4_R→SO_L, AIN_R→MR_L; signed)  [spk/s]',
          'nerves/muscles — LEFT eye  (pull-only)  [spk/s]',
          'nerves/muscles — RIGHT eye  (pull-only)  [spk/s]']
for a, ti in zip(ax, titles):
    a.set_title(ti, fontsize=9, loc='left')
    for name, tt in out_lab:
        a.text(tt, 0.90, name, transform=a.get_xaxis_transform(), fontsize=7,
               color='#333', ha='center', va='top',
               bbox=dict(boxstyle='round,pad=0.1', fc='white', ec='none', alpha=0.65))

for i in (0, 1, 2, 3, 4, 5, 7):
    ax[i].legend(loc='upper left', bbox_to_anchor=(1.005, 1.0), fontsize=7.5, frameon=False)
ax[-1].set_xlabel('time (s)')
fig.suptitle('Final Common Pathway cascade — 9-position gaze (±30°, 1 m) + near/far vergence', fontsize=12)
fig.tight_layout()
out = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_fcp_cascade.png'
fig.savefig(out, dpi=110, bbox_inches='tight'); print('saved', out)
print(f'T={T_END:.0f}s. eye final H/V={eye[-1,0]:.1f}/{eye[-1,1]:.1f}, '
      f'vergence range {eye_verg.min():.1f}..{eye_verg.max():.1f} deg')
