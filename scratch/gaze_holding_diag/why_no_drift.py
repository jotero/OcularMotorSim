"""Why doesn't the strobed eye drift toward zero?  Because K_cereb_fl=1 cancels the
leak exactly.  Sweep the cancellation gain and watch the centripetal drift appear."""
import numpy as np, jax, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_spv_states, ni_net, ni_null

DT, DEG, TEND = 0.001, 40.0, 10.0
T_STEP, ACQ, PERIOD, ON_MS = 0.1, 1.0, 1.0, 20.0
B = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0, sigma_pos=0,
                            sigma_vel=0, sigma_slip=0), sigma_acc=0)
t = np.arange(0.0, TEND, DT); T = len(t)
pt = np.zeros((T, 3), np.float32)
pt[:, 0] = np.where(t < T_STEP, 0.0, np.tan(np.radians(DEG))).astype(np.float32); pt[:, 2] = 1.0
pulsed = np.where((t >= T_STEP) & (t < ACQ), 1.0, 0.0).astype(np.float32)
pulsed[(t >= ACQ) & (np.mod(t - ACQ, PERIOD) < ON_MS * 1e-3)] = 1.0

def run(K):
    th = with_brain(B, K_cereb_fl=K)
    st = simulate(th, t, target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T,3), np.float32)),
                  target_strobed_array=np.zeros(T, np.float32),
                  scene_present_array=np.zeros(T, np.float32),
                  target_present_array=pulsed, max_steps=int(T*1.1)+1000,
                  sim_config=SimConfig(warmup_s=0.0), return_states=True, key=jax.random.PRNGKey(0))
    eye = (np.array(st.plant.left[:,0]) + np.array(st.plant.right[:,0]))/2.0
    spv = extract_spv_states(st, t, eye='version')[:,0]
    w = (t >= 2.0); v = spv[w][np.isfinite(spv[w])]
    return eye, float(np.mean(v)), np.array(ni_net(st))[:,0], np.array(ni_null(st))[:,0]

fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
fig.suptitle(f'{DEG:.0f}° eccentric, DARK, target strobed {ON_MS:.0f} ms/{PERIOD:.0f} s — '
             f'why no centripetal drift?\nK_cereb_fl scales how much of the {B.brain.tau_i:g} s '
             f'NI leak the flocculus cancels (1.0 = cancel it ALL)', fontsize=12, fontweight='bold')
print(f'{"K_cereb_fl":>11} {"SPV":>9} {"tau_eff":>9} {"eye@10s":>9}   expected leak = '
      f'-{DEG}/{B.brain.tau_i:g} = {-DEG/float(B.brain.tau_i):.2f} deg/s')
cmap = plt.get_cmap('viridis')
for i, K in enumerate((1.0, 0.9, 0.75, 0.5, 0.0)):
    eye, spv, ni, nul = run(K)
    tau = DEG/abs(spv) if abs(spv) > 1e-6 else float('inf')
    print(f'{K:11.2f} {spv:+9.3f} {tau:9.1f} {eye[-1]:9.3f}')
    c = cmap(i/4)
    axes[0].plot(t, eye, color=c, lw=1.5, label=f'K_fl={K:.2f}  (SPV {spv:+.2f} deg/s, τ≈{tau:.0f} s)')
    axes[1].plot(t, ni,  color=c, lw=1.3)
axes[0].axhline(DEG, color='k', lw=0.6, ls=':')
axes[0].set_ylabel('Eye yaw (deg)'); axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.15)
axes[0].set_title('Eye position — the leak only appears once the flocculus stops cancelling it')
axes[1].axhline(DEG, color='k', lw=0.6, ls=':')
axes[1].set_ylabel('NI net (deg)'); axes[1].set_xlabel('Time (s)'); axes[1].grid(True, alpha=0.15)
axes[1].set_title('Neural integrator state (same colours)')
fig.tight_layout(rect=[0,0,1,0.93])
fig.savefig('scratch/gaze_holding_diag/why_no_drift.png', dpi=110)
print('\nfigure -> scratch/gaze_holding_diag/why_no_drift.png')
