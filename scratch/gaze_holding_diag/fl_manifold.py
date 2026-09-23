"""Constant-tau_eff manifold: can the cerebellum give gaze-evoked nystagmus?

The flocculus adds  fl_drive = K_cereb_fl * (ni_net - ni_null) / tau_i  to the NI
velocity input, so the NI leak becomes -(1-K)/tau_i and

    tau_eff = tau_i / (1 - K_cereb_fl)

With the current default tau_i = 25 s, tau_eff >= 25 s for EVERY K in [0,1]: a full
floccular lesion still leaves a normal gaze-holding TC, so GEN is unreachable. The fix
is to re-split the pair -- tau_i = the intrinsic brainstem leak, K = the cerebellar
extension. This sweeps (tau_i, K) pairs that all target the SAME tau_eff while the
floccular loop gain (~1/tau_i) varies 20x, then lesions K at each point to see the
GEN that should appear.

Protocol: 40 deg eccentric, DARK, target GENUINELY flashed 20 ms / 1 s (open loop
between flashes -- no visual feedback, so this measures the integrator alone).
"""
import numpy as np, jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ni_net, extract_spv_states

DT, DEG, TEND = 0.001, 40.0, 10.0
T_STEP, ACQ, PERIOD, ON_MS = 0.1, 1.0, 1.0, 20.0

BASE = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0, sigma_pos=0,
                               sigma_vel=0, sigma_slip=0), sigma_acc=0)
t = np.arange(0.0, TEND, DT); T = len(t)

flash = np.zeros(T, np.float32)
flash[(t >= T_STEP) & (t < ACQ)] = 1.0
flash[(t >= ACQ) & (np.mod(t - ACQ, PERIOD) < ON_MS * 1e-3)] = 1.0
pt = np.zeros((T, 3), np.float32)
pt[:, 0] = np.where(t < T_STEP, 0.0, np.tan(np.radians(DEG))).astype(np.float32)
pt[:, 2] = 1.0


def run(tau_i, K):
    th = with_brain(BASE, tau_i=float(tau_i), K_cereb_fl=float(K))
    st = simulate(th, t,
                  target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32)),
                  target_strobed_array=np.zeros(T, np.float32),   # flag OFF: real flashes
                  scene_present_array=np.zeros(T, np.float32),    # DARK
                  target_present_array=flash,
                  max_steps=int(T * 1.1) + 1000, sim_config=SimConfig(warmup_s=0.0),
                  return_states=True, key=jax.random.PRNGKey(0))
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    # Open-loop drift: slope over the last 700 ms of each inter-flash gap.
    segs = [(eye[int((k + 0.95) / DT)] - eye[int((k + 0.25) / DT)]) / 0.70
            for k in range(int(ACQ) + 1, int(TEND) - 1)]
    drift = float(np.mean(segs))
    # Oscillation: ripple of eye velocity WITHIN the gaps, after removing the drift.
    v = np.gradient(eye, DT)
    gap = np.zeros(T, bool)
    for k in range(int(ACQ) + 1, int(TEND) - 1):
        gap[int((k + 0.25) / DT):int((k + 0.95) / DT)] = True
    ripple = float(np.std(v[gap] - drift))
    # True slow-phase velocity, quick phases masked out — the only valid readout
    # once nystagmus appears (net displacement then measures slow phase MINUS the
    # quick phases that undo it, which underestimates the leak badly).
    spv_all = extract_spv_states(st, t, eye='version')[:, 0]
    sp = spv_all[gap]
    spv = float(np.nanmean(sp[np.isfinite(sp)])) if np.isfinite(sp).any() else float('nan')
    return drift, ripple, float(np.max(np.abs(eye))), spv


PAIRS = [(25.0, 1.00), (25.0, 0.00), (10.0, 0.60), (5.0, 0.80),
         (2.5, 0.90), (2.5, 0.88), (1.25, 0.95)]

print(f'{"tau_i":>6} {"K_fl":>6} {"tau_eff pred":>13} {"SPV deg/s":>12} '
      f'{"tau_eff meas":>13} {"ripple":>9} {"|eye|max":>9}')
print('-' * 76)
for tau_i, K in PAIRS:
    pred = tau_i / (1 - K) if K < 1 else float('inf')
    d, rip, mx, spv = run(tau_i, K)
    meas = (-DEG / spv) if abs(spv) > 1e-4 else float('inf')
    print(f'{tau_i:6.2f} {K:6.2f} {pred:13.1f} {spv:+12.4f} {meas:13.1f} {rip:9.3f} {mx:9.2f}')

print('\n--- LESION (K_cereb_fl = 0) at each tau_i: does GEN appear? ---')
print(f'{"tau_i":>6} {"tau_eff pred":>13} {"SPV deg/s":>12} {"tau_eff meas":>13} {"ripple":>9}')
print('-' * 58)
for tau_i in (25.0, 10.0, 5.0, 2.5, 1.25):
    d, rip, mx, spv = run(tau_i, 0.0)
    meas = (-DEG / spv) if abs(spv) > 1e-4 else float('inf')
    print(f'{tau_i:6.2f} {tau_i:13.1f} {spv:+12.4f} {meas:13.1f} {rip:9.3f}')
print('\ndrift < 0 = centripetal (normal leak / GEN);  ripple >> 0 = oscillation')
