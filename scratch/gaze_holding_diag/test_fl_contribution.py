"""Where does gaze-holding drift come from: intrinsic NI leak, flocculus, or NI null?

40 deg eccentric, DARK, target genuinely flashed 20 ms/1 s (open-loop between
flashes). Sweeps the two candidate mechanisms:
  K_cereb_fl   1.0 = perfect leak cancellation, 0 = floccular lesion (raw tau_i leak)
  tau_ni_adapt 20 s default, 1e6 = null adaptation disabled
Expected if the leak were uncancelled: drift = -ecc/tau_i = -40/25 = -1.6 deg/s.
"""
import numpy as np, jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import ni_net, ni_null

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


def run(theta):
    st = simulate(theta, t,
                  target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32)),
                  target_strobed_array=np.zeros(T, np.float32),
                  scene_present_array=np.zeros(T, np.float32),
                  target_present_array=flash,
                  max_steps=int(T * 1.1) + 1000, sim_config=SimConfig(warmup_s=0.0),
                  return_states=True, key=jax.random.PRNGKey(0))
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    # Open-loop drift: last 700 ms of each inter-flash gap, averaged
    seg = [(eye[int((k + 0.95) / DT)] - eye[int((k + 0.25) / DT)]) / 0.70
           for k in range(int(ACQ) + 1, int(TEND) - 1)]
    return eye, float(np.mean(seg)), np.array(ni_net(st))[:, 0], np.array(ni_null(st))[:, 0]


CONDS = [
    ('default (fl=1, adapt=20 s)',      BASE),
    ('flocculus LESION (fl=0)',         with_brain(BASE, K_cereb_fl=0.0)),
    ('half flocculus (fl=0.5)',         with_brain(BASE, K_cereb_fl=0.5)),
    ('no null adapt (adapt=1e6)',       with_brain(BASE, tau_ni_adapt=1e6)),
    ('fl=0 AND no null adapt',          with_brain(BASE, K_cereb_fl=0.0, tau_ni_adapt=1e6)),
]

print(f'{"condition":30} {"drift":>9} {"tau_eff":>10} {"eye@2s":>8} {"eye@end":>8} '
      f'{"ni_null@end":>12}')
print('-' * 84)
for label, th in CONDS:
    eye, drift, ni, null = run(th)
    tau = (-DEG / drift) if abs(drift) > 1e-4 else float('inf')
    tau_s = f'{tau:10.1f}' if np.isfinite(tau) else '       inf'
    print(f'{label:30} {drift:+9.4f} {tau_s} {eye[2000]:8.3f} {eye[-1]:8.3f} {null[-1]:12.3f}')
print('\ndrift < 0 = centripetal (toward centre, normal leak);  > 0 = centrifugal')
print(f'raw intrinsic leak would give -{DEG}/tau_i = {-DEG / float(BASE.brain.tau_i):.3f} deg/s')
