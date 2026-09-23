"""Which part of the floccular change destabilizes the VISUALLY-DRIVEN hold?

The dark/flashed (open-loop) sweep found no instability anywhere, even at 20x
floccular loop gain. The bench's "dark" is NOT open loop -- strobed=1 suppresses
only the target's velocity channel, its POSITION stays continuously visible -- and
there the same parameters give ~20 deg/s of centrifugal drift. So the failure needs
the visual position-feedback loop.

This separates the two things the re-split changed at once:
  * the LEAK itself            (tau_eff shortens)
  * the FLOCCULAR FEEDBACK LOOP (fl_drive, which scales as K/tau_i)

(25, 0.0) is the key diagnostic row: a 25 s leak with NO floccular feedback at all.
If that is stable while (5, 0.8) -- the same tau_eff via a strong loop -- is not, the
instability is the fl_drive loop closing through vision, not the leak.

Protocol = the bench's own: continuous target, strobed=1, noiseless, 6 s hold.
"""
import numpy as np, jax
from oculomotor.sim.simulator import PARAMS_DEFAULT, with_brain, with_sensory, simulate, SimConfig
from oculomotor.sim import kinematics as km
from oculomotor.analysis import extract_spv_states

DT, T_HOLD, T_STEP, T_MEAS = 0.001, 6.0, 0.1, 2.0
ECC = [0.0, 20.0, 40.0]
BASE = with_brain(with_sensory(PARAMS_DEFAULT, sigma_canal=0, sigma_pos=0,
                               sigma_vel=0, sigma_slip=0), sigma_acc=0)
t = np.arange(0.0, T_HOLD, DT); T = len(t)


def hold(theta, deg, lit):
    pt = np.zeros((T, 3), np.float32)
    pt[:, 0] = np.where(t < T_STEP, 0.0, np.tan(np.radians(deg))).astype(np.float32)
    pt[:, 2] = 1.0
    st = simulate(theta, t,
                  target=km.build_target(t, lin_pos=pt, lin_vel=np.zeros((T, 3), np.float32)),
                  target_strobed_array=np.ones(T, np.float32),          # bench condition
                  scene_present_array=(np.ones if lit else np.zeros)(T).astype(np.float32),
                  target_present_array=np.ones(T, np.float32),          # continuously present
                  max_steps=int(T * 1.1) + 1000, sim_config=SimConfig(warmup_s=0.0),
                  return_states=True, key=jax.random.PRNGKey(0))
    eye = (np.array(st.plant.left[:, 0]) + np.array(st.plant.right[:, 0])) / 2.0
    spv = extract_spv_states(st, t, eye='version')[:, 0]
    w = t >= T_MEAS
    s = spv[w][np.isfinite(spv[w])]
    return float(np.mean(s)) if len(s) else float('nan'), float(np.max(np.abs(eye)))


CANDIDATES = [
    ('old default       ', 25.0, 1.00),
    ('leak, NO FL loop  ', 25.0, 0.00),
    ('half loop gain    ', 10.0, 0.60),
    ('current default   ',  5.0, 0.80),
    ('strong leak, no FL',  5.0, 0.00),
]

print(f'{"candidate":20} {"tau_i":>6} {"K":>5} {"tau_eff":>8} '
      f'{"SPV@40 dark":>12} {"slope k":>9} {"SPV@40 light":>13} {"|eye|max":>9}')
print('-' * 92)
for label, tau_i, K in CANDIDATES:
    th = with_brain(BASE, tau_i=tau_i, K_cereb_fl=K)
    dark = [hold(th, d, False) for d in ECC]
    light40, _ = hold(th, 40.0, True)
    spvs = [d[0] for d in dark]
    slope = float(-np.polyfit(ECC, spvs, 1)[0])      # + = centripetal (normal)
    te = tau_i / (1 - K) if K < 1 else float('inf')
    print(f'{label:20} {tau_i:6.1f} {K:5.2f} {te:8.1f} {spvs[-1]:+12.3f} {slope:+9.4f} '
          f'{light40:+13.3f} {max(d[1] for d in dark):9.2f}')
print('\nslope k > 0 = centripetal (normal leak);  k < 0 = centrifugal (the failure)')
print('band: |k| <= 0.05 and |SPV@40| <= 4 deg/s dark, <= 2 light')
