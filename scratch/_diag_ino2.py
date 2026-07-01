"""2nd-order plant: healthy main sequence (accuracy) + INO with PARTIAL g_mlf
(adduction lag: slow ramp that still reaches + holds).  Sweep tau_muscle."""
import numpy as np, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain, with_plant

Z = 100.0
def saccade(theta, deg):
    t = np.arange(0.0, 1.2, DT); pt3 = np.zeros((len(t), 3)); pt3[:, 2] = Z
    pt3[:, 0] = np.where(t >= 0.1, Z * np.tan(np.radians(deg)), 0.0)
    st = _run(t, jnp.array(pt3), key=0, params=theta, max_s=int(1.2 / DT) + 200)
    return t, np.array(st.plant.left)[:, 0], np.array(st.plant.right)[:, 0]

for tau_m in (0.013, 0.020, 0.030):
    P = with_plant(THETA_NOISELESS, tau_muscle=tau_m)
    P = with_brain(P, tau_muscle=tau_m)                       # NI/vergence copy matches plant
    print(f"\n===== tau_muscle = {tau_m*1000:.0f} ms =====")
    print("  HEALTHY main sequence (accuracy + peak vel):")
    for deg in (5, 10, 20, 40):
        t, L, R = saccade(P, deg)
        pv = np.abs(np.gradient(L, DT)).max()
        print(f"    {deg:2d}deg: L_end {L[-1]:6.2f}  peakvel {pv:4.0f}  (formula {700*(1-np.exp(-deg/7)):4.0f})")
    print("  INO partial g_mlf_L=0.3 (want: slow ramp, reaches + holds):")
    for deg in (20, 40):
        t, L, R = saccade(with_brain(P, g_mlf_L=0.3), deg)
        pv = np.abs(np.gradient(L, DT)).max()
        held = L[-150:].mean()
        print(f"    {deg:2d}deg: L(adduct) peak|vel| {pv:4.0f}  final {L[-1]:6.2f}  held {held:6.2f}  R(abduct)_end {R[-1]:6.2f}")
