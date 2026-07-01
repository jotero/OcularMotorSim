"""Debug the INO 'pulse': is it just the baseline-limited LR disinhibition fed
through the first-order plant (velocity = 0.5*diff / tau_p), or is there an extra
pulse on top?  Decompose the LEFT-eye yaw drive at peak velocity (g_mlf_L=0)."""
import numpy as np, jax, jax.numpy as jnp
from oculomotor.benchmarks.bench_saccades import _run, THETA_NOISELESS, DT
from oculomotor.sim.simulator import with_brain
import oculomotor.models.brain_models.final_common_pathway as fcp
from oculomotor.models.plant_models.muscle_geometry import LR_L, MR_L

NM = float(fcp._NERVE_MAX)
theta = with_brain(THETA_NOISELESS, g_mlf_L=0.0)
tau_p = float(theta.plant.tau_p)
base  = float(theta.brain.r_baseline[0])           # abducens (LR_L) baseline

t = np.arange(0.0, 1.0, DT)
pt3 = np.zeros((len(t), 3)); pt3[:, 2] = 100.0
pt3[:, 0] = np.where(t >= 0.1, 100.0 * np.tan(np.radians(20.0)), 0.0)   # far rightward 20deg
st = _run(t, jnp.array(pt3), key=0, params=theta)

L    = np.array(st.plant.left)[:, 0]
Lvel = np.gradient(L, DT)

mn    = jnp.array(np.array(st.brain.fcp.mn))
nerve = np.array(jax.vmap(lambda m: fcp._smooth_clip(
    fcp._ROUTE @ fcp.read_activations(fcp.State(mn=m), theta.brain).mn,
    theta.brain.g_nerve * NM))(mn))
lr, mr = nerve[:, LR_L], nerve[:, MR_L]
diff   = mr - lr
motor  = 0.5 * diff                                 # M_PLANT_EYE_L yaw decode = 0.5*(MR-LR)
vel_plant = (motor - 0.5 * (mr - lr)[80]) / tau_p    # 1st-order plant: (cmd - rest)/tau_p, x_p~rest

i0, ip = 80, int(np.argmax(np.abs(Lvel)))
print(f"baseline = {base:.1f} deg/s,  tau_p = {tau_p:.3f} s")
print(f"--- peak eye velocity at t={ip*DT:.2f}s ---")
print(f"  LR_L nerve : rest {lr[i0]:6.1f} -> {lr[ip]:6.1f}    disinhibition drop = {lr[i0]-lr[ip]:5.1f}  (max possible = baseline {base:.0f})")
print(f"  MR_L nerve : rest {mr[i0]:6.1f} -> {mr[ip]:6.1f}    MR contribution    = {mr[ip]-mr[i0]:+5.1f}")
print(f"  diff MR-LR : rest {diff[i0]:6.1f} -> {diff[ip]:6.1f}    PULSE in diff      = {diff[ip]-diff[i0]:+5.1f}")
print(f"  motor cmd  : 0.5*diff = {motor[ip]:6.1f} deg")
print(f"--- velocity check ---")
print(f"  measured eye vel          = {Lvel[ip]:6.1f} deg/s")
print(f"  1st-order plant predicts  = {vel_plant[ip]:6.1f} deg/s   (= 0.5*diff_pulse / tau_p)")
print(f"  i.e. baseline {base:.0f} -> 0.5*{base:.0f}={0.5*base:.0f} deg cmd -> /tau_p -> {0.5*base/tau_p:5.0f} deg/s")
