"""Tune mlf_lead by the EC/version criterion (post-saccadic ring on (L+R)/2 vel),
fold vs pair-shift. Ring = the version error that leaks into VS/OKR (what the EC
must cancel for clean pursuit/OKN). Also report disconjugacy + peak vel as guards.
Suppression OFF (loops only), so the ring is the controller's own residual."""
import sys
import numpy as np, jax.numpy as jnp
import oculomotor.models.brain_models.final_common_pathway as fcp
fcp._FOLD = (len(sys.argv) > 1 and sys.argv[1] == 'fold')
from oculomotor.benchmarks.bench_saccades import _run, _pt3, extract_z_opn, THETA_NOISELESS, DT
from oculomotor.analysis import extract_burst
from oculomotor.sim.simulator import with_brain

t = np.arange(0.0, 0.9, DT)

def metrics(mlf, amp):
    P = with_brain(THETA_NOISELESS, mn_ff_yaw=1.0, mlf_lead=mlf, saccadic_suppression_steepness=0.0)
    st = _run(t, _pt3(t, amp, t_jump=0.1), key=0, max_s=int(0.9 / DT) + 200, params=P)
    L = np.array(st.plant.left)[:, 0]; R = np.array(st.plant.right)[:, 0]
    eye = (L + R) / 2.0; evel = np.gradient(eye, DT)
    z = extract_z_opn(st); ub = np.array(extract_burst(st, P))[:, 0]
    fast = np.abs(ub) > 20.0
    be = int(np.where(fast)[0][-1]) if fast.any() else 0
    rw = np.zeros_like(t, bool); rw[be + 30:be + 230] = True; rw &= (z >= 50.0)
    ring = float(np.max(np.abs(evel[rw]))) if rw.any() else float('nan')
    lr = L - R; disconj = float(np.max(np.abs(lr - lr[-1])))
    return ring, disconj, float(np.abs(evel).max())

mlfs = [0.0, 0.5, 1.0, 1.5, 2.0]
amps = [5.0, 10.0, 20.0]
grid = {(a, m): metrics(m, a) for a in amps for m in mlfs}

mode = "FOLD" if fcp._FOLD else "PAIR-SHIFT"
hdr = f'{"amp":>4} ' + ' '.join(f'{"mlf="+format(m,".2f"):>8}' for m in mlfs)
print(f'==== mode: {mode} ====')
print('post-sac version RING deg/s  [EC/version error, lower=better]:'); print(hdr + '   best')
for a in amps:
    r = [grid[(a, m)][0] for m in mlfs]
    print(f'{a:4.0f} ' + ' '.join(f'{x:8.2f}' for x in r) + f'   @{mlfs[int(np.nanargmin(r))]:.2f}')
print('transient disconjugacy deg  [lower=better]:'); print(hdr + '   best')
for a in amps:
    d = [grid[(a, m)][1] for m in mlfs]
    print(f'{a:4.0f} ' + ' '.join(f'{x:8.3f}' for x in d) + f'   @{mlfs[int(np.nanargmin(d))]:.2f}')
print('saccade peak version vel deg/s:'); print(hdr)
for a in amps:
    p = [grid[(a, m)][2] for m in mlfs]
    print(f'{a:4.0f} ' + ' '.join(f'{x:8.0f}' for x in p))
