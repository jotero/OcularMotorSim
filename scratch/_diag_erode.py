"""Is the post-saccadic 'ring' real slow-phase drift, or np.gradient bleeding the
masked corrective-saccade velocity across the mask boundary? Erode the z_opn>=50
mask by ~15 ms each side so no boundary sample touches a masked fast phase."""
import numpy as np
from oculomotor.benchmarks.bench_saccades import _run, _pt3, extract_z_opn, THETA_NOISELESS, DT
from oculomotor.analysis import ni_net

T_end, t_jump = 0.9, 0.1
t = np.arange(0.0, T_end, DT)
K = 15  # erosion half-width (samples ~ ms)


def eroded(ok, k):
    er = ok.copy()
    for s in range(1, k + 1):
        er[s:] &= ok[:-s]
        er[:-s] &= ok[s:]
    return er


for amp in [5.0, 40.0]:
    pt3 = _pt3(t, amp, t_jump=t_jump)
    st = _run(t, pt3, key=0, max_s=int(T_end / DT) + 200, params=THETA_NOISELESS)
    L = np.array(st.plant.left); R = np.array(st.plant.right)
    ver = (L + R) / 2.0
    vrg = L - R
    vel_ver = np.gradient(ver[:, 0], DT)
    vel_vrg = np.gradient(vrg[:, 0], DT)
    z = extract_z_opn(st)
    base = (t >= t_jump + 0.05) & (t <= 0.45)
    ok = z >= 50.0
    m_cur = base & ok
    m_ero = base & eroded(ok, K)
    n_fast = int(np.sum(np.diff((~ok).astype(int)) > 0))   # fast-phase onsets in trace
    pkc = lambda v: float(np.max(np.abs(v[m_cur]))) if m_cur.any() else float('nan')
    pke = lambda v: float(np.max(np.abs(v[m_ero]))) if m_ero.any() else float('nan')
    print(f'\n=== {amp:.0f} deg  (#fast phases in trace = {n_fast}) ===')
    print(f'  version  ring: current-mask = {pkc(vel_ver):6.2f}   eroded-mask = {pke(vel_ver):6.2f} deg/s')
    print(f'  vergence ring: current-mask = {pkc(vel_vrg):6.2f}   eroded-mask = {pke(vel_vrg):6.2f} deg/s')
    print(f'  samples kept: current = {int(m_cur.sum())}, eroded = {int(m_ero.sum())}')
