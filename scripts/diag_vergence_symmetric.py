"""Symmetric vergence diagnostic — noiseless, AC/A + CA/C off, Listing off.

Sweeps a range of conv and div amplitudes; reports peak vergence velocity,
rise time (10–90%), and steady-state error per amplitude. SVBN cannot fire
in symmetric trials (no version saccade → no OPN pause → z_act ≈ 0), so this
isolates the slow integrator + direct phasic path.

Clinical targets (Zee 1992 Table 1, pure-vergence — no saccade):
    Convergence:  10° amplitude  →  peak ~41–58 deg/s
    Divergence:   2.5° amplitude →  peak ~9.5–13 deg/s

Usage:
    python -X utf8 scripts/diag_vergence_symmetric.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import PARAMS_DEFAULT, simulate, with_brain, with_sensory
from oculomotor.sim import kinematics as km

SHOW = '--show' in sys.argv
DT   = 0.001
IPD  = 0.064

# Noiseless debug params — same recipe as bench_vergence.py PARAMS_VERG_DEBUG.
# WITH_ACA env var lets us toggle AC/A on (default = off, pure vergence loop).
_aca_on = os.environ.get('WITH_ACA', '0') == '1'
PARAMS = with_brain(
    with_sensory(PARAMS_DEFAULT,
                 sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
    CA_C=0.0,
    **({} if _aca_on else {'AC_A': 0.0}),
    g_burst_verg=0.0,
    sigma_acc=0.0,
    listing_gain=0.0,
)


def _verg_angle(depth_m):
    return 2.0 * np.degrees(np.arctan(IPD / 2.0 / depth_m))


def _depth_for_verg(verg_deg):
    """Inverse — depth (m) that produces the given vergence angle."""
    return IPD / 2.0 / np.tan(np.radians(verg_deg) / 2.0)


def _run_sym_step(t, v_start_deg, v_end_deg, T_STEP):
    """Symmetric depth step from one vergence angle (deg) to another. Returns SimState."""
    d_start = _depth_for_verg(v_start_deg)
    d_end   = _depth_for_verg(v_end_deg)
    p0 = np.array([0.0, 0.0, float(d_start)])
    p1 = np.array([0.0, 0.0, float(d_end)])
    pt = np.where((t >= T_STEP)[:, None], p1, p0)
    return simulate(PARAMS, t,
                    target=km.build_target(t, lin_pos=pt),
                    scene_present_array=np.ones(len(t)),
                    return_states=True)


def _metrics(t, verg, T_STEP, target):
    """Peak velocity, 10-90 rise time, steady-state, peak position past target.

    Returns: (peak_vel, rise, ss, overshoot_pct)
        overshoot_pct: signed % of amplitude that the trace went BEYOND target
                       at any point post-step.  Positive = trace went past target
                       in the direction of motion (typical overshoot).
                       Negative = trace stayed short of target the whole time.
    """
    post = t >= T_STEP
    v_post = verg[post]
    t_post = t[post] - T_STEP

    vel = np.gradient(verg, DT)
    peak_vel = vel[post]
    direction = np.sign(target - verg[post][0])
    peak = float(direction * np.max(direction * peak_vel))

    # 10–90% rise time
    v0 = float(v_post[0])
    span = target - v0
    if abs(span) > 1e-3:
        lo_lvl = v0 + 0.1 * span
        hi_lvl = v0 + 0.9 * span
        crossed_lo = np.where(direction * (v_post - lo_lvl) >= 0)[0]
        crossed_hi = np.where(direction * (v_post - hi_lvl) >= 0)[0]
        t_lo = float(t_post[crossed_lo[0]]) if len(crossed_lo) else np.nan
        t_hi = float(t_post[crossed_hi[0]]) if len(crossed_hi) else np.nan
        rise = t_hi - t_lo
    else:
        rise = np.nan

    # Steady state: average over last 1 s
    ss = float(v_post[-int(1.0 / DT):].mean())

    # Peak excursion past target (signed by direction of motion)
    # direction · (v_max_past_target − target) > 0 means trace overshot.
    extreme = direction * np.max(direction * v_post)   # most "advanced" point in motion dir
    overshoot_deg = float(extreme - direction * abs(target))
    # Convert to % of amplitude
    if abs(span) > 1e-3:
        overshoot_pct = float(direction * (extreme - target) / abs(span) * 100.0)
    else:
        overshoot_pct = 0.0
    return peak, rise, ss, overshoot_pct


def main():
    # Amplitudes from a baseline of 4° vergence (≈ 1 m target)
    BASE_VERG  = 4.0   # deg
    AMPLITUDES = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0]
    T_STEP     = 1.0
    TOTAL      = 8.0   # long enough to reach SS
    t = np.arange(0.0, TOTAL, DT)

    # Allow gain sweep via env vars (only override if env var actually set)
    overrides = {}
    if 'K_PHASIC_VERG' in os.environ:
        overrides['K_phasic_verg'] = float(os.environ['K_PHASIC_VERG'])
    if 'K_VERG' in os.environ:
        overrides['K_verg'] = float(os.environ['K_VERG'])
    if 'TAU_VERG' in os.environ:
        overrides['tau_verg'] = float(os.environ['TAU_VERG'])
    params = with_brain(PARAMS, **overrides) if overrides else PARAMS
    print(f'K_phasic_verg = {params.brain.K_phasic_verg}   '
          f'K_verg = {params.brain.K_verg}   tau_verg = {params.brain.tau_verg} s')

    def _run(t_, v_start, v_end, T_STEP_):
        d_start = _depth_for_verg(v_start)
        d_end   = _depth_for_verg(v_end)
        p0 = np.array([0.0, 0.0, float(d_start)])
        p1 = np.array([0.0, 0.0, float(d_end)])
        pt = np.where((t_ >= T_STEP_)[:, None], p1, p0)
        return simulate(params, t_,
                        target=km.build_target(t_, lin_pos=pt),
                        scene_present_array=np.ones(len(t_)),
                        return_states=True)

    rows = []
    traces_conv = {}
    traces_div  = {}

    for amp in AMPLITUDES:
        # Convergence: from baseline outward → baseline + amp
        v_start = BASE_VERG
        v_end   = BASE_VERG + amp
        st = _run(t, v_start, v_end, T_STEP)
        eL = np.array(st.plant.left[:, 0])
        eR = np.array(st.plant.right[:, 0])
        verg_c = eL - eR
        traces_conv[amp] = verg_c
        pc, rc, sc, oc = _metrics(t, verg_c, T_STEP, v_end)
        rows.append(('conv', amp, v_start, v_end, pc, rc, sc, sc - v_end, oc))

        # Divergence: from baseline + amp → baseline
        v_start = BASE_VERG + amp
        v_end   = BASE_VERG
        st = _run(t, v_start, v_end, T_STEP)
        eL = np.array(st.plant.left[:, 0])
        eR = np.array(st.plant.right[:, 0])
        verg_d = eL - eR
        traces_div[amp] = verg_d
        pd_, rd, sd, od = _metrics(t, verg_d, T_STEP, v_end)
        rows.append(('div', amp, v_start, v_end, pd_, rd, sd, sd - v_end, od))

    # ── Print table ───────────────────────────────────────────────────────────
    print()
    print('Symmetric vergence — noiseless, AC/A=CA/C=0, Listing off')
    print('=' * 78)
    print(f'{"dir":>4}  {"amp":>5}  {"start":>6}  {"end":>6}  {"peak":>8}  {"rise":>7}  {"ss":>7}  {"err":>7}  {"over":>6}')
    print(f'{"":>4}  {"deg":>5}  {"deg":>6}  {"deg":>6}  {"deg/s":>8}  {"s":>7}  {"deg":>7}  {"deg":>7}  {"%":>6}')
    print('-' * 88)
    for r in rows:
        print(f'{r[0]:>4}  {r[1]:>5.1f}  {r[2]:>6.2f}  {r[3]:>6.2f}  {r[4]:>8.2f}  '
              f'{r[5]:>7.3f}  {r[6]:>7.2f}  {r[7]:>+7.2f}  {r[8]:>+6.1f}')
    print()
    print('Zee 1992 Table 1 (pure vergence):')
    print('  conv 10°  → peak  41–58 deg/s')
    print('  div  2.5° → peak   9.5–13 deg/s')
    print()

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle('Symmetric vergence — noiseless, AC/A=CA/C=0, Listing off',
                 fontsize=12, fontweight='bold')

    cmap = plt.cm.viridis
    norm = plt.Normalize(vmin=min(AMPLITUDES), vmax=max(AMPLITUDES))

    # Convergence traces
    ax = axes[0, 0]
    for amp in AMPLITUDES:
        ax.plot(t, traces_conv[amp], color=cmap(norm(amp)), lw=1.3, label=f'{amp:.0f}°')
        ax.axhline(BASE_VERG + amp, color=cmap(norm(amp)), lw=0.5, ls=':', alpha=0.4)
    ax.axvline(T_STEP, color='gray', lw=0.7, ls=':')
    ax.set_title('Convergence step (from 4°)', fontsize=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vergence L−R (deg)')
    ax.legend(fontsize=8, title='Amplitude', loc='lower right')
    ax.grid(True, alpha=0.3)

    # Divergence traces
    ax = axes[0, 1]
    for amp in AMPLITUDES:
        ax.plot(t, traces_div[amp], color=cmap(norm(amp)), lw=1.3, label=f'{amp:.0f}°')
        ax.axhline(BASE_VERG, color='gray', lw=0.5, ls=':', alpha=0.4)
    ax.axvline(T_STEP, color='gray', lw=0.7, ls=':')
    ax.set_title('Divergence step (back to 4°)', fontsize=10)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Vergence L−R (deg)')
    ax.legend(fontsize=8, title='Amplitude', loc='upper right')
    ax.grid(True, alpha=0.3)

    # Main sequence — peak velocity vs amplitude
    ax = axes[1, 0]
    amps  = [r[1] for r in rows if r[0] == 'conv']
    pks_c = [r[4] for r in rows if r[0] == 'conv']
    pks_d = [abs(r[4]) for r in rows if r[0] == 'div']
    ax.plot(amps, pks_c, 'o-', color='C0', lw=1.5, label='Conv (model)')
    ax.plot(amps, pks_d, 's--', color='C3', lw=1.5, label='Div (model)')
    # Zee bands
    ax.axhspan(41, 58, xmin=0, xmax=1, color='C0', alpha=0.10,
               label='Zee conv (41–58 °/s @ 10°)')
    ax.axhspan(9.5, 13, xmin=0, xmax=1, color='C3', alpha=0.10,
               label='Zee div (9.5–13 °/s @ 2.5°)')
    ax.axvline(10.0, color='C0', lw=0.5, ls=':')
    ax.axvline(2.5,  color='C3', lw=0.5, ls=':')
    ax.set_title('Main sequence (symmetric)', fontsize=10)
    ax.set_xlabel('Amplitude (deg)')
    ax.set_ylabel('Peak vergence velocity (deg/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Steady-state error vs amplitude
    ax = axes[1, 1]
    ss_err_c = [r[7] for r in rows if r[0] == 'conv']
    ss_err_d = [r[7] for r in rows if r[0] == 'div']
    ax.plot(amps, ss_err_c, 'o-', color='C0', lw=1.5, label='Conv (model)')
    ax.plot(amps, ss_err_d, 's--', color='C3', lw=1.5, label='Div (model)')
    ax.axhline(0, color='gray', lw=0.8, ls='--')
    ax.axhspan(-0.1, 0.1, color='green', alpha=0.10, label='±6 arcmin band')
    ax.set_title('Steady-state error', fontsize=10)
    ax.set_xlabel('Amplitude (deg)')
    ax.set_ylabel('SS − target (deg)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'diag_vergence_symmetric.png')
    fig.savefig(out, dpi=110)
    print(f'Figure saved → {out}')
    if SHOW:
        plt.show()


if __name__ == '__main__':
    main()
