"""VOR / OKR benchmarks — Raphan Fig.9 replication, nystagmus zoom, TC comparison, cascade.

Usage:
    python -X utf8 scripts/bench_vor_okr.py
    python -X utf8 scripts/bench_vor_okr.py --show
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from oculomotor.benchmarks import bench_utils as utils

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib
if '--show' not in sys.argv:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor.sim.simulator import (
    PARAMS_DEFAULT, with_brain, with_sensory, simulate,
)
from oculomotor.models.sensory_models.sensory_model import (
    N_CANALS, FLOOR, _SOFTNESS, PINV_SENS,
)
from oculomotor.models.brain_models.perception_cyclopean import C_slip
from oculomotor.sim import kinematics as km
from oculomotor.analysis import (
    ax_fmt, extract_canal, vs_net, vs_null, ni_net, fit_tc, extract_spv_states,
    read_brain_acts,
)
from oculomotor.benchmarks.bench_metrics import Metric
from oculomotor.benchmarks import bode

SHOW  = '--show' in sys.argv
DT    = 0.001

# Behavioral params: realistic noise on (defaults from SensoryParams + BrainParams).
THETA = PARAMS_DEFAULT

# Cascade DEBUG params: all sensory + accumulator noise off so the cascade traces
# show only the pure signal flow (no microsaccades, no fixational drift).
THETA_NOISELESS = with_brain(
    with_sensory(PARAMS_DEFAULT, sigma_canal=0.0, sigma_slip=0.0, sigma_pos=0.0, sigma_vel=0.0),
    sigma_acc=0.0)


def _simulate(theta, t_arr, head_vel=None, scene_vel=None, scene_present=None,
              target_present=None, target=None, key=0):
    T   = len(t_arr)
    t   = np.array(t_arr)
    hv  = np.array(head_vel)  if head_vel  is not None else np.zeros((T, 3), np.float32)
    sv  = np.array(scene_vel) if scene_vel is not None else np.zeros((T, 3), np.float32)
    sp  = scene_present  if scene_present  is not None else jnp.zeros(T)
    tp  = target_present if target_present is not None else jnp.zeros(T)
    ms  = int(len(t_arr) * 1.05) + 500
    # target=None → simulate()'s default world-stationary straight-ahead point
    # (correct for VOR: scene + target fixed in the world, head rotates). OKR
    # passes an explicit target that co-rotates with the scene (see _okr_bode).
    return simulate(theta, t_arr,
                    head=km.build_kinematics(t, rot_vel=hv),
                    scene=km.build_kinematics(t, rot_vel=sv),
                    target=target,
                    scene_present_array=sp, target_present_array=tp,
                    max_steps=ms, return_states=True,
                    key=jax.random.PRNGKey(key))


# ── Figure 1: Raphan 1979 Fig.9 replication ───────────────────────────────────

def _bin_average(t, y, bin_s=0.5):
    """Bin a (T,) signal in `bin_s`-second windows; return (centers, means).

    Used to display SPV as discrete points so post-saccadic spikes get
    averaged out into the slow-phase trend, à la Raphan's original Fig.9.
    """
    t = np.asarray(t); y = np.asarray(y)
    t0, t1 = float(t[0]), float(t[-1])
    edges = np.arange(t0, t1 + bin_s, bin_s)
    idx   = np.clip(((t - t0) / bin_s).astype(int), 0, len(edges) - 2)
    means = np.full(len(edges) - 1, np.nan)
    for k in range(len(edges) - 1):
        m = (idx == k)
        if m.any():
            means[k] = np.nanmean(y[m])
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, means


def _raphan(show):
    """3×2 Raphan 1979 Fig.9 replication: SPV (left col) + CUP/INT/SPV (right col)."""
    V_STIM   = 30.0
    ON_DUR   = 30.0
    OFF_DUR  = 50.0
    TOTAL    = ON_DUR + OFF_DUR
    BASELINE = 3.0
    B = int(BASELINE / DT)

    theta = THETA

    # ── A/B: VOR in dark ──────────────────────────────────────────────────────
    _head = km.head_rotation_step(V_STIM, rotate_dur=ON_DUR, coast_dur=OFF_DUR, dt=DT)
    t_rot, hv_3d = _head.t, _head.rot_vel
    t_vor   = np.concatenate([np.arange(1 - B, 1) * DT - DT, np.array(t_rot)])
    hv_full = np.concatenate([np.zeros((B, 3)), np.array(hv_3d)])
    T_vor   = len(t_vor)
    st_vor  = _simulate(theta, jnp.array(t_vor), head_vel=jnp.array(hv_full),
                        scene_present=jnp.zeros(T_vor),
                        target_present=jnp.zeros(T_vor), key=0)
    ev_vor    = np.gradient(np.array(st_vor.plant.left[:, 0]), DT)
    spv_vor_d = -extract_spv_states(st_vor, t_vor)[:, 0]   # negate: compensatory = positive
    cup_vor   = extract_canal(st_vor)
    int_vor   = vs_net(st_vor)[:, 0]                     # x_L−x_R > 0 during rightward VOR
    ni_vor    = ni_net(st_vor)[:, 0]
    eye_vor   = np.array(st_vor.plant.left[:, 0])
    hv_1d     = hv_full[:, 0]

    tau_vor, t_fit_vor, y_fit_vor = fit_tc(
        t_vor, spv_vor_d, t_start=ON_DUR + 1.0, t_end=ON_DUR + OFF_DUR - 5.0,
        label='VOR post-rot TC')

    # ── C/D: OKN + OKAN ───────────────────────────────────────────────────────
    t_stim  = jnp.arange(0.0, TOTAL, DT)
    t_okn   = np.concatenate([np.arange(1 - B, 1) * DT - DT, np.array(t_stim)])
    T_okn   = len(t_okn)
    t_okn_j = jnp.array(t_okn)
    sv = jnp.zeros((T_okn, 3)).at[:, 0].set(
             jnp.where((t_okn_j >= 0.0) & (t_okn_j < ON_DUR), V_STIM, 0.0))
    sp = jnp.where((t_okn_j >= 0.0) & (t_okn_j < ON_DUR), 1.0, 0.0)
    st_okn  = _simulate(theta, t_okn_j, scene_vel=sv,
                        scene_present=sp, target_present=jnp.zeros(T_okn), key=1)
    ev_okn    = np.gradient(np.array(st_okn.plant.left[:, 0]), DT)
    spv_okn_d = extract_spv_states(st_okn, t_okn)[:, 0]    # positive: eye tracks scene
    int_okn   = -vs_net(st_okn)[:, 0]                    # x_L−x_R < 0 → negate for display
    ni_okn    = ni_net(st_okn)[:, 0]
    eye_okn   = np.array(st_okn.plant.left[:, 0])

    tau_okan, t_fit_okan, y_fit_okan = fit_tc(
        t_okn, spv_okn_d, t_start=ON_DUR + 1.0, t_end=TOTAL - 5.0,
        label='OKAN TC')

    # ── E/F: VVOR (rotation in lit stationary scene, stop in dark) ────────────
    # target_present=0: pursuit suppressed — it would fight the VOR by integrating
    # the VOR-induced target slip (the EC cancels u_pursuit but not u_vor, so
    # pursuit sees persistent error and saturates against the VOR).
    # Fast phases are centering saccades (SG uses x_ni centering when target_in_vf=0),
    # matching the Raphan (1979) paradigm.  scene_present=1 enables OKR/OKAN.
    t_vor_j    = jnp.array(t_vor)
    scene_vvor = jnp.where((t_vor_j >= 0.0) & (t_vor_j < ON_DUR), 1.0, 0.0)
    st_vvor   = _simulate(theta, t_vor_j, head_vel=jnp.array(hv_full),
                          scene_present=scene_vvor,
                          target_present=jnp.zeros(T_vor),
                          key=2)
    ev_vvor    = np.gradient(np.array(st_vvor.plant.left[:, 0]), DT)
    spv_vvor_d = -extract_spv_states(st_vvor, t_vor)[:, 0]
    cup_vvor   = extract_canal(st_vvor)
    int_vvor   = vs_net(st_vvor)[:, 0]
    ni_vvor    = ni_net(st_vvor)[:, 0]
    eye_vvor   = np.array(st_vvor.plant.left[:, 0])

    mask_ss   = (t_vor > 10.0) & (t_vor < 25.0)
    vvor_gain = (np.mean(np.abs(spv_vvor_d[mask_ss])) /
                 (np.mean(np.abs(hv_1d[mask_ss])) + 1e-6))
    print(f'  VVOR yaw gain (10–25 s): {vvor_gain:.3f}  (target > 0.85)')

    cup_okn  = extract_canal(st_okn)
    cup_vvor = extract_canal(st_vvor)

    # Bundle the simulations so the cascade figure can reuse them (no re-run).
    _sims = dict(
        theta=theta, ON_DUR=ON_DUR,
        t_vor=np.asarray(t_vor), t_okn=np.asarray(t_okn),
        st_vor=st_vor, st_vvor=st_vvor, st_okn=st_okn,
        hv_yaw=np.asarray(hv_1d),                  # head-vel yaw (VOR / VVOR input)
        scene_yaw_okn=np.asarray(sv)[:, 0],        # scene-vel yaw (OKR input)
    )

    # ── Plotting: 3×2 layout matching Raphan Fig.9 ───────────────────────────
    # Left col (A, C, E): SPV only.  Right col (B, D, F): SPV + Cupula + Integrator.
    fig, axes = plt.subplots(3, 2, figsize=(12, 11))
    fig.suptitle(
        'Raphan, Matsuo & Cohen (1979) Fig. 9 — Replication\n'
        'Left: slow-phase velocity  |  Right: S.P.VEL + Cupula + Integrator (VS)',
        fontsize=10, fontweight='bold')
    xlim = (-BASELINE, TOTAL)
    vl   = dict(color='k', lw=0.8, ls='--', alpha=0.5)

    def _lbl(ax, letter):
        ax.text(0.02, 0.92, letter, transform=ax.transAxes,
                fontsize=12, fontweight='bold', va='top')

    # SPV traces are shown as 0.5-s binned scatter so post-saccadic spikes
    # average into the slow-phase trend (matches Raphan's original Fig.9
    # presentation: discrete data points, not a continuous trace).
    spv_kw = dict(s=18, marker='o', edgecolors='none', color=utils.C['spv'])

    # A — VOR dark: SPV only
    ax = axes[0, 0]
    ax.plot(t_vor, -hv_1d,    color=utils.C['head'], lw=1.0, ls=':', alpha=0.7, label='−head vel')
    t_b, y_b = _bin_average(t_vor, spv_vor_d, 0.5)
    ax.scatter(t_b, y_b, label='S.P.VEL', **spv_kw)
    if tau_vor is not None:
        ax.plot(t_fit_vor, y_fit_vor, color='tomato', lw=1.5, ls='--', label=f'fit τ={tau_vor:.1f}s')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title(f'Step rotation {V_STIM:.0f} deg/s — darkness')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'A')

    # B — VOR dark: SPV + Cupula + Integrator
    ax = axes[0, 1]
    t_b, y_b = _bin_average(t_vor, spv_vor_d, 0.5)
    ax.scatter(t_b, y_b, label='S.P.VEL', **spv_kw)
    ax.plot(t_vor, cup_vor,   color=utils.C['canal'], lw=1.2, ls='--', label='Cupula')
    ax.plot(t_vor, int_vor,   color=utils.C['vs'],    lw=1.2, ls='-.',  label='Integrator (VS)')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title('VOR dark: S.P.VEL + Cupula + Integrator')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'B')

    # C — OKN + OKAN: SPV only
    ax = axes[1, 0]
    scene_ref = np.where((t_okn >= 0.0) & (t_okn < ON_DUR), V_STIM, 0.0)
    ax.plot(t_okn, scene_ref,  color=utils.C['scene'], lw=1.0, ls=':', alpha=0.7, label='scene vel')
    t_b, y_b = _bin_average(t_okn, spv_okn_d, 0.5)
    ax.scatter(t_b, y_b, label='S.P.VEL', **spv_kw)
    if tau_okan is not None:
        ax.plot(t_fit_okan, y_fit_okan, color='tomato', lw=1.5, ls='--', label=f'OKAN τ={tau_okan:.1f}s')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title(f'Surround velocity {V_STIM:.0f} deg/s — OKN then OKAN')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(-BASELINE, TOTAL); _lbl(ax, 'C')

    # D — OKN + OKAN: SPV + Cupula (≈0) + Integrator
    ax = axes[1, 1]
    t_b, y_b = _bin_average(t_okn, spv_okn_d, 0.5)
    ax.scatter(t_b, y_b, label='S.P.VEL', **spv_kw)
    ax.plot(t_okn, cup_okn,   color=utils.C['canal'],  lw=1.2, ls='--', label='Cupula')
    ax.plot(t_okn, int_okn,   color=utils.C['vs'],     lw=1.2, ls='-.',  label='Integrator (VS)')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_title('OKN: S.P.VEL + Cupula + Integrator')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(-BASELINE, TOTAL); _lbl(ax, 'D')

    # E — VVOR: SPV only
    ax = axes[2, 0]
    ax.plot(t_vor, -hv_1d,    color=utils.C['head'], lw=1.0, ls=':', alpha=0.7, label='−head vel')
    t_b, y_b = _bin_average(t_vor, spv_vvor_d, 0.5)
    ax.scatter(t_b, y_b, label=f'S.P.VEL (gain={vvor_gain:.2f})', **spv_kw)
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_xlabel('Time (s)')
    ax.set_title(f'Rotation {V_STIM:.0f} deg/s in light → stop in dark')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'E')

    # F — VVOR: SPV + Cupula + Integrator
    ax = axes[2, 1]
    t_b, y_b = _bin_average(t_vor, spv_vvor_d, 0.5)
    ax.scatter(t_b, y_b, label='S.P.VEL', **spv_kw)
    ax.plot(t_vor, cup_vvor,   color=utils.C['canal'],  lw=1.2, ls='--', label='Cupula')
    ax.plot(t_vor, int_vvor,   color=utils.C['vs'],     lw=1.2, ls='-.',  label='Integrator (VS)')
    ax.axvline(0.0, **vl); ax.axvline(ON_DUR, **vl)
    ax.set_ylabel('deg/s'); ax.set_xlabel('Time (s)')
    ax.set_title('VVOR: S.P.VEL + Cupula + Integrator')
    ax.legend(fontsize=7); ax_fmt(ax); ax.set_xlim(*xlim); _lbl(ax, 'F')

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'vor_raphan_fig9', show=show, params=THETA,
                              conditions='Dark/lit conditions per panel — head velocity step + scene step (Raphan 1979 Fig.9 protocol)')

    # ── Quantitative metrics (measured from states, not the figure) ──────────
    # OKN steady-state gain: mean SPV over the last 10 s of the scene-on phase,
    # divided by scene velocity. Eye should track the surround (gain ≈ 1).
    okn_ss   = (t_okn >= ON_DUR - 10.0) & (t_okn < ON_DUR - 1.0)
    okn_gain = float(np.mean(spv_okn_d[okn_ss]) / V_STIM)

    metrics = [
        Metric('vor_okr_vvor_gain', vvor_gain, 
               lo=0.85, hi=1.10, golden_tol=0.08, units='',
               cite='Raphan, Matsuo & Cohen (1979)',
               desc='VVOR slow-phase gain during rotation in light (10–25 s)'),
        Metric('vor_okr_postrot_tc',
               float('nan') if tau_vor is None else float(tau_vor), 
               lo=10.0, hi=30.0, golden_tol=0.15, units='s',
               cite='Raphan, Matsuo & Cohen (1979)',
               desc='VOR post-rotatory SPV decay time constant (VS-extended)'),
        Metric('vor_okr_okan_tc',
               float('nan') if tau_okan is None else float(tau_okan), 
               lo=10.0, hi=30.0, golden_tol=0.15, units='s',
               cite='Raphan, Matsuo & Cohen (1979)',
               desc='OKAN SPV decay time constant after scene off (~tau_vs)'),
        Metric('vor_okr_okn_ss_gain', okn_gain, 
               lo=0.75, hi=1.10, golden_tol=0.08, units='',
               cite='Raphan, Matsuo & Cohen (1979)',
               desc='OKN steady-state SPV gain (last 10 s of scene-on)'),
    ]

    fig = utils.fig_meta(path, rp,
        title='Raphan 1979 Fig. 9 Replication',
        description='Panels A–F matching Raphan et al. (1979) Fig.9. Left col: SPV only. '
                    'Right col: CUP (canal estimate), INT (velocity storage), SPV overlaid. '
                    'A/B: VOR in dark. C/D: OKN+OKAN. E/F: VVOR (light→dark).',
        expected='A: post-rot SPV TC 10–30 s. C: OKN gain~1, OKAN TC~20 s. '
                 'E: VVOR gain>0.85 during rotation; post-rot TC similar to A. '
                 'B/D/F: INT follows SPV; CUP decays at canal TC (~5 s).',
        citation='Raphan, Matsuo & Cohen (1979) Exp Brain Res 35:229–248',
        fig_type='behavior')
    fig['metrics'] = metrics
    return fig, _sims


# ── Figure 2: OKN nystagmus zoom ──────────────────────────────────────────────

def _okn_zoom(show):
    """Zoomed OKN trace showing sawtooth nystagmus in first 15 s."""
    ON_DUR = 15.0
    t_arr  = jnp.arange(0.0, ON_DUR, DT)
    T      = len(t_arr)
    t_np   = np.array(t_arr)

    sv = jnp.zeros((T, 3)).at[:, 0].set(30.0)
    sp = jnp.ones(T)

    st     = _simulate(THETA, t_arr, scene_vel=sv,
                       scene_present=sp, target_present=jnp.zeros(T), key=3)
    eye    = np.array(st.plant.left[:, 0])
    ev  = np.gradient(eye, DT)
    spv = extract_spv_states(st, t_np)[:, 0]

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    fig.suptitle('OKN Nystagmus — Sawtooth Waveform Zoom (30 deg/s scene)', fontsize=11)

    axes[0].plot(t_np, eye, color=utils.C['eye'], lw=0.8, label='eye position')
    axes[0].set_ylabel('Eye position (deg)')
    axes[0].set_title('Eye Position — sawtooth fast phases visible')
    axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.25)

    axes[1].plot(t_np, ev,  color='steelblue', lw=0.5, alpha=0.4, label='eye vel (raw)')
    axes[1].plot(t_np, spv, color=utils.C['spv'], lw=2.0, label='SPV (fast phases removed)')
    axes[1].axhline(30.0, color=utils.C['scene'], lw=1.0, ls=':', alpha=0.7,
                    label='scene vel = 30 deg/s')
    axes[1].set_ylim(-80, 80)
    axes[1].set_ylabel('Eye velocity (deg/s)'); axes[1].set_xlabel('Time (s)')
    axes[1].set_title('Eye Velocity — slow phases (OKN) + fast phase resets')
    axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.25)

    fig.tight_layout()
    path, rp = utils.save_fig(fig, 'okn_nystagmus_zoom', show=show, params=THETA,
                              conditions='Lit, full-field scene velocity step (OKN sawtooth + post-OKAN)')
    return utils.fig_meta(path, rp,
        title='OKN Nystagmus Zoom',
        description='First 15 s of OKN at 30 deg/s. Top: eye position showing sawtooth waveform. '
                    'Bottom: raw eye velocity (fast phases visible) and SPV (fast phases removed).',
        expected='Clear sawtooth; SPV ≈ 30 deg/s during slow phases; fast phases reset periodically.',
        citation='Raphan et al. (1979)',
        fig_type='behavior')


# ── Figure 3: VOR/OKR signal cascade ──────────────────────────────────────────

def _cascade(sims, show):
    """Internal signal cascade for VOR-dark / VVOR / OKR — 3 columns, REUSING the
    Raphan Fig.9 simulations (no separate sim run). Rows: input → canal afferent /
    retinal slip → velocity storage → neural integrator → eye velocity, plus a
    fast-phase zoom (eye position + velocity)."""
    from scipy.special import softplus as _spf
    theta  = sims['theta']
    t_vor  = sims['t_vor'];  t_okn = sims['t_okn']
    ON     = sims['ON_DUR']
    hv_yaw = sims['hv_yaw'];  sv_yaw = sims['scene_yaw_okn']

    def _canal_u(st):
        x2   = np.array(st.sensory.canal.x2)
        k, f = float(_SOFTNESS), float(FLOOR)
        y_c  = -f + _spf(k * (x2 + f)) / k + _spf(k * (x2 - f)) / k
        return (np.array(PINV_SENS) @ y_c.T).T[:, 0]
    def _slip(st):
        return np.array(read_brain_acts(st, theta).pc.scene_angular_vel)[:, 0]

    # Per column: (title, state, time, head-vel input, scene-vel input)
    cols = [
        ('VOR (dark)',                     sims['st_vor'],  t_vor, hv_yaw, None),
        ('VVOR (lit stationary scene)',    sims['st_vvor'], t_vor, hv_yaw, None),
        ('OKR (scene motion, head still)', sims['st_okn'],  t_okn, None,  sv_yaw),
    ]
    ZOOM_T0, ZOOM_T1 = 10.0, 15.0   # window during the stimulus-on phase (fast phases)

    fig, axes = plt.subplots(7, 3, figsize=(16, 14),
                             gridspec_kw={'hspace': 0.4, 'wspace': 0.30,
                                          'height_ratios': [1, 1, 1, 1, 1, 0.8, 0.8]})
    fig.suptitle('VOR / VVOR / OKR — Step Response (internal signals from the Raphan '
                 'Fig.9 simulations; the Bode figures cover the sinusoidal response)', fontsize=11)

    row_labels = ['Input (deg/s)',
                  'Canal afferent / retinal slip (deg/s)',
                  'Velocity storage x_VS (deg/s)',
                  'Neural integrator x_NI (deg)',
                  'Eye velocity (deg/s)',
                  f'Eye pos zoom [{ZOOM_T0:.0f}–{ZOOM_T1:.0f} s] (deg)',
                  f'Eye vel zoom [{ZOOM_T0:.0f}–{ZOOM_T1:.0f} s] (deg/s)']
    for r, lbl in enumerate(row_labels):
        axes[r, 0].set_ylabel(lbl, fontsize=8)
    vl = dict(color='k', lw=0.8, ls='--', alpha=0.4)

    for c, (title, st, t, hin, sin) in enumerate(cols):
        axes[0, c].set_title(title, fontsize=9)
        u_can = _canal_u(st);  slip = _slip(st)
        x_vs  = vs_net(st)[:, 0];  x_ni = ni_net(st)[:, 0]
        eye   = np.array(st.plant.left[:, 0]);  ev = np.gradient(eye, DT)
        spv   = extract_spv_states(st, t)[:, 0]

        if hin is not None:
            axes[0, c].plot(t, hin,  color=utils.C['head'], lw=1.4, label='head vel')
            axes[0, c].plot(t, -hin, color=utils.C['dark'], lw=0.9, ls=':', alpha=0.6, label='−head (ideal eye)')
        if sin is not None:
            axes[0, c].plot(t, sin,  color=utils.C['scene'], lw=1.4, label='scene vel')
        axes[1, c].plot(t, u_can, color=utils.C['canal'], lw=1.3, label='canal → u_canal')
        axes[1, c].plot(t, slip,  color='darkorange', lw=1.0, alpha=0.85, label='slip_delayed')
        axes[2, c].plot(t, x_vs,  color=utils.C['vs'], lw=1.3, label='x_VS net')
        axes[3, c].plot(t, x_ni,  color=utils.C['ni'], lw=1.3, label='x_NI net')
        axes[4, c].plot(t, ev,    color=utils.C['eye'], lw=0.9, alpha=0.5, label='eye vel (raw)')
        axes[4, c].plot(t, spv,   color=utils.C['spv'], lw=1.6, label='SPV')
        axes[4, c].set_xlabel('Time (s)', fontsize=8)
        for r in range(5):
            axes[r, c].axvline(ON, **vl); ax_fmt(axes[r, c]); axes[r, c].legend(fontsize=6.5)

        zm = (t >= ZOOM_T0) & (t <= ZOOM_T1)
        axes[5, c].plot(t[zm], eye[zm], color=utils.C['eye'], lw=1.0)
        axes[5, c].set_xlim(ZOOM_T0, ZOOM_T1); ax_fmt(axes[5, c]); axes[5, c].set_xlabel('Time (s)', fontsize=8)
        axes[6, c].plot(t[zm], ev[zm],  color=utils.C['eye'], lw=0.8, alpha=0.5, label='eye vel')
        axes[6, c].plot(t[zm], spv[zm], color=utils.C['spv'], lw=1.6, label='SPV')
        axes[6, c].set_xlim(ZOOM_T0, ZOOM_T1); axes[6, c].set_ylim(-100, 100)
        ax_fmt(axes[6, c]); axes[6, c].legend(fontsize=6.5); axes[6, c].set_xlabel('Time (s)', fontsize=8)

    fig.tight_layout(pad=0.4)
    path, rp = utils.save_fig(fig, 'vor_okr_cascade', show=show, params=theta,
        conditions='VOR / VVOR / OKR internal cascade — reuses the Raphan Fig.9 simulations (3 columns)')
    # Visual-only debug cascade (no metrics; the Raphan figure owns the gains/TCs).
    return utils.fig_meta(path, rp,
        title='VOR / VVOR / OKR — Step Response',
        description='Step / transient response — internal signals from the SAME simulations as the '
                    'Raphan Fig.9 figure (no separate run); the Bode figures below cover the '
                    'sinusoidal (frequency) response. Columns: VOR in dark, VVOR (rotation in a lit '
                    'stationary scene), OKR (scene motion, head still). Rows: input → canal afferent / '
                    'retinal slip → velocity storage → neural integrator → eye velocity, plus a '
                    'fast-phase zoom (position + velocity). Vertical dashed line = stimulus off.',
        expected='VOR: canal drives VS/NI, eye counter-rotates. VVOR: canal + OKR slip both feed '
                 'VS, gaze stays stable. OKR: slip drives VS/NI, eye follows scene.',
        citation='Raphan, Matsuo & Cohen (1979); Robinson (1975)',
        fig_type='cascade')


# ── Figures: VOR / OKR frequency response (Bode, noiseless) ──────────────────

# Extends well past the canal/plant corner so the VOR high-frequency rolloff +
# phase lead are visible (the VOR is broadband, ~flat to several Hz). dt=0.001 s
# resolves up to 20 Hz with ≥50 samples/cycle for the sinusoid fit.
_BODE_FREQS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0,
                        5.0, 7.0, 10.0, 15.0, 20.0])

# VOR Bode amplitude / settle policy.
#  - Cap PEAK POSITION at _VOR_BODE_POS_MAX so the earth-fixed target never leaves
#    the foveatable range (~±25°): velocity = min(VMAX, POS_MAX·2πf) (bode.capped_velocity_amp).
#    Without this a constant-velocity sweep puts the target at ±95° at 0.05 Hz and
#    the light+target point collapses into out-of-range nystagmus (a pure artifact).
#  - Run each point past the velocity-storage transient (~4·TC_VS) at low f so the
#    steady-state fit isn't biased by the still-charging storage.
_VOR_BODE_VMAX     = 30.0      # deg/s — peak velocity above the position-cap knee
_VOR_BODE_POS_MAX  = 20.0      # deg   — peak position excursion cap (< ~±25° fixation limit)
_VOR_TC_VS         = 35.0      # s     — velocity-storage settle (post-rotatory TC scale)
_VOR_SETTLE_FRAC   = 0.5       # fit uses the last half of each record
# Opt-in low-frequency extension (down to 0.005 Hz) that reveals the dark high-pass
# rolloff + phase lead. OFF in the suite (long records at low f are slow); enable
# via _vor_bode(low_f=True) for the illustrative figure.
_VOR_BODE_FREQS_LOWF = np.array([0.005, 0.01, 0.02])


def _vor_bode(show, low_f=False):
    """VOR frequency response — sinusoidal head yaw, NOISELESS. Eye velocity
    (SPV) ÷ head velocity (compensatory gain ≈ 1). Conditions: dark, light, light+target.

    Peak position is capped at _VOR_BODE_POS_MAX so the earth-fixed target never
    leaves range (no out-of-range artifact). `low_f=True` extends the sweep down
    to 0.005 Hz to show the dark high-pass rolloff (slow — long low-f records)."""
    AMP, N_CYC = _VOR_BODE_VMAX, 4
    CONDS = [('dark', 0.0, 0.0, '#444444'),
             ('light', 1.0, 0.0, '#1b7837'),
             ('light+target', 1.0, 1.0, '#c0392b')]
    freqs = (np.concatenate([_VOR_BODE_FREQS_LOWF, _BODE_FREQS]) if low_f else _BODE_FREQS)

    def make(scene_p, target_p):
        def run_fn(f):
            w = 2 * np.pi * f
            V = bode.capped_velocity_amp(f, AMP, _VOR_BODE_POS_MAX)   # keep target in range
            period = 1.0 / f
            # Fit uses the last (1 − settle_frac); size the record so that window
            # holds ≥ N_CYC cycles AND (at low f) clears the ~4·TC_VS storage transient.
            T_end = (N_CYC * period) / (1.0 - _VOR_SETTLE_FRAC)
            # Full velocity-storage settle (~4·TC) is only paid for in the opt-in
            # low_f sweep — the suite would otherwise spend minutes on the 0.05 Hz
            # point. In the suite the cap already removes the gross artifact; the
            # mid-band gain is ~1 through the transient, so a 4-cycle record suffices.
            if low_f and period > 0.3 * _VOR_TC_VS:
                T_end = max(T_end, 4.0 * _VOR_TC_VS / _VOR_SETTLE_FRAC)
            # Floor: high-f records must clear the plant/processing transient (~0.5 s)
            # + the SPV-mask window before the fit half — else 5–20 Hz points notch out.
            T_end = float(np.clip(T_end, 10.0, 1600.0))
            t = np.arange(0.0, T_end, DT); Tn = len(t)
            hv = np.zeros((Tn, 3)); hv[:, 0] = V * np.sin(w * t)     # oscillate throughout (starts at 0)
            st = _simulate(THETA_NOISELESS, jnp.array(t), head_vel=jnp.array(hv),
                           scene_present=jnp.full(Tn, scene_p),
                           target_present=jnp.full(Tn, target_p), key=0)
            spv = extract_spv_states(st, t, eye='version')[:, 0]
            return t, hv[:, 0], -spv      # −eye so compensatory is in phase with head
        return run_fn

    series = []
    for lab, sp, tp, col in CONDS:
        _, g, p = bode.bode_sweep(make(sp, tp), freqs, settle_frac=_VOR_SETTLE_FRAC)
        series.append(dict(label=lab, gains=g, phases=p, color=col))
    _ef = np.logspace(np.log10(freqs.min()), np.log10(20.0), 140)
    fig, out = bode.make_bode_multi(freqs, series,
        'VOR — Frequency Response (body yaw rotation, noiseless)',
        ref_hz=0.5, gain_label='Gain (eye vel ÷ head vel)',
        expected=bode.expected_bode('flat', g0=1.0, f=_ef,
            label='expected: ideal VOR ≈ unity, broadband (Cohen, Matsuo & Raphan 1977)'))
    path, rp = utils.save_fig(fig, 'vor_bode', show=show, params=THETA_NOISELESS,
        conditions=f'Dark / light / light+target, NOISELESS — sinusoidal head yaw '
                   f'{freqs.min():.3g}–20 Hz (≤30 deg/s, peak position capped at '
                   f'{_VOR_BODE_POS_MAX:.0f}°)')
    metrics = []
    for lab, *_ in CONDS:
        m = out[lab]; k = lab.replace('+', '_').replace(' ', '')
        metrics.append(Metric(f'vor_bode_gain_max_{k}', float(m['gain_max']),
            lo=0.4, hi=1.2, golden_tol=0.1, units='',
            cite='Cohen, Matsuo & Raphan (1977)', desc=f'VOR peak gain — {lab}'))
        if m['fc_lo'] is not None:
            metrics.append(Metric(f'vor_bode_fc_lo_{k}', float(m['fc_lo']),
                lo=None, hi=None, golden_tol=0.25, units='Hz',
                cite='Cohen, Matsuo & Raphan (1977)',
                desc=f'VOR low-side −3 dB corner (canal high-pass) — {lab}'))
        if m['fc_hi'] is not None:
            metrics.append(Metric(f'vor_bode_fc_hi_{k}', float(m['fc_hi']),
                lo=None, hi=None, golden_tol=0.25, units='Hz',
                cite='Cohen, Matsuo & Raphan (1977)', desc=f'VOR high-side −3 dB cutoff — {lab}'))
    fm = utils.fig_meta(path, rp,
        title='VOR — Bode (body rotation)',
        description='Sinusoidal head yaw sweep (≤30 deg/s, peak position capped at '
                    f'{_VOR_BODE_POS_MAX:.0f}° so the target stays in range), NOISELESS. '
                    'Eye velocity (SPV) ÷ head velocity in dark, light, light+target.',
        expected='Light gain ≈ 1 across frequency (VVOR); dark gain rolls off (high-pass) '
                 'below ~0.01 Hz — visible only with low_f=True.',
        citation='Cohen, Matsuo & Raphan (1977); Raphan et al. (1979)',
        fig_type='behavior')
    fm['metrics'] = metrics
    return fm


def _okr_bode(show):
    """OKR frequency response — sinusoidal full-field scene motion (head still),
    NOISELESS. Eye velocity (SPV) ÷ scene velocity. Conditions: scene, scene+target."""
    AMP, N_CYC, SETTLE = 20.0, 5, 2.0
    POS_MAX = 20.0   # deg — cap peak scene/target excursion (keep co-rotating target in range)
    CONDS = [('scene', 0.0, '#1b7837'), ('scene+target', 1.0, '#c0392b')]

    def make(target_p):
        def run_fn(f):
            # Floor the oscillation duration (≥5 s) so the settle_frac analysis
            # window always lands inside the oscillation, not the pre-oscillation
            # dead time — otherwise high-f points (short records) fit garbage.
            T_end = min(SETTLE + max(N_CYC / f, 5.0), 50.0)
            t = np.arange(0.0, T_end, DT); Tn = len(t); w = 2 * np.pi * f
            on = t >= SETTLE
            # Cap peak scene/target excursion (= V/w) so the co-rotating target
            # stays foveatable — else at 0.05 Hz it swings to 64° and the
            # scene+target point dips below pure scene (out-of-range artifact).
            V  = bode.capped_velocity_amp(f, AMP, POS_MAX)
            sv = np.zeros((Tn, 3)); sv[:, 0] = np.where(on, V * np.sin(w * (t - SETTLE)), 0.0)
            # OKR: body still; the foveal target co-rotates WITH the full-field
            # scene about the head's yaw axis (same angular velocity). With a
            # target present, pursuit then REINFORCES OKN — instead of a
            # world-fixed target anchoring fixation and suppressing it.
            target = km.build_target(t, vel_yaw_deg_s=sv[:, 0], distance_m=1.0)
            st = _simulate(THETA_NOISELESS, jnp.array(t), scene_vel=jnp.array(sv),
                           target=target,
                           scene_present=jnp.ones(Tn),
                           target_present=jnp.full(Tn, target_p), key=1)
            spv = extract_spv_states(st, t, eye='version')[:, 0]
            return t, sv[:, 0], spv       # eye follows scene (same sign)
        return run_fn

    series = []
    for lab, tp, col in CONDS:
        _, g, p = bode.bode_sweep(make(tp), _BODE_FREQS, settle_frac=0.45)
        series.append(dict(label=lab, gains=g, phases=p, color=col))
    _ef = np.logspace(np.log10(0.05), np.log10(20.0), 140)
    fig, out = bode.make_bode_multi(_BODE_FREQS, series,
        'OKR — Frequency Response (full-field scene motion, noiseless)',
        ref_hz=0.5, gain_label='Gain (eye vel ÷ scene vel)',
        expected=bode.expected_bode('lowpass', fc=1.0, g0=1.0, f=_ef,
            label='expected: OKR low-pass, ~1 Hz cutoff (Cohen & Raphan; model-expected)'))
    path, rp = utils.save_fig(fig, 'okr_bode', show=show, params=THETA_NOISELESS,
        conditions='Scene / scene+target, NOISELESS — sinusoidal scene velocity 0.05–20 Hz (20 deg/s)')
    metrics = []
    for lab, tp, col in CONDS:
        m = out[lab]; k = lab.replace('+', '_').replace(' ', '')
        metrics.append(Metric(f'okr_bode_gain_max_{k}', float(m['gain_max']),
            lo=0.4, hi=1.05, golden_tol=0.1, units='',
            cite='Cohen, Matsuo & Raphan (1977)', desc=f'OKR peak gain — {lab}'))
        if m['fc_hi'] is not None:
            metrics.append(Metric(f'okr_bode_fc_hi_{k}', float(m['fc_hi']),
                lo=0.1, hi=None, golden_tol=0.25, units='Hz',
                cite='Cohen, Matsuo & Raphan (1977)', desc=f'OKR −3 dB bandwidth — {lab}'))
    fm = utils.fig_meta(path, rp,
        title='OKR — Bode (full-field scene motion)',
        description='Sinusoidal scene-velocity sweep (0.05–20 Hz, 20 deg/s), NOISELESS. '
                    'Eye velocity (SPV) ÷ scene velocity for scene-only and scene+target. '
                    'OKR is low-pass — gain rolls off above ~0.5 Hz; a stationary target suppresses it.',
        expected='Gain ≈ 1 at low f, rolls off above ~0.5 Hz.',
        citation='Cohen, Matsuo & Raphan (1977) J Neurophysiol',
        fig_type='behavior')
    fm['metrics'] = metrics
    return fm


# ── Section entry point ────────────────────────────────────────────────────────

SECTION = dict(
    id='vor_okr', title='2. VOR / OKR',
    description='Vestibulo-ocular reflex and optokinetic response. '
                'Replicates Raphan et al. (1979) Fig.9: VOR in dark, OKN+OKAN, and VVOR. '
                'Slow-phase velocity shown as 0.5-s binned scatter (post-saccadic spikes averaged into the '
                'slow-phase trend). Tests post-rotatory and OKAN time constants, VVOR gain, and nystagmus waveform. '
                'Scene slip → VS goes through K_vor_direct·(saccadically-gated slip) + K_cereb_okr·(cerebellar EC '
                'correction); nodulus-uvula gravity dumping (K_cereb_nu) sets the velocity-storage TC.',
)


def run(show=False):
    print('\n=== VOR / OKR ===')
    figs = []
    print('  1/4  Raphan Fig.9 replication …')
    fm_raphan, sims = _raphan(show)         # also yields the shared simulations
    figs.append(fm_raphan)
    print('  2/4  step response (reuses Raphan sims) …')
    figs.append(_cascade(sims, show))       # no separate simulation — sits below Raphan
    print('  3/4  VOR Bode (body rotation) …')
    figs.append(_vor_bode(show))
    print('  4/4  OKR Bode (scene motion) …')
    figs.append(_okr_bode(show))
    return figs


if __name__ == '__main__':
    run(show=SHOW)
