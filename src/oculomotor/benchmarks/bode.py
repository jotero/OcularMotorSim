"""Shared Bode-sweep helper for closed-loop frequency-response benchmarks.

Every closed-loop oculomotor subsystem (VOR, OKR, pursuit, vergence,
accommodation, accommodative-vergence, tVOR) is characterized the same way:
drive it with a sinusoid at a set of frequencies, fit the steady-state output
sinusoid, and report **gain** (output/input amplitude) and **phase** (deg,
negative = lag) versus frequency. Each bench supplies a tiny per-frequency
``run_fn(f) -> (t, drive, output)``; the sweep + fit + metrics live here so
every system's Bode plot is built and measured identically.

Phase convention: output modelled as ``amp · sin(2πf·t + φ)``; reported phase is
``φ_out − φ_in`` wrapped to (−180°, 180°]. Negative = output lags input.
"""

import numpy as np


def fit_sinusoid(t, y, f):
    """Least-squares fit ``y ≈ A·sin(2πf t) + B·cos(2πf t) + C``.

    Returns (amplitude, phase_rad, offset), where amplitude·sin(2πf t + phase)
    is the fitted oscillation (amplitude = hypot(A, B), phase = atan2(B, A)).
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    w = 2.0 * np.pi * f
    M = np.stack([np.sin(w * t), np.cos(w * t), np.ones_like(t)], axis=1)
    (A, B, C), *_ = np.linalg.lstsq(M, y, rcond=None)
    return float(np.hypot(A, B)), float(np.arctan2(B, A)), float(C)


def bode_point(t, drive, output, f, settle_frac=0.5, output_mask=None):
    """Gain and phase (deg) of `output` relative to `drive` at frequency `f`.

    Only the steady-state tail is used (the last ``1 − settle_frac`` of the
    record), so the initial transient doesn't bias the fit.

    GAP-AWARE: if `output_mask` (a boolean (T,) array of VALID output samples) is
    given, the output sinusoid is fit only over those samples. This is for SPV
    outputs, where fast-phase epochs are masked: fitting the *valid* slow-phase
    samples directly avoids the amplitude loss of interpolating a straight chord
    across blanked high-frequency cycles (which fakes a high-f rolloff). The
    least-squares sinusoid fit handles non-contiguous samples fine as long as
    they span the cycle. The drive is always clean → fit over the full window.
    """
    t = np.asarray(t, dtype=float)
    t0, t1 = t[0], t[-1]
    win = t >= (t0 + settle_frac * (t1 - t0))
    amp_in,  ph_in,  _ = fit_sinusoid(t[win], np.asarray(drive)[win],  f)
    ow = win if output_mask is None else (win & np.asarray(output_mask, dtype=bool))
    if ow.sum() < 4:            # too few valid samples for a 3-param fit → fall back
        ow = win
    amp_out, ph_out, _ = fit_sinusoid(t[ow], np.asarray(output)[ow], f)
    gain = amp_out / amp_in if amp_in > 1e-9 else float('nan')
    phase = np.degrees(((ph_out - ph_in + np.pi) % (2.0 * np.pi)) - np.pi)
    return gain, phase


def bode_sweep(run_fn, freqs, settle_frac=0.5):
    """Sweep `freqs`, calling ``run_fn(f)`` at each.

    ``run_fn(f)`` returns either ``(t, drive, output)`` or
    ``(t, drive, output, output_mask)``, where ``output_mask`` flags the VALID
    output samples for a gap-aware fit (e.g. the slow-phase mask of an SPV
    output).  The drive is always treated as clean.

    Returns (freqs, gains, phases_deg) as float arrays.
    """
    gains, phases = [], []
    for f in freqs:
        res = run_fn(f)
        t, drive, output = res[0], res[1], res[2]
        omask = res[3] if len(res) > 3 else None
        g, p = bode_point(t, drive, output, f, settle_frac, output_mask=omask)
        gains.append(g)
        phases.append(p)
    return np.asarray(freqs, float), np.asarray(gains, float), np.asarray(phases, float)


def capped_velocity_amp(f, v_max, pos_max):
    """Peak velocity for a sinusoidal Bode point whose peak POSITION excursion is
    capped at ``pos_max`` (deg).

    For a sinusoid, position amplitude = v/(2πf), so holding velocity constant
    makes the excursion blow up as 1/f at low frequency — which drives an
    earth-fixed fixation target far out of the oculomotor / visual range (e.g.
    ±95° at 0.05 Hz, 30 deg/s) and corrupts the point (the target becomes
    unfoveatable; pursuit winds up; the response degrades into nystagmus).
    Capping position keeps every point physical::

        v = min(v_max, pos_max · 2πf)

    Above the knee ``f_knee = v_max/(2π·pos_max)`` it is the constant ``v_max``;
    below it, velocity scales down with f so the excursion stays ≤ ``pos_max``.
    Use this in every closed-loop Bode ``run_fn`` so no sweep leaves range.
    """
    return float(min(v_max, pos_max * 2.0 * np.pi * f))


def _interp_crossing(freqs, gains, thresh):
    """Highest frequency at which `gains` crosses `thresh` (log-f linear interp).

    Used for the −3 dB bandwidth. Returns NaN if no crossing in range.
    """
    freqs = np.asarray(freqs, float)
    gains = np.asarray(gains, float)
    lf = np.log10(freqs)
    for i in range(len(freqs) - 1):
        g0, g1 = gains[i], gains[i + 1]
        if (g0 - thresh) * (g1 - thresh) <= 0 and g0 != g1:
            frac = (thresh - g0) / (g1 - g0)
            return float(10.0 ** (lf[i] + frac * (lf[i + 1] - lf[i])))
    return float('nan')


def bode_metrics(freqs, gains, phases=None, ref_hz=None, highpass=False):
    """Unified scalar metrics from a Bode sweep — one scheme for every system:

        gain_max : peak gain across the sweep
        fc_lo    : low-side −3 dB cutoff — frequency where the gain rises through
                   gain_max/√2 below the peak (high-pass character; None when the
                   gain never falls on the low side — i.e. low-pass / flat)
        fc_hi    : high-side −3 dB cutoff / bandwidth — frequency where the gain
                   falls through gain_max/√2 above the peak (low-pass character;
                   None when the gain never falls on the high side — high-pass / flat)

    `phases`, `ref_hz`, `highpass` are accepted for call compatibility but no
    longer yield metrics (the phase curve stays on the plot only; the peak is
    auto-located, so `highpass` no longer matters).
    """
    freqs = np.asarray(freqs, float)
    gains = np.asarray(gains, float)
    if not len(freqs):
        return dict(gain_max=float('nan'), fc_lo=None, fc_hi=None)
    i_peak   = int(np.argmax(gains))
    gain_max = float(gains[i_peak])
    thresh   = gain_max / np.sqrt(2.0)
    fc_hi = _interp_crossing(freqs[i_peak:],           gains[i_peak:],           thresh)
    fc_lo = _interp_crossing(freqs[:i_peak + 1][::-1], gains[:i_peak + 1][::-1], thresh)
    _clean = lambda x: None if (x is None or x != x) else float(x)   # NaN/None → None
    return dict(gain_max=gain_max, fc_lo=_clean(fc_lo), fc_hi=_clean(fc_hi))


# Standard frequency grid (Hz) for closed-loop sweeps — log-spaced 0.05–5 Hz.
STD_FREQS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])


def expected_bode(kind, fc=1.0, g0=1.0, g_inf=1.0, label='expected', f=None):
    """Hypothetical / textbook expected Bode curve on a fine grid — returns a
    dict(f=, gain=, phase=, label=) ready to pass as `expected=` to the make_bode
    helpers (drawn as a thin no-dot reference line).

    kind:
      'lowpass'  — first-order LP, DC gain g0, corner fc:  g0/√(1+(f/fc)²)
      'highpass' — first-order HP, HF gain g_inf, corner fc: g_inf·(f/fc)/√(1+(f/fc)²)
      'flat'     — constant gain g0 (e.g. VOR ≈ unity, broadband)
    """
    if f is None:
        f = np.logspace(np.log10(0.05), np.log10(5.0), 120)
    f = np.asarray(f, float)
    r = f / fc
    if kind == 'lowpass':
        gain  = g0 / np.sqrt(1.0 + r ** 2)
        phase = -np.degrees(np.arctan(r))
    elif kind == 'highpass':
        gain  = g_inf * r / np.sqrt(1.0 + r ** 2)
        phase = np.degrees(np.arctan(1.0 / np.maximum(r, 1e-9)))
    elif kind == 'flat':
        gain  = np.full_like(f, g0)
        phase = np.zeros_like(f)
    else:
        raise ValueError(f'unknown expected_bode kind: {kind}')
    return dict(f=f, gain=gain, phase=phase, label=label)


def _overlay_expected(axg, axp, expected):
    """Overlay a thin, marker-less expected curve on gain + phase axes.
    `expected` = dict(f=, gain=, phase=, label=)."""
    if not expected:
        return
    ef, eg, ep = expected['f'], expected['gain'], expected['phase']
    lab = expected.get('label', 'expected')
    axg.loglog(ef, eg, '-', color='#999999', lw=1.1, alpha=0.9, zorder=1, label=lab)
    axp.semilogx(ef, ep, '-', color='#999999', lw=1.1, alpha=0.9, zorder=1)


def _plain_gain_yaxis(axg):
    """Plain-number y-ticks (1, 0.8, 0.6, 0.4 …) on the log gain axis instead of
    10ˣ scientific notation — gains read much more naturally that way."""
    from matplotlib.ticker import FuncFormatter, LogLocator
    fmt = FuncFormatter(lambda y, _: f'{y:g}')
    axg.yaxis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 4.0, 6.0, 8.0)))
    axg.yaxis.set_major_formatter(fmt)
    axg.yaxis.set_minor_formatter(fmt)


def make_bode_figure(freqs, gains, phases, title, ref_hz=0.5, highpass=False,
                     gain_label='Gain', figsize=(7.5, 6.0), color='#2166ac',
                     expected=None):
    """Standard 2-panel Bode figure (gain loglog + phase semilogx) + metrics.

    Returns (fig, metrics_dict) where metrics_dict has gain_low / bw_hz / phase_ref.
    The −3 dB level and bandwidth corner are marked on the gain panel.
    """
    import matplotlib.pyplot as plt
    m = bode_metrics(freqs, gains, phases, ref_hz=ref_hz, highpass=highpass)
    fig, (axg, axp) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=11, fontweight='bold')

    axg.loglog(freqs, gains, 'o-', color=color, lw=1.6, ms=5)
    gm = m['gain_max']
    axg.axhline(gm, color='gray', lw=0.8, ls=':', alpha=0.7, label=f'max gain {gm:.3g}')
    if gm > 0:
        axg.axhline(gm / np.sqrt(2.0), color='tomato', lw=0.8, ls='--', alpha=0.6,
                    label='−3 dB')
    for fc, lbl in ((m['fc_lo'], 'fc_lo'), (m['fc_hi'], 'fc_hi')):
        if fc is not None:
            axg.axvline(fc, color='tomato', lw=1.0, ls=':', alpha=0.7, label=f'{lbl} {fc:.2g} Hz')
    _overlay_expected(axg, axp, expected)
    axg.set_ylabel(gain_label, fontsize=9)
    axg.grid(True, which='both', alpha=0.2)
    axg.legend(fontsize=8, loc='best')
    _plain_gain_yaxis(axg)

    axp.semilogx(freqs, phases, 's-', color=color, lw=1.6, ms=5)
    axp.axhline(0, color='k', lw=0.4)
    if ref_hz is not None:
        axp.axvline(ref_hz, color='gray', lw=0.8, ls=':', alpha=0.6)
    axp.set_ylabel('Phase (deg, − = lag)', fontsize=9)
    axp.set_xlabel('Frequency (Hz)', fontsize=9)
    axp.grid(True, which='both', alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, m


def make_bode_multi(freqs, series, title, ref_hz=0.5, highpass=False,
                    gain_label='Gain', figsize=(8.5, 6.5), expected=None):
    """Multi-curve Bode — several conditions on one gain + phase figure.

    series: list of dict(label=, gains=, phases=, color=).
    Returns (fig, {label: metrics_dict}) with per-series gain_low/bw_hz/phase_ref.
    """
    import matplotlib.pyplot as plt
    fig, (axg, axp) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=11, fontweight='bold')
    out = {}
    for s in series:
        lab = s['label']
        g = np.asarray(s['gains'], float)
        p = np.asarray(s['phases'], float)
        col = s.get('color')
        out[lab] = bode_metrics(freqs, g, p, ref_hz=ref_hz, highpass=highpass)
        axg.loglog(freqs, g, 'o-', color=col, lw=1.5, ms=4, label=lab)
        axp.semilogx(freqs, p, 's-', color=col, lw=1.5, ms=4, label=lab)
    axg.axhline(1.0, color='gray', lw=0.6, ls=':', alpha=0.5)
    _overlay_expected(axg, axp, expected)
    axg.set_ylabel(gain_label, fontsize=9)
    axg.grid(True, which='both', alpha=0.2)
    axg.legend(fontsize=8, loc='best')
    _plain_gain_yaxis(axg)
    axp.axhline(0, color='k', lw=0.4)
    if ref_hz is not None:
        axp.axvline(ref_hz, color='gray', lw=0.6, ls=':', alpha=0.4)
    axp.set_ylabel('Phase (deg, − = lag)', fontsize=9)
    axp.set_xlabel('Frequency (Hz)', fontsize=9)
    axp.grid(True, which='both', alpha=0.2)
    axp.legend(fontsize=8, loc='best')
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, out


if __name__ == '__main__':
    # Self-test: a known first-order low-pass H(s)=1/(1+sτ). The fit should
    # recover gain = 1/√(1+(ωτ)²) and phase = −atan(ωτ) at each frequency.
    tau = 0.3
    ok = True
    for f in [0.1, 0.5, 1.0, 2.0]:
        w = 2 * np.pi * f
        g_true = 1.0 / np.sqrt(1 + (w * tau) ** 2)
        p_true = -np.degrees(np.arctan(w * tau))
        t = np.arange(0.0, 12.0, 0.001)
        drive = np.sin(w * t)
        output = g_true * np.sin(w * t + np.radians(p_true))
        g, p = bode_point(t, drive, output, f)
        dg, dp = abs(g - g_true), abs(p - p_true)
        ok &= dg < 1e-3 and dp < 0.2
        print(f'f={f:>4} Hz  gain {g:.4f} (true {g_true:.4f}, Δ{dg:.1e})  '
              f'phase {p:+7.2f}° (true {p_true:+7.2f}°, Δ{dp:.1e})')
    print('SELF-TEST', 'PASS' if ok else 'FAIL')
