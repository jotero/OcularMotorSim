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


def bode_point(t, drive, output, f, settle_frac=0.5):
    """Gain and phase (deg) of `output` relative to `drive` at frequency `f`.

    Only the steady-state tail is used (the last ``1 − settle_frac`` of the
    record), so the initial transient doesn't bias the fit.
    """
    t = np.asarray(t, dtype=float)
    t0, t1 = t[0], t[-1]
    mask = t >= (t0 + settle_frac * (t1 - t0))
    amp_in,  ph_in,  _ = fit_sinusoid(t[mask], np.asarray(drive)[mask],  f)
    amp_out, ph_out, _ = fit_sinusoid(t[mask], np.asarray(output)[mask], f)
    gain = amp_out / amp_in if amp_in > 1e-9 else float('nan')
    phase = np.degrees(((ph_out - ph_in + np.pi) % (2.0 * np.pi)) - np.pi)
    return gain, phase


def bode_sweep(run_fn, freqs, settle_frac=0.5):
    """Sweep `freqs`, calling ``run_fn(f) -> (t, drive, output)`` at each.

    Returns (freqs, gains, phases_deg) as float arrays.
    """
    gains, phases = [], []
    for f in freqs:
        t, drive, output = run_fn(f)
        g, p = bode_point(t, drive, output, f, settle_frac)
        gains.append(g)
        phases.append(p)
    return np.asarray(freqs, float), np.asarray(gains, float), np.asarray(phases, float)


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


def bode_metrics(freqs, gains, phases, ref_hz=None, highpass=False):
    """Uniform scalar metrics from a Bode sweep.

    Returns dict with:
        gain_low : low-frequency (DC-ish) gain — gains[0], or gains[-1] if highpass
        bw_hz    : −3 dB bandwidth relative to gain_low (corner frequency)
        phase_ref: phase (deg) at `ref_hz` (nearest swept frequency), or None

    `highpass=True` (e.g. tVOR) takes the plateau at the HIGH-frequency end and
    the corner where gain falls to gain_low/√2 going down in frequency.
    """
    freqs = np.asarray(freqs, float)
    gains = np.asarray(gains, float)
    gain_low = float(gains[-1] if highpass else gains[0])
    thresh = gain_low / np.sqrt(2.0)
    bw = _interp_crossing(freqs, gains, thresh)
    phase_ref = None
    if ref_hz is not None and len(freqs):
        phase_ref = float(phases[int(np.argmin(np.abs(freqs - ref_hz)))])
    return dict(gain_low=gain_low, bw_hz=bw, phase_ref=phase_ref)


# Standard frequency grid (Hz) for closed-loop sweeps — log-spaced 0.05–5 Hz.
STD_FREQS = np.array([0.05, 0.1, 0.2, 0.35, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 5.0])


def make_bode_figure(freqs, gains, phases, title, ref_hz=0.5, highpass=False,
                     gain_label='Gain', figsize=(7.5, 6.0), color='#2166ac'):
    """Standard 2-panel Bode figure (gain loglog + phase semilogx) + metrics.

    Returns (fig, metrics_dict) where metrics_dict has gain_low / bw_hz / phase_ref.
    The −3 dB level and bandwidth corner are marked on the gain panel.
    """
    import matplotlib.pyplot as plt
    m = bode_metrics(freqs, gains, phases, ref_hz=ref_hz, highpass=highpass)
    fig, (axg, axp) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    fig.suptitle(title, fontsize=11, fontweight='bold')

    axg.loglog(freqs, gains, 'o-', color=color, lw=1.6, ms=5)
    gl = m['gain_low']
    axg.axhline(gl, color='gray', lw=0.8, ls=':', alpha=0.7,
                label=f'{"high" if highpass else "low"}-f gain {gl:.3g}')
    if gl > 0:
        axg.axhline(gl / np.sqrt(2.0), color='tomato', lw=0.8, ls='--', alpha=0.6,
                    label='−3 dB')
    if m['bw_hz'] == m['bw_hz']:   # not NaN
        axg.axvline(m['bw_hz'], color='tomato', lw=1.0, ls=':', alpha=0.7,
                    label=f'BW {m["bw_hz"]:.2g} Hz')
    axg.set_ylabel(gain_label, fontsize=9)
    axg.grid(True, which='both', alpha=0.2)
    axg.legend(fontsize=8, loc='best')

    axp.semilogx(freqs, phases, 's-', color=color, lw=1.6, ms=5)
    axp.axhline(0, color='k', lw=0.4)
    if ref_hz is not None and m['phase_ref'] is not None:
        axp.axvline(ref_hz, color='gray', lw=0.8, ls=':', alpha=0.6)
        axp.annotate(f'{m["phase_ref"]:.0f}° @ {ref_hz:g} Hz',
                     xy=(ref_hz, m['phase_ref']), fontsize=8,
                     xytext=(5, 5), textcoords='offset points')
    axp.set_ylabel('Phase (deg, − = lag)', fontsize=9)
    axp.set_xlabel('Frequency (Hz)', fontsize=9)
    axp.grid(True, which='both', alpha=0.2)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, m


def make_bode_multi(freqs, series, title, ref_hz=0.5, highpass=False,
                    gain_label='Gain', figsize=(8.5, 6.5)):
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
    axg.set_ylabel(gain_label, fontsize=9)
    axg.grid(True, which='both', alpha=0.2)
    axg.legend(fontsize=8, loc='best')
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
