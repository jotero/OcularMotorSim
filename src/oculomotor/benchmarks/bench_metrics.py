"""Quantitative regression harness for the benchmark suite.

Benchmarks compute scalar metrics (gains, time constants, peak velocities)
directly from the simulation ``states`` arrays — never from the rendered
figures.  Each bench attaches a list of :class:`Metric` objects to a module
global ``METRICS`` during ``run()``; this module collects them, checks each
against two independent criteria, and prints a pass/fail table.

Two check types, both wanted:

* **Physiological band** (``lo`` / ``hi``) — literature-anchored bounds.  These
  are the "must-have" assertions (VOR gain 0.9–1.0, OKAN TC ~10–30 s).
* **Golden snapshot** (``golden_tol``) — fractional drift vs the last frozen
  value in ``golden_metrics.json``.  Catches regressions even where there is no
  physiological ground truth.  Refreeze with ``--update`` after an intended
  change.

A metric may carry both.  ``tier='gate'`` metrics fail the run on breach
(non-zero exit); ``tier='monitor'`` metrics are advisory (reported, never fatal).

Usage::

    python -X utf8 -m oculomotor.benchmarks.bench_metrics            # check
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --update   # refreeze golden
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Optional

# Benches whose figures + metrics this harness gathers. Grows as benches are wired in.
BENCH_MODULES = [
    'bench_saccades',
    'bench_vor_okr',
]

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'golden_metrics.json')


# ── Metric definition ─────────────────────────────────────────────────────────

@dataclass
class Metric:
    """One scalar measured from a simulation, with its acceptance criteria.

    Args:
        name:        unique key (snake_case, e.g. 'vor_okan_tc').
        value:       the measured scalar (NaN if extraction failed).
        tier:        'gate' (breach fails the run) or 'monitor' (advisory).
        lo, hi:      inclusive physiological band; either may be None for a
                     one-sided bound. None/None disables the band check.
        golden_tol:  fractional drift tolerance vs the frozen golden snapshot
                     (e.g. 0.1 = ±10%). None disables snapshot tracking.
        units:       display unit ('deg/s', 's', …).
        cite:        literature anchor for the band.
        desc:        one-line human description.
    """
    name: str
    value: float
    tier: str = 'monitor'
    lo: Optional[float] = None
    hi: Optional[float] = None
    golden_tol: Optional[float] = None
    units: str = ''
    cite: str = ''
    desc: str = ''


@dataclass
class Result:
    metric: Metric
    golden: Optional[float]
    band_ok: Optional[bool]    # None = no band defined
    drift: Optional[float]     # fractional drift vs golden, None if no golden
    drift_ok: Optional[bool]   # None = no snapshot to compare against
    status: str                # 'pass' | 'fail' | 'warn' | 'new'


# ── Golden snapshot store ─────────────────────────────────────────────────────

def load_golden(path: str = GOLDEN_PATH) -> dict:
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_golden(metrics: list[Metric], path: str = GOLDEN_PATH) -> None:
    """Freeze current values as the golden snapshot (sorted for stable diffs)."""
    snap = {m.name: (None if _isnan(m.value) else float(m.value)) for m in metrics}
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(snap.items())), f, indent=2)
        f.write('\n')


def _isnan(x) -> bool:
    try:
        return math.isnan(float(x))
    except (TypeError, ValueError):
        return True


# ── Standalone JSON data layer (server-side render reads these) ───────────────
# Two hand-inspectable JSONs under web/benchmarks/ are the canonical artifacts;
# the HTML is just a rendered view of them (no data embedded only in the page):
#   metrics_ranges.json    editable bands/tiers per metric (you tune this)
#   benchmarks_data.json   structure + measured values + golden (written by sims)
# Both the CLI gate and the page render read ranges.json, so they always agree.

_RANGE_FIELDS = ('lo', 'hi', 'tier', 'golden_tol', 'units', 'cite', 'desc')


def _web_dir():
    from oculomotor.benchmarks import bench_utils as utils
    return utils.BENCH_DIR


def ranges_path():
    return os.path.join(_web_dir(), 'metrics_ranges.json')


def data_path():
    return os.path.join(_web_dir(), 'benchmarks_data.json')


def load_ranges(path=None) -> dict:
    path = path or ranges_path()
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def seed_ranges(metrics, path=None) -> dict:
    """Ensure ranges.json has an entry per metric. Seeds missing names from the
    bench-defined defaults; PRESERVES existing (hand-edited) entries. Returns
    the merged dict. The code bands are thus only an initial default — once
    seeded, metrics_ranges.json is the source of truth you edit."""
    path = path or ranges_path()
    ranges = load_ranges(path)
    changed = False
    for m in metrics:
        if m.name not in ranges:
            ranges[m.name] = {k: getattr(m, k) for k in _RANGE_FIELDS}
            changed = True
    if changed:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(dict(sorted(ranges.items())), f, indent=2)
            f.write('\n')
    return ranges


def apply_ranges(metrics, ranges):
    """Override each metric's band/tier/etc from ranges.json (which wins over the
    inline code defaults). Names absent from ranges keep their code values."""
    out = []
    for m in metrics:
        r = ranges.get(m.name)
        if r:
            m = replace(m, **{k: r[k] for k in _RANGE_FIELDS if k in r})
        out.append(m)
    return out


def write_benchmarks_data(sections_data, golden, path=None) -> None:
    """Serialize the full gallery (structure + values + golden) to data.json.

    sections_data: [(section_meta_dict, [fig_dict, …]), …]; each fig_dict has a
    'metrics' list of Metric. Only name/value/golden are stored here — bands live
    in ranges.json — so the same values render against whatever ranges you set.
    """
    import datetime, oculomotor
    path = path or data_path()
    sections = []
    for meta, figs in sections_data:
        fig_out = []
        for f in figs:
            fig_out.append(dict(
                rel=f.get('rel', ''), title=f.get('title', ''),
                description=f.get('description', ''), expected=f.get('expected', ''),
                citation=f.get('citation', ''), type=f.get('type', 'behavior'),
                ref_rel=f.get('ref_rel', ''), diff_status=f.get('diff_status', ''),
                diff=f.get('diff', None),
                metrics=[dict(name=m.name,
                              value=(None if _isnan(m.value) else float(m.value)),
                              golden=golden.get(m.name))
                         for m in f.get('metrics', [])],
            ))
        sections.append(dict(id=meta.get('id', ''), title=meta.get('title', ''),
                             description=meta.get('description', ''),
                             runtime_s=meta.get('runtime_s'),
                             version=meta.get('version'),
                             generated=meta.get('generated'), figures=fig_out))
    blob = dict(generated=datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                version=oculomotor.__version__, sections=sections)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(blob, f, indent=2)
        f.write('\n')


def load_benchmarks_data(path=None) -> dict:
    path = path or data_path()
    if not os.path.isfile(path):
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def metric_from_record(rec: dict, ranges: dict) -> Metric:
    """Rebuild a Metric from a data.json record + ranges.json spec (for render)."""
    spec = ranges.get(rec['name'], {})
    val = rec.get('value')
    return Metric(
        name=rec['name'],
        value=float('nan') if val is None else float(val),
        tier=spec.get('tier', 'monitor'),
        lo=spec.get('lo'), hi=spec.get('hi'),
        golden_tol=spec.get('golden_tol'),
        units=spec.get('units', ''), cite=spec.get('cite', ''),
        desc=spec.get('desc', ''))


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(metrics: list[Metric], golden: dict) -> list[Result]:
    results = []
    for m in metrics:
        v = float(m.value)
        nan = _isnan(v)

        band_ok = None
        if m.lo is not None or m.hi is not None:
            band_ok = (not nan
                       and (m.lo is None or v >= m.lo)
                       and (m.hi is None or v <= m.hi))

        g = golden.get(m.name)
        drift = drift_ok = None
        if m.golden_tol is not None and g is not None:
            denom = abs(g) if abs(g) > 1e-9 else 1.0
            drift = (v - g) / denom
            drift_ok = (not nan) and abs(drift) <= m.golden_tol

        breached = (band_ok is False) or (drift_ok is False)
        if breached:
            status = 'fail' if m.tier == 'gate' else 'warn'
        elif m.golden_tol is not None and g is None:
            status = 'new'           # no snapshot yet — record on next --update
        else:
            status = 'pass'

        results.append(Result(m, g, band_ok, drift, drift_ok, status))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

_MARK = {'pass': 'PASS', 'fail': 'FAIL', 'warn': 'WARN', 'new': 'NEW '}


def _fmt(x, nan='   --') -> str:
    if x is None:
        return nan
    if _isnan(x):
        return ' nan'
    return f'{x:7.3f}'


def format_table(results: list[Result]) -> str:
    lines = []
    h = f'{"":4} {"metric":24} {"value":>9} {"band":>16} {"golden":>9} {"drift":>8}  tier'
    lines.append(h)
    lines.append('-' * len(h))
    for r in results:
        m = r.metric
        band = '       --       '
        if m.lo is not None or m.hi is not None:
            lo = '-inf' if m.lo is None else f'{m.lo:g}'
            hi = '+inf' if m.hi is None else f'{m.hi:g}'
            band = f'[{lo:>6},{hi:>6}]'
        drift = '' if r.drift is None else f'{r.drift * 100:+6.1f}%'
        unit = f' {m.units}' if m.units else ''
        lines.append(
            f'{_MARK[r.status]:4} {m.name:24} {_fmt(m.value)}{unit:<0} '
            f'{band:>16} {_fmt(r.golden)} {drift:>8}  {m.tier}')
    return '\n'.join(lines)


def summarize(results: list[Result]) -> dict:
    from collections import Counter
    return dict(Counter(r.status for r in results))


# ── Gather + main ─────────────────────────────────────────────────────────────

def gather(show: bool = False) -> list:
    """Run each wired bench; return [(section_title, [fig_dict, …]), …].

    Each fig_dict is what a bench panel function returns (path/rel/title/…) with
    a ``metrics`` key holding that figure's list of :class:`Metric`. Figures and
    metrics travel together so the dashboard can render them side by side.
    """
    import importlib
    sections = []
    for name in BENCH_MODULES:
        mod = importlib.import_module(f'oculomotor.benchmarks.{name}')
        print(f'\n--- running {name} for metrics ---')
        figs = mod.run(show=show) or []
        title = getattr(mod, 'SECTION', {}).get('title', name)
        sections.append((title, figs))
    return sections


# ── HTML dashboard ────────────────────────────────────────────────────────────

_HTML_STATUS = {
    'pass': ('#d4edda', '#155724', 'PASS'),
    'fail': ('#f8d7da', '#721c24', 'FAIL'),
    'warn': ('#fff3cd', '#856404', 'WARN'),
    'new':  ('#e2e3e5', '#383d41', 'NEW'),
}


def _html_chip(status: str) -> str:
    bg, fg, lbl = _HTML_STATUS.get(status, _HTML_STATUS['new'])
    return (f'<span style="display:inline-block;padding:2px 9px;border-radius:12px;'
            f'font-size:11px;font-weight:700;white-space:nowrap;'
            f'background:{bg};color:{fg}">{lbl}</span>')


def _html_num(x, unit='', nan='—'):
    if x is None:
        return nan
    if _isnan(x):
        return 'nan'
    return f'{x:.4g}{(" " + unit) if unit else ""}'


def _metric_table_html(metrics: list, golden: dict) -> str:
    """Right-column metrics table for one figure, or a visual-check note."""
    results = evaluate(metrics, golden)
    if not results:
        return ('<div class="novis">Visual check — no quantitative metric '
                'reduced from this figure yet.</div>')
    rows = []
    for r in results:
        m = r.metric
        # Band cell carries its literature source as a hover tooltip, so every
        # acceptance band is visibly anchored to a reference (dotted underline =
        # hoverable). Bands without a source render plain.
        band = '—'
        if m.lo is not None or m.hi is not None:
            lo = '−∞' if m.lo is None else f'{m.lo:g}'
            hi = '+∞' if m.hi is None else f'{m.hi:g}'
            if m.cite:
                band = f'<span class="band-src" title="source: {m.cite}">[{lo}, {hi}]</span>'
            else:
                band = f'[{lo}, {hi}]'
        drift = '—' if r.drift is None else f'{r.drift * 100:+.1f}%'
        # Tier (gate/monitor) stays in the data + CLI table; it's dropped from the
        # rendered page per request (the PASS/WARN chip already conveys severity).
        rows.append(f"""
          <tr>
            <td>{_html_chip(r.status)}</td>
            <td class="what" title="{m.name}">{m.desc}</td>
            <td class="num">{_html_num(m.value, m.units)}</td>
            <td class="num">{band}</td>
            <td class="num">{_html_num(r.golden)}</td>
            <td class="num">{drift}</td>
          </tr>""")
    return f"""<table>
      <thead><tr><th></th><th>measure</th><th>value</th><th>band</th>
        <th>golden</th><th>drift</th></tr></thead>
      <tbody>{''.join(rows)}</tbody></table>"""


def freeze_golden_from_data(path: str = GOLDEN_PATH) -> dict:
    """Snapshot every metric value in benchmarks_data.json as the golden baseline.

    Covers ALL sections (not only the wired gate modules) and needs no sims —
    the data file is already the canonical measured-value store, so freezing is
    just a copy of its values. Run the suite (full or --only) first to refresh.
    """
    data = load_benchmarks_data()
    snap = {}
    for sec in data.get('sections', []):
        for fig in sec.get('figures', []):
            for m in fig.get('metrics', []):
                v = m.get('value')
                snap[m['name']] = None if v is None else float(v)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(dict(sorted(snap.items())), f, indent=2)
        f.write('\n')
    return snap


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if '--update' in argv:
        snap = freeze_golden_from_data()
        if not snap:
            print('No benchmarks_data.json metrics to freeze — run the suite first.')
            return 0
        print(f'golden refrozen from benchmarks_data.json '
              f'({len(snap)} metrics, all sections) → {GOLDEN_PATH}')
        return 0

    # Gate: re-run the wired benches and check the CURRENT model vs golden + ranges.
    sections = gather()
    golden = load_golden()
    metrics = [m for _, figs in sections for fig in figs
               for m in fig.get('metrics', [])]
    # Bands come from metrics_ranges.json (seeded from code, then your edits win),
    # so the CLI gate evaluates against the same ranges the page renders.
    ranges = seed_ranges(metrics)
    metrics = apply_ranges(metrics, ranges)
    results = evaluate(metrics, golden)

    print('\n' + '=' * 78)
    print('QUANTITATIVE BENCHMARK METRICS  (gate: ' + ', '.join(BENCH_MODULES) + ')')
    print('=' * 78)
    print(format_table(results))
    tally = summarize(results)
    print('-' * 78)
    print(f'summary: {tally}')

    n_fail = tally.get('fail', 0)
    n_new = tally.get('new', 0)
    if n_new:
        print(f'\n{n_new} new metric(s) without a golden value — run --update to freeze.')
    if n_fail:
        print(f'\nFAILED: {n_fail} gate metric(s) out of band or drifted.')
        return 1
    print('\nAll gate metrics within band and tolerance.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
