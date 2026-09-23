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

A metric may carry both.  An OUT-OF-BAND value is a hard **FAIL** (red, non-zero
exit) — the physiological assertion.  A golden DRIFT while the value is still in
band is **DRIFT** (amber, informational — re-freeze with --update after an
intended change), not a failure.  A metric carrying *neither* criterion is
**UNRATED** — it is measured but nothing checks it, which is a coverage hole, not
a pass.  No gate/monitor tier split.

Scoring is **offline by default**: ``benchmarks_data.json`` already holds every
measured value from the last suite run, so the whole scoreboard (all sections,
not just the simulated ones) scores in ~2 s with no ODE solves.  Re-simulating is
opt-in via ``--run``, because the full suite is ~60 min.  Since offline values are
only as fresh as the run that produced them, every report prints per-section
staleness (the code version each section was measured at vs the current build).

Three artifacts back this, all hand-inspectable:

* ``web/benchmarks/benchmarks_data.json``  measured values (written by suite runs)
* ``web/benchmarks/metrics_ranges.json``   editable bands (source of truth once seeded)
* ``benchmarks/golden_metrics.json``       frozen reference snapshot
* ``web/benchmarks/metrics_history.jsonl`` append-only per-run log → the trend column

Usage::

    python -X utf8 -m oculomotor.benchmarks.bench_metrics                # score data.json (fast)
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --fails        # only non-passing rows
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --section vor  # filter by section id
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --run          # re-simulate, then score
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --run saccades # re-simulate one section
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --update       # refreeze golden
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --history NAME # one metric across runs
    python -X utf8 -m oculomotor.benchmarks.bench_metrics --prune-ranges # drop orphan band entries
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from dataclasses import dataclass, replace
from typing import Optional

# Benches whose figures + metrics this harness gathers, in report order.
# Canonical list: reports/run_benchmarks.py imports it rather than keeping its own,
# so a bench can never be wired into the report but missing from the metric gate.
BENCH_MODULES = [
    'bench_saccades',
    'bench_vor_okr',
    'bench_gravity',     # also renders the T-VOR figures (merged section)
    'bench_pursuit',
    'bench_vergence',
    'bench_fixation',
    'bench_gaze_holding',
    'bench_listing',
    'bench_fcp',
    'bench_pupil',
    'bench_eyelid',
]

GOLDEN_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'golden_metrics.json')


# ── Metric definition ─────────────────────────────────────────────────────────

@dataclass
class Metric:
    """One scalar measured from a simulation, with its acceptance criteria.

    Out of band → FAIL (red). In band but drifted from golden → DRIFT (amber,
    informational). No gate/monitor tier split.

    Args:
        name:        unique key (snake_case, e.g. 'vor_okan_tc').
        value:       the measured scalar (NaN if extraction failed).
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
#   metrics_ranges.json    editable bands per metric (you tune this)
#   benchmarks_data.json   structure + measured values + golden (written by sims)
# Both the CLI gate and the page render read ranges.json, so they always agree.

_RANGE_FIELDS = ('lo', 'hi', 'golden_tol', 'units', 'cite', 'desc')


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


def references_path():
    return os.path.join(_web_dir(), 'references.json')


def load_references(path=None) -> dict:
    """Full titled references for the bibliography, keyed by cite_key (see
    cite_key()). Optional + hand-editable; missing keys fall back to the short
    citation strings carried in the benchmark data."""
    path = path or references_path()
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
    """Override each metric's band/etc from ranges.json (which wins over the
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


# ── Run history (the trend column) ────────────────────────────────────────────
# Golden is a single deliberately-frozen reference, so it cannot answer "is this
# getting better?" across a tuning session — re-freezing destroys the comparison.
# metrics_history.jsonl is the append-only complement: one JSON record per suite
# run, so a metric's trajectory over successive attempts stays readable. Append
# only; never rewrite (the whole value is that old rows are immutable).

def history_path():
    return os.path.join(_web_dir(), 'metrics_history.jsonl')


def load_history(path=None) -> list:
    """Read the run log, oldest first. Malformed lines are skipped rather than
    fatal — a truncated tail must not break the scoreboard."""
    path = path or history_path()
    if not os.path.isfile(path):
        return []
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def record_history(sections_data=None, sections_run=None, path=None) -> dict:
    """Append one record for the current measured values.

    ``sections_data`` is the in-memory [(meta, figs), …] from a suite run; when
    omitted the values are read back from benchmarks_data.json (so a bare
    ``--record`` can bootstrap history from whatever is already on disk).
    ``sections_run`` names the sections actually re-simulated — the rest of the
    record is carried-over older measurements, and saying so keeps the log honest.
    """
    from oculomotor.benchmarks import bench_utils as utils
    import datetime
    values, sec_versions = {}, {}
    if sections_data is None:
        data = load_benchmarks_data()
        for sec in data.get('sections', []):
            sec_versions[sec.get('id', '')] = sec.get('version')
            for fig in sec.get('figures', []):
                for m in fig.get('metrics', []):
                    values[m['name']] = m.get('value')
    else:
        for meta, figs in sections_data:
            sec_versions[meta.get('id', '')] = meta.get('version')
            for fig in figs:
                for m in fig.get('metrics', []):
                    values[m.name] = None if _isnan(m.value) else float(m.value)
    rec = dict(ts=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
               version=utils.bench_version(),
               sections_run=sorted(sections_run) if sections_run else [],
               section_versions=sec_versions,
               values=values)
    path = path or history_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(rec) + '\n')
    return rec


def previous_values(history: list = None, path=None) -> dict:
    """Values from the most recent history record — the trend baseline.

    Deliberately the previous *record*, not the previous distinct value: "no
    change since last run" is itself the answer you want when checking whether a
    parameter edit moved a metric at all.
    """
    history = load_history(path) if history is None else history
    return dict(history[-1].get('values', {})) if history else {}


def metric_from_record(rec: dict, ranges: dict) -> Metric:
    """Rebuild a Metric from a data.json record + ranges.json spec (for render)."""
    spec = ranges.get(rec['name'], {})
    val = rec.get('value')
    return Metric(
        name=rec['name'],
        value=float('nan') if val is None else float(val),
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

        if band_ok is False:
            status = 'fail'          # OUT OF BAND — physiological violation (red, fails the run)
        elif drift_ok is False:
            status = 'drift'         # in band, but moved from the golden snapshot
                                     # (amber, informational — re-freeze after an intended change)
        elif m.golden_tol is not None and g is None:
            status = 'new'           # no snapshot yet — record on next --update
        elif band_ok is None and drift_ok is None:
            status = 'unrated'       # measured, but NO criterion applies — a coverage
                                     # hole (missing from ranges, or band+tol both None).
                                     # Never call this a pass: nothing was checked.
        else:
            status = 'pass'

        results.append(Result(m, g, band_ok, drift, drift_ok, status))
    return results


# ── Reporting ─────────────────────────────────────────────────────────────────

_MARK = {'pass': 'PASS', 'fail': 'FAIL', 'drift': 'DRIFT', 'new': 'NEW', 'unrated': '----'}

_HEADER = (f'{"":5} {"metric":34} {"value":>9} {"":7} {"band":>15} '
           f'{"golden":>9} {"vs gold":>8} {"vs prev":>8}')


def _fmt(x, nan='   --') -> str:
    if x is None:
        return nan
    if _isnan(x):
        return ' nan'
    return f'{x:7.3f}'


def _pct(x) -> str:
    """Signed percentage, or '' when there is nothing to compare against."""
    if x is None:
        return ''
    if _isnan(x):
        return '   nan'
    return f'{x * 100:+6.1f}%'


def _trend(value, prev) -> Optional[float]:
    """Fractional change vs the previous run's value for the same metric."""
    if prev is None or _isnan(value) or _isnan(prev):
        return None
    denom = abs(prev) if abs(prev) > 1e-9 else 1.0
    return (float(value) - float(prev)) / denom


def format_table(results: list[Result], prev: dict = None, header: bool = True) -> str:
    """One row per metric. ``prev`` (name → last run's value) adds the trend column."""
    prev = prev or {}
    lines = []
    if header:
        lines.append(_HEADER)
        lines.append('-' * len(_HEADER))
    for r in results:
        m = r.metric
        band = '      --       '
        if m.lo is not None or m.hi is not None:
            lo = '-inf' if m.lo is None else f'{m.lo:g}'
            hi = '+inf' if m.hi is None else f'{m.hi:g}'
            band = f'[{lo:>6},{hi:>6}]'
        # Units get their own left-aligned column so the numeric columns stay
        # aligned — 'deg/s' next to the value used to shove every later column right.
        lines.append(
            f'{_MARK[r.status]:5} {m.name:34} {_fmt(m.value)} {m.units[:7]:<7} '
            f'{band:>15} {_fmt(r.golden)} {_pct(r.drift):>8} '
            f'{_pct(_trend(m.value, prev.get(m.name))):>8}')
    return '\n'.join(lines)


def format_report(scored: list, prev: dict = None, fails_only: bool = False) -> str:
    """Full scoreboard: metrics grouped under their section, each section headed by
    when it was measured and whether that measurement is stale.

    ``scored`` is [(section_meta, [Result, …]), …] from :func:`score_from_data`.
    Grouping matters because a failure is only actionable once you know which
    section to re-run — and whether its numbers predate the current code.
    """
    from oculomotor.benchmarks import bench_utils as utils
    cur = utils.bench_version()
    out = [_HEADER, '-' * len(_HEADER)]
    for meta, results in scored:
        shown = [r for r in results if not (fails_only and r.status == 'pass')]
        tally = summarize(results)
        bits = [f'{len(results)} metrics']
        if meta.get('generated'):
            bits.append(meta['generated'])
        ver = meta.get('version')
        if ver:
            bits.append(ver if ver == cur else f'{ver} STALE (code now {cur})')
        counts = ' '.join(f'{_MARK[s]}:{n}' for s, n in sorted(tally.items()))
        out.append('')
        out.append(f'── {meta.get("id", "?"):22} {" · ".join(bits)}  {counts}')
        if shown:
            out.append(format_table(shown, prev, header=False))
        elif results:
            out.append(f'{"":5} (all {len(results)} metrics pass)')
        else:
            out.append(f'{"":5} (no metrics — visual check only)')
    return '\n'.join(out)


def summarize(results: list[Result]) -> dict:
    from collections import Counter
    return dict(Counter(r.status for r in results))


# ── Offline scoring (no simulation) ───────────────────────────────────────────

def score_from_data(section_filter: str = None) -> list:
    """Score every metric in benchmarks_data.json against ranges + golden.

    Returns [(section_meta, [Result, …]), …]. No ODE solves — the data file is
    already the canonical measured-value store, so this covers ALL sections
    (including ones the gate never re-simulates) in a couple of seconds.
    """
    data, ranges, golden = load_benchmarks_data(), load_ranges(), load_golden()
    scored = []
    for sec in data.get('sections', []):
        sid = sec.get('id', '')
        if section_filter and section_filter not in sid:
            continue
        meta = dict(id=sid, title=sec.get('title', ''), version=sec.get('version'),
                    generated=sec.get('generated'), runtime_s=sec.get('runtime_s'))
        metrics = [metric_from_record(rec, ranges)
                   for fig in sec.get('figures', []) for rec in fig.get('metrics', [])]
        scored.append((meta, evaluate(metrics, golden)))
    return scored


def audit() -> dict:
    """Cross-check the three artifacts for drift *between* them (as opposed to in
    the model): band entries nobody emits any more, emitted metrics nobody bands,
    and metrics with no golden value. Each is a silent hole in the gate.

    Deliberately takes no ``scored`` argument and always reads the FULL data file:
    scoped against a ``--section`` subset, every band outside that section would
    look like an orphan — and --prune-ranges would then delete it.
    """
    ranges, golden = load_ranges(), load_golden()
    scored = score_from_data()
    emitted = {r.metric.name for _, results in scored for r in results}
    return dict(
        orphan_ranges=sorted(set(ranges) - emitted),
        missing_ranges=sorted(n for n in emitted if n not in ranges),
        missing_golden=sorted(n for n in emitted if golden.get(n) is None),
        drift_only=sorted(r.metric.name for _, results in scored for r in results
                          if r.metric.lo is None and r.metric.hi is None),
    )


def prune_ranges(path=None) -> list:
    """Delete band entries for metrics no bench emits any more (renames orphan
    them). Golden self-prunes on --update because it is rebuilt from data.json;
    ranges is seed-only, so stale entries linger until removed here."""
    path = path or ranges_path()
    ranges = load_ranges(path)
    orphans = audit()['orphan_ranges']
    if orphans:
        for k in orphans:
            del ranges[k]
        with open(path, 'w', encoding='utf-8') as f:
            # Same dump options as seed_ranges — a different ensure_ascii would
            # rewrite every non-ASCII desc and bury the real change in churn.
            json.dump(dict(sorted(ranges.items())), f, indent=2)
            f.write('\n')
    return orphans


# ── Re-simulation (opt-in; the slow path) ─────────────────────────────────────

def resimulate(names: list = None) -> list:
    """Re-run bench sims, refresh the data artifacts + report page, and return the
    resulting [(section_meta, figs), …].

    Delegates to reports.run_benchmarks rather than re-running modules here: that
    is the one place that stamps each section with the code version it ran at,
    writes benchmarks_data.json, seeds ranges and records history. A second
    orchestration path would inevitably skip one of those. Imported lazily —
    run_benchmarks imports this module at its top.
    """
    from oculomotor.reports import run_benchmarks as rb
    sections_data = rb._run_partial(names) if names else rb._run_all_benches()
    rb.generate_html([(m, f) for m, f in sections_data
                      if m.get('id') not in rb.EXCLUDE_SECTIONS])
    return sections_data


# ── HTML dashboard ────────────────────────────────────────────────────────────

_HTML_STATUS = {
    'pass':    ('#d4edda', '#155724', 'PASS'),
    'fail':    ('#f8d7da', '#721c24', 'FAIL'),
    'drift':   ('#fff3cd', '#856404', 'DRIFT'),
    'new':     ('#e2e3e5', '#383d41', 'NEW'),
    'unrated': ('#e2e3e5', '#6c757d', 'UNRATED'),
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


# ── Citation numbering (shared by the metric table + the page bibliography) ────

_CITE_YEAR_RE = re.compile(r'(1[89]\d\d|20\d\d)')


def split_cites(s: str) -> list:
    """Split a citation string into individual references on ';'."""
    return [p.strip() for p in (s or '').split(';') if p.strip()]


def cite_key(s: str):
    """Dedup key for one reference: first-author token + 4-digit year. Returns
    None when there's no year (i.e. not a real paper — e.g. a 'numerical sanity'
    note or a 'Debug diagnostic'), so such entries are never numbered."""
    m = _CITE_YEAR_RE.search(s or '')
    if not m:
        return None
    head = (s[:m.start()]).replace('(', ' ')
    toks = re.sub(r'[^A-Za-z ]', ' ', head).split()
    return f'{toks[0].lower()}{m.group(1)}' if toks else None


def cite_links(cite_str: str, cite_map: dict) -> str:
    """Render a (possibly multi-paper) citation as bracketed, linked reference
    numbers — '[3]' / '[3, 7]' — each anchored to the page bibliography with the
    full reference as a hover tooltip. Non-paper / unknown cites render ''."""
    if not cite_map:
        return ''
    nums = {}
    for part in split_cites(cite_str):
        ent = cite_map.get(cite_key(part))
        if ent:
            nums[ent[0]] = ent[1]
    if not nums:
        return ''
    links = ', '.join(
        f'<a class="cref" href="#ref-{n}" title="{full}">{n}</a>'
        for n, full in sorted(nums.items()))
    return f'<span class="crefs">[{links}]</span>'


def _metric_table_html(metrics: list, golden: dict, cite_map: dict = None) -> str:
    """Right-column metrics table for one figure, or a visual-check note.

    When ``cite_map`` (reference key → (number, full text)) is supplied, each
    band's literature source is shown as a numbered link into the bibliography."""
    results = evaluate(metrics, golden)
    if not results:
        return ('<div class="novis">Visual check — no quantitative metric '
                'reduced from this figure yet.</div>')
    rows = []
    for r in results:
        m = r.metric
        # Band cell shows its bounds plus the numbered reference(s) that anchor
        # them (links to the bibliography, full cite on hover). Bands without a
        # literature source (e.g. numerical sanity gates) render plain.
        band = '—'
        if m.lo is not None or m.hi is not None:
            lo = '−∞' if m.lo is None else f'{m.lo:g}'
            hi = '+∞' if m.hi is None else f'{m.hi:g}'
            refs = cite_links(m.cite, cite_map)
            band = f'[{lo}, {hi}]' + (f' {refs}' if refs else '')
        drift = '—' if r.drift is None else f'{r.drift * 100:+.1f}%'
        # No tiers — every metric is equally important; the PASS/FAIL chip conveys
        # whether it's in band (any breach is red).
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


def _arg_value(argv, flag):
    """Value following ``--flag`` (or in ``--flag=value``); None if absent, '' if
    the flag is present with no value."""
    for i, a in enumerate(argv):
        if a == flag:
            nxt = argv[i + 1] if i + 1 < len(argv) else ''
            return '' if nxt.startswith('-') else nxt
        if a.startswith(flag + '='):
            return a.split('=', 1)[1]
    return None


def print_history(name: str, path=None) -> int:
    """One metric's trajectory across runs — the question golden cannot answer."""
    history = load_history(path)
    rows = [(r, r.get('values', {})[name]) for r in history if name in r.get('values', {})]
    if not rows:
        print(f'No history for {name!r} '
              f'({len(history)} run(s) logged{"" if history else "; none yet"}).')
        return 1
    ranges = load_ranges()
    spec = ranges.get(name, {})
    lo, hi = spec.get('lo'), spec.get('hi')
    band = ('' if lo is None and hi is None else
            f'   band [{"-inf" if lo is None else f"{lo:g}"}, '
            f'{"+inf" if hi is None else f"{hi:g}"}]')
    print(f'\n{name}{band}   golden={_fmt(load_golden().get(name))}')
    print(f'{"run":20} {"version":22} {"value":>9} {"Δ prev":>8}  sections re-run')
    print('-' * 88)
    prev = None
    for rec, val in rows:
        # No sections_run ⇒ a --record snapshot of whatever data.json already held,
        # not a full re-run. Say so rather than implying everything was measured.
        ran = ', '.join(s.replace('bench_', '')
                        for s in rec.get('sections_run', [])) or '(snapshot)'
        print(f'{rec.get("ts", ""):20} {str(rec.get("version", "")):22} '
              f'{_fmt(val)} {_pct(_trend(val, prev)):>8}  {ran}')
        prev = val
    return 0


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

    hist = _arg_value(argv, '--history')
    if hist:
        return print_history(hist)

    if '--record' in argv:
        rec = record_history()
        print(f'recorded {len(rec["values"])} metric value(s) at {rec["ts"]} '
              f'({rec["version"]}) → {history_path()}')
        return 0

    # Trend baseline must be read BEFORE --run appends this run's own record,
    # otherwise every metric would compare against itself and read as unchanged.
    prev = previous_values()

    if '--prune-ranges' in argv:
        orphans = prune_ranges()
        print(f'pruned {len(orphans)} orphan band entr(y/ies) from {ranges_path()}')
        for k in orphans:
            print(f'   - {k}')
        return 0

    run = _arg_value(argv, '--run')
    if run is not None:
        names = [s.strip() for s in run.split(',') if s.strip()] or None
        print(f'Re-simulating: {", ".join(names) if names else "all sections"} '
              f'(this is the slow path — the full suite is ~60 min).')
        resimulate(names)

    section = _arg_value(argv, '--section') or None
    scored = score_from_data(section)
    if not scored:
        print('No benchmarks_data.json metrics to score — run --run once to create it.')
        return 0

    results = [r for _, results in scored for r in results]
    tally = summarize(results)

    print('\n' + '=' * len(_HEADER))
    print(f'QUANTITATIVE BENCHMARK METRICS  ({len(results)} metrics, '
          f'{len(scored)} section(s){f", filter={section!r}" if section else ""})')
    print('=' * len(_HEADER))
    print(format_report(scored, prev, fails_only='--fails' in argv))
    print('-' * len(_HEADER))
    print(f'summary: {tally}')

    # Staleness: offline values are only as fresh as the run that measured them.
    from oculomotor.benchmarks import bench_utils as utils
    cur = utils.bench_version()
    stale = [m.get('id') for m, _ in scored if m.get('version') and m['version'] != cur]
    if stale:
        print(f'\nSTALE: {len(stale)}/{len(scored)} section(s) measured at an older '
              f'build (code now {cur}): {", ".join(stale)}.'
              f'\n  Re-measure with --run {",".join(s for s in stale if s)}')

    aud = audit()          # always full-data — never scoped to --section
    for key, msg in (('missing_ranges', 'emitted metric(s) with NO band entry'),
                     ('orphan_ranges',  'band entr(y/ies) no bench emits (--prune-ranges)'),
                     ('missing_golden', 'metric(s) with no golden value (--update)')):
        if aud[key]:
            print(f'\n{len(aud[key])} {msg}: {", ".join(aud[key][:8])}'
                  f'{" …" if len(aud[key]) > 8 else ""}')
    if aud['drift_only']:
        print(f'\n{len(aud["drift_only"])} metric(s) have no physiological band '
              f'(drift-tracked only): {", ".join(aud["drift_only"])}')

    n_fail, n_drift = tally.get('fail', 0), tally.get('drift', 0)
    n_new, n_unrated = tally.get('new', 0), tally.get('unrated', 0)
    if n_new:
        print(f'\n{n_new} new metric(s) without a golden value — run --update to freeze.')
    if n_drift:
        print(f'\n{n_drift} metric(s) DRIFTED from golden but stay in band — informational; '
              f're-freeze with --update after an intended change.')
    if n_unrated:
        print(f'\n{n_unrated} metric(s) UNRATED — measured but no band and no golden '
              f'tolerance, so nothing checks them.')
    if n_fail:
        print(f'\nFAILED: {n_fail} metric(s) OUT OF BAND.')
        return 1
    print('\nAll metrics within band.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
