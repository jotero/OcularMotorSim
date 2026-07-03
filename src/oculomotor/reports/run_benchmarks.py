"""Run all benchmark scripts and generate web/index.html.

Usage:
    python -X utf8 scripts/run_benchmarks.py           # run all sections
    python -X utf8 scripts/run_benchmarks.py --html-only # regenerate HTML from existing figures
    python -X utf8 scripts/run_benchmarks.py --show     # show figures interactively

Individual sections can be re-run by running their own scripts:
    python -X utf8 scripts/bench_saccades.py
    python -X utf8 scripts/bench_vor_okr.py
    ... etc.
Then re-run run_benchmarks.py --html-only to rebuild the report.
"""

import sys
import os
import datetime
import importlib
import subprocess
import hashlib

from oculomotor.benchmarks import bench_utils as utils
from oculomotor.benchmarks.bench_metrics import (
    _metric_table_html, _html_chip, load_golden, evaluate, summarize,
    load_ranges, apply_ranges, split_cites, cite_key, cite_links,
)
import oculomotor

_BENCH_VERSION = None


def bench_version():
    """`oculomotor.__version__`, but when the tree is dirty append a short hash of
    the working-tree diff. Plain '-dirty' is identical across edits, so it can't
    tell two uncommitted states apart; the diff hash makes each one distinct, so
    a section re-run after an edit gets a different version and the page can flag
    the mix. Computed once per process (stable across all sections of one run)."""
    global _BENCH_VERSION
    if _BENCH_VERSION is not None:
        return _BENCH_VERSION
    ver = oculomotor.__version__
    if ver.endswith('-dirty'):
        try:
            diff = subprocess.run(
                ['git', 'diff', 'HEAD'], cwd=utils._ROOT,
                capture_output=True, text=True, timeout=15).stdout
            h = hashlib.sha1(diff.encode('utf-8', 'ignore')).hexdigest()[:7]
            ver = f'{ver}.{h}'
        except Exception:
            pass
    _BENCH_VERSION = ver
    return ver

SHOW      = '--show' in sys.argv
HTML_ONLY = '--html-only' in sys.argv


def _parse_only(argv):
    """Parse `--only saccades,vor_okr` (or `--only=...`) → list of section names,
    or None. Names match module names with or without the 'bench_' prefix."""
    for i, a in enumerate(argv):
        if a.startswith('--only='):
            raw = a.split('=', 1)[1]
        elif a == '--only' and i + 1 < len(argv):
            raw = argv[i + 1]
        else:
            continue
        return [s.strip() for s in raw.split(',') if s.strip()]
    return None


ONLY = _parse_only(sys.argv)   # partial run: re-run only these sections

MODULES = [
    'bench_saccades',
    'bench_vor_okr',
    'bench_gravity',     # also renders the T-VOR figures (merged section)
    'bench_pursuit',
    'bench_vergence',
    'bench_fixation',
    'bench_listing',
    'bench_fcp',
    'bench_pupil',
    'bench_eyelid',
]

# Section ids never rendered into the report (data may still exist in
# benchmarks_data.json from older runs, so filter at render time too).
EXCLUDE_SECTIONS = {'clinical', 'tvor'}   # 'tvor' merged into 'gravity' (drop orphan)


# ── HTML generation ───────────────────────────────────────────────────────────

_HTML_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
html { scroll-behavior: smooth; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: #f5f5f5; color: #222; display: flex; }
nav  { width: 200px; height: 100vh; background: #1a1a2e; color: #eee;
       padding: 20px 0; position: sticky; top: 0; align-self: flex-start;
       overflow-y: auto; flex-shrink: 0; }
nav h2  { font-size: 13px; padding: 0 16px 12px; color: #aaa;
          text-transform: uppercase; letter-spacing: 0.05em; }
nav a   { display: block; padding: 8px 16px; color: #ccc; text-decoration: none;
          font-size: 13px; border-left: 3px solid transparent; }
nav a:hover { background: #2a2a4e; color: #fff; border-left-color: #4a90d9; }
main { flex: 1; padding: 32px; max-width: 1600px; }
h1   { font-size: 22px; margin-bottom: 4px; }
.meta { font-size: 12px; color: #888; margin-bottom: 32px; }
.section    { margin-bottom: 48px; }
.section h2 { font-size: 18px; margin-bottom: 6px; border-bottom: 2px solid #ddd;
              padding-bottom: 6px; }
.rt { font-size: 13px; color: #888; font-weight: normal; }
.stale { font-size: 12px; font-weight: 600; color: #856404; background: #fff3cd;
         border: 1px solid #f6c90e; border-radius: 10px; padding: 1px 8px; }
.warn-banner { background: #fff3cd; color: #856404; border: 1px solid #f6c90e;
               border-radius: 6px; padding: 10px 14px; font-size: 13px;
               margin-bottom: 24px; }
.section > p { font-size: 13px; color: #555; margin-bottom: 16px; }
.fig-grid   { display: grid; grid-template-columns: 1fr; gap: 20px; }
.fig-card   { background: #fff; border: 1px solid #e0e0e0; border-radius: 8px;
              padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,.08);
              display: grid; grid-template-columns: 1.1fr 1fr; gap: 24px;
              align-items: start; max-width: 1300px; }
.fig-main, .fig-metrics { min-width: 0; }
.fig-metrics { overflow-x: auto; }
.fig-card a img { width: 100%; border-radius: 4px; display: block;
                  border: 1px solid #eee; cursor: zoom-in; }
.fig-metrics table { width: 100%; border-collapse: collapse; font-size: 12px; }
.fig-metrics th { background: #1a1a2e; color: #ddd; text-align: left; padding: 6px 7px;
                  font-size: 10px; text-transform: uppercase; letter-spacing: .03em; }
.fig-metrics td { padding: 6px 7px; border-top: 1px solid #eee; vertical-align: top;
                  overflow-wrap: anywhere; }
.fig-metrics td.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
.fig-metrics td.what { font-size: 11px; color: #444; }
.fig-metrics code { font-size: 11px; }
.fig-metrics .desc { font-size: 10px; color: #777; margin-top: 2px; }
.fig-metrics .novis { font-size: 12px; color: #999; font-style: italic; padding: 12px;
                      background: #fafafa; border-radius: 6px; }
.fig-main h3 { font-size: 14px; margin: 0 0 8px; }
/* Narrative sits in the right column BELOW the metrics table — a titled note
   card describing the panels and what to look for, plus the literature anchor. */
.narrative   { margin-top: 16px; padding: 12px 14px; background: #fafbfc;
               border: 1px solid #eef0f3; border-radius: 6px; }
.narrative .narr-title { font-size: 10px; font-weight: 700; text-transform: uppercase;
               letter-spacing: .05em; color: #94a3b8; margin: 0 0 8px; }
.narrative .desc { font-size: 12px; color: #444; line-height: 1.55; margin: 0 0 8px; }
.narrative .citation { font-size: 11px; color: #888; font-style: italic; margin: 0; }
/* In-text numbered citations [n] → bibliography (full cite on hover). */
.cref  { color: #2563eb; text-decoration: none; font-weight: 600; }
.cref:hover { text-decoration: underline; }
.crefs { white-space: nowrap; }
.fig-metrics .crefs { font-size: 11px; margin-left: 2px; }
.biblio { margin-top: 4px; }
.biblio .ref   { display: flex; gap: 8px; font-size: 12px; color: #333;
                 padding: 5px 2px; border-top: 1px solid #f0f0f0; line-height: 1.5;
                 scroll-margin-top: 16px; }
.biblio .ref-n { color: #888; min-width: 24px; text-align: right;
                 font-variant-numeric: tabular-nums; }
.biblio .ref:target { background: #fffbea; }
.fig-pair       { display: grid; grid-template-columns: 1fr 1fr; gap: 6px;
                  align-items: start; }
.fig-pair .lbl  { font-size: 10px; color: #888; text-transform: uppercase;
                  letter-spacing: 0.04em; margin: 0 0 2px 2px; }
.diff-badge     { display: inline-block; padding: 2px 8px; border-radius: 12px;
                  font-size: 10px; font-weight: 700; letter-spacing: 0.04em;
                  text-transform: uppercase; margin-bottom: 6px; margin-right: 6px; }
.diff-match     { background: #d4edda; color: #155724; }
.diff-changed   { background: #f8d7da; color: #721c24; }
.diff-shape     { background: #fff3cd; color: #856404; }
"""

_HTML_LIGHTBOX = """
<div id="lb" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;
     background:rgba(0,0,0,.85);z-index:1000;cursor:zoom-out;align-items:center;
     justify-content:center;">
  <img id="lb-img" style="max-width:95vw;max-height:95vh;border-radius:4px;">
</div>
<script>
(function(){
  var lb=document.getElementById('lb'),
      li=document.getElementById('lb-img');
  document.querySelectorAll('.fig-main a').forEach(function(a){
    a.addEventListener('click',function(e){
      e.preventDefault();
      li.src=a.href; lb.style.display='flex';
    });
  });
  lb.addEventListener('click',function(){ lb.style.display='none'; });
})();
</script>
"""


def _diff_badge_html(fig):
    """Render the regression-status badge for a figure — only when a reference
    comparison actually ran. The bare 'no reference' case (the common one) is
    suppressed: it's noise, not a result."""
    status = fig.get('diff_status', 'no-ref')
    diff   = fig.get('diff')
    if status == 'match':
        return f'<span class="diff-badge diff-match">match (Δ={diff:.4f})</span>'
    if status == 'changed':
        return f'<span class="diff-badge diff-changed">CHANGED (Δ={diff:.4f})</span>'
    if status == 'shape-changed':
        return '<span class="diff-badge diff-shape">layout changed</span>'
    return ''  # 'no-ref' / 'unavailable' — no badge


def _figure_card(fig, golden, cite_map=None):
    rel    = fig.get('rel', '')
    title  = fig.get('title', '')
    desc   = fig.get('description', '')
    exp    = fig.get('expected', '')
    cit    = fig.get('citation', '')
    refrel = fig.get('ref_rel', '')

    # Check file exists
    path  = fig.get('path', '')
    if path and not os.path.isfile(path):
        img_html = '<div style="padding:30px;text-align:center;color:#aaa;font-size:13px;">Figure not yet generated</div>'
    elif refrel:
        # Side-by-side: current | reference
        img_html = (
            '<div class="fig-pair">'
            f'  <div><div class="lbl">current</div>'
            f'    <a href="{rel}" target="_blank"><img src="{rel}" alt="{title}"></a></div>'
            f'  <div><div class="lbl">reference</div>'
            f'    <a href="{refrel}" target="_blank"><img src="{refrel}" alt="{title} (reference)"></a></div>'
            '</div>'
        )
    else:
        img_html = f'<a href="{rel}" target="_blank"><img src="{rel}" alt="{title}"></a>'

    metrics_html = _metric_table_html(fig.get('metrics', []), golden, cite_map)

    # Titled narrative note (figure description + what the panels should show)
    # plus the citation rendered as numbered links into the bibliography.
    narr_text = ' '.join(t for t in (desc, exp) if t)
    narr_html = f'<p class="desc">{narr_text}</p>' if narr_text else ''
    cit_refs  = cite_links(cit, cite_map)
    cit_html  = f'<p class="citation">&#128214; {cit_refs}</p>' if cit_refs else ''
    narrative = (f'<div class="narrative">'
                 f'<h4 class="narr-title">What this shows</h4>{narr_html}{cit_html}'
                 f'</div>' if (narr_text or cit_refs) else '')

    return f"""
    <div class="fig-card">
      <div class="fig-main">
        <h3>{title}</h3>
        {_diff_badge_html(fig)}
        {img_html}
      </div>
      <div class="fig-metrics">
        {metrics_html}
        {narrative}
      </div>
    </div>"""


def _section_html(section_meta, figs, golden, cite_map=None):
    sid   = section_meta.get('id', '')
    title = section_meta.get('title', '')
    desc  = section_meta.get('description', '')
    rt    = section_meta.get('runtime_s')
    ver   = section_meta.get('version')
    gen   = section_meta.get('generated')
    cur   = bench_version()
    stale = bool(ver) and ver != cur

    bits = []
    if rt:  bits.append(f'{rt:.1f}s')
    if ver: bits.append(ver)
    if gen: bits.append(gen)
    meta_html  = f' <span class="rt">· {" · ".join(bits)}</span>' if bits else ''
    stale_html = (f' <span class="stale">&#9888; out of date — code now {cur}</span>'
                  if stale else '')
    cards = '\n'.join(_figure_card(f, golden, cite_map) for f in figs)
    return f"""
  <section class="section" id="{sid}">
    <h2>{title}{meta_html}{stale_html}</h2>
    <p>{desc}</p>
    <div class="fig-grid">
      {cards}
    </div>
  </section>"""


def _build_bibliography(sections_data):
    """Number every distinct paper cited across the page (figure citations +
    metric band sources) in first-appearance reading order.

    Returns (cite_map, biblio): cite_map maps a reference key → (number, full
    text); biblio is the ordered [(number, full text), …] list. The richest
    string seen for a paper (the one carrying journal/volume) becomes its
    canonical bibliography entry — unless a full titled reference is supplied in
    references.json (keyed by the same cite_key), which then wins."""
    from oculomotor.benchmarks.bench_metrics import load_references
    refs_db = load_references()
    order, best = [], {}

    def consider(cstr):
        for part in split_cites(cstr):
            k = cite_key(part)
            if not k:
                continue                      # not a paper (no year) → never numbered
            if k not in best:
                order.append(k); best[k] = part
            elif len(part) > len(best[k]):
                best[k] = part                # prefer the fuller citation string

    for _, figs in sections_data:
        for f in figs:
            for m in f.get('metrics', []):    # table (band sources) read first…
                consider(getattr(m, 'cite', '') or '')
            consider(f.get('citation', ''))   # …then the narrative citation
    # Full titled reference from references.json wins; else the richest cite string.
    disp = lambda k: refs_db.get(k) or best[k]
    cite_map = {k: (i + 1, disp(k)) for i, k in enumerate(order)}
    biblio   = [(i + 1, disp(k)) for i, k in enumerate(order)]
    return cite_map, biblio


def _bibliography_html(biblio):
    """Paper-style numbered reference list; each entry's id is the anchor the
    in-text [n] links jump to (and highlight via :target)."""
    if not biblio:
        return ''
    items = '\n'.join(
        f'      <div class="ref" id="ref-{n}"><span class="ref-n">{n}.</span>'
        f'<span class="ref-t">{full}</span></div>'
        for n, full in biblio)
    return f"""
  <section class="section" id="bibliography">
    <h2>References</h2>
    <p>Every acceptance band and figure is anchored to the literature below;
       in-text numbers link here (full citation on hover).</p>
    <div class="biblio">
{items}
    </div>
  </section>"""


def generate_html(sections_data):
    """Generate web/index.html from list of (section_meta, figs) tuples."""
    ts  = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    ver = bench_version()

    nav_links = '\n'.join(
        f'    <a href="#{s["id"]}">{s["title"]}</a>'
        for s, _ in sections_data
    )
    golden = load_golden()
    # Bands come from the standalone metrics_ranges.json (your edits win); apply
    # them so a band tweak shows up on a plain --html-only re-render, no sims.
    ranges = load_ranges()
    for _, figs in sections_data:
        for f in figs:
            if f.get('metrics'):
                f['metrics'] = apply_ranges(f['metrics'], ranges)
    all_metrics = [m for _, figs in sections_data for f in figs
                   for m in f.get('metrics', [])]
    # Bibliography numbering must follow the applied ranges (metric .cite lives in
    # ranges), so build it here after the apply loop above.
    cite_map, biblio = _build_bibliography(sections_data)
    tally = summarize(evaluate(all_metrics, golden))
    tally_html = (' &nbsp; '.join(f'{_html_chip(s)} {n}' for s, n in sorted(tally.items()))
                  or 'no metrics yet')

    # Version-consistency warning: each section records the code version it ran
    # at. Partial re-runs leave older sections behind, so flag a mixed/stale set.
    seen   = [s.get('version') for s, _ in sections_data if s.get('version')]
    distinct = sorted(set(seen))
    stale  = [s.get('title') for s, _ in sections_data
              if s.get('version') and s.get('version') != ver]
    if stale:
        warn_html = (f'<div class="warn-banner">&#9888; <b>{len(stale)} of '
                     f'{len(sections_data)}</b> sections were run on an older version '
                     f'(current build <b>{ver}</b>). Versions present: '
                     f'{", ".join(distinct)}. Re-run for a consistent picture — '
                     f'stale: {", ".join(stale)}.</div>')
    elif len(distinct) > 1:
        warn_html = (f'<div class="warn-banner">&#9888; sections were run at different '
                     f'versions: {", ".join(distinct)}.</div>')
    else:
        warn_html = ''

    sections_html = '\n'.join(_section_html(s, f, golden, cite_map)
                              for s, f in sections_data)
    biblio_html = _bibliography_html(biblio)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>ViSiOMlab — Benchmark Report</title>
  <style>{_HTML_CSS}</style>
</head>
<body>
  <nav>
    <h2 style="margin-bottom:4px;">Pages</h2>
    <a href="../">LLM Simulator</a>
    <a href="../clinical_benchmarks/">Clinical Benchmarks</a>
    <a href="../experiments/">Experiments</a>
    <a href="../parameters.html">Parameters</a>
    <div style="border-top:1px solid #2a2a4e;margin:10px 0 8px;"></div>
    <h2>Sections</h2>
{nav_links}
    <a href="#bibliography">References</a>
  </nav>
  <main>
    <h1>ViSiOMlab — Benchmark Report</h1>
    <p class="meta">
      Generated: <strong>{ts}</strong> &nbsp;|&nbsp;
      Version: <strong>{ver}</strong> &nbsp;|&nbsp;
      <a href="../BENCHMARKS.md">BENCHMARKS.md</a>
    </p>
    <p class="meta">Metric gate: {tally_html}</p>
    {warn_html}
{sections_html}
{biblio_html}
  </main>
  {_HTML_LIGHTBOX}
</body>
</html>"""

    os.makedirs(utils.DOCS_DIR, exist_ok=True)
    with open(utils.HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f'\nHTML report written: {utils.HTML_PATH}')


# ── Build sections: from JSON (fast, no sims) or by running benches (slow) ────

def _load_sections_from_data():
    """Reconstruct (section_meta, figs) from benchmarks_data.json + metrics_ranges.json
    WITHOUT running any simulation. Used by --html-only so editing the ranges JSON
    and re-rendering is instant. Figures load from their on-disk PNGs via 'rel'."""
    from oculomotor.benchmarks import bench_metrics as bm
    data   = bm.load_benchmarks_data()
    ranges = bm.load_ranges()
    if not data:
        print('  No benchmarks_data.json yet — run the full suite once to create it.')
        return []
    sections_data = []
    for sec in data.get('sections', []):
        meta = dict(id=sec.get('id', ''), title=sec.get('title', ''),
                    description=sec.get('description', ''),
                    runtime_s=sec.get('runtime_s'),
                    version=sec.get('version'),
                    generated=sec.get('generated'))
        figs = []
        for fr in sec.get('figures', []):
            figs.append(dict(
                path='', rel=fr.get('rel', ''), title=fr.get('title', ''),
                description=fr.get('description', ''), expected=fr.get('expected', ''),
                citation=fr.get('citation', ''), type=fr.get('type', 'behavior'),
                ref_rel=fr.get('ref_rel', ''), diff_status=fr.get('diff_status', ''),
                diff=fr.get('diff', None),
                metrics=[bm.metric_from_record(r, ranges) for r in fr.get('metrics', [])],
            ))
        sections_data.append((meta, figs))
    return sections_data


def _run_one_module(mod_name):
    """Run a single bench module (sims) → (section_meta, figs), stamping the
    wall-clock runtime + code version this section was actually run at."""
    import time
    mod = importlib.import_module(f'oculomotor.benchmarks.{mod_name}')
    t0 = time.perf_counter()
    try:
        figs = mod.run(show=SHOW)
    except Exception as e:
        print(f'  ERROR in {mod_name}: {e}')
        import traceback; traceback.print_exc()
        figs = []
    elapsed = time.perf_counter() - t0
    print(f'  [{mod_name}] ran in {elapsed:.1f}s')
    figs = [utils.ref_meta(dict(f), base_dir=utils.BENCH_DIR, ref_dir=utils.REF_DIR) for f in figs]
    meta = dict(mod.SECTION)
    meta['runtime_s'] = round(elapsed, 2)
    meta['version']   = bench_version()                 # version THIS section ran at
    meta['generated'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    return meta, figs


def _write_data(sections_data):
    """Seed editable ranges + write the standalone benchmarks_data.json."""
    from oculomotor.benchmarks import bench_metrics as bm
    all_metrics = [m for _, figs in sections_data for f in figs for m in f.get('metrics', [])]
    bm.seed_ranges(all_metrics)
    bm.write_benchmarks_data(sections_data, bm.load_golden())


def _run_all_benches():
    """Run every bench module and write the data artifacts."""
    sections_data = [_run_one_module(m) for m in MODULES]
    _write_data(sections_data)
    return sections_data


def _resolve_only(names):
    """Map requested names (with/without 'bench_' prefix) to MODULES entries."""
    want = set()
    for n in names:
        n = n.strip()
        want.add(n)
        want.add(n[len('bench_'):] if n.startswith('bench_') else 'bench_' + n)
    return [m for m in MODULES if m in want or m.replace('bench_', '') in want]


def _run_partial(names):
    """Re-run only the named sections; keep every other section as-is from
    benchmarks_data.json. The re-run sections get the current version stamp;
    the rest keep theirs — which is exactly what surfaces the mixed-version
    warning on the page."""
    targets = _resolve_only(names)
    valid = ', '.join(m.replace('bench_', '') for m in MODULES)
    if not targets:
        print(f'  --only matched no modules: {names}.  Valid: {valid}')
        return _load_sections_from_data()
    sections_data = _load_sections_from_data()
    if not sections_data:
        print('  No benchmarks_data.json yet — running the full suite first.')
        return _run_all_benches()
    by_id = {meta.get('id'): i for i, (meta, _) in enumerate(sections_data)}
    print(f'  Partial run: {", ".join(t.replace("bench_", "") for t in targets)}')
    for mod_name in targets:
        meta, figs = _run_one_module(mod_name)
        sid = meta.get('id')
        if sid in by_id:
            sections_data[by_id[sid]] = (meta, figs)
        else:
            sections_data.append((meta, figs))
    _write_data(sections_data)
    return sections_data


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(utils.FIGS_DIR, exist_ok=True)

    if ONLY:
        sections_data = _run_partial(ONLY)
    elif HTML_ONLY:
        sections_data = _load_sections_from_data()
    else:
        sections_data = _run_all_benches()

    # Drop excluded sections (e.g. clinical) from the rendered page even if they
    # still linger in benchmarks_data.json from an earlier full run.
    sections_data = [(m, f) for m, f in sections_data
                     if m.get('id') not in EXCLUDE_SECTIONS]

    generate_html(sections_data)
    print(f'\nDone. Open: {utils.HTML_PATH}')

    # Tally reference-comparison results so a regression is loud at the CLI too.
    all_figs = [f for _, figs in sections_data for f in figs]
    if all_figs:
        from collections import Counter
        tally = Counter(f.get('diff_status', 'no-ref') for f in all_figs)
        print(f'\nReference comparison: {dict(tally)}')
        changed = [f['title'] for f in all_figs if f.get('diff_status') == 'changed']
        if changed:
            print('  Changed vs. reference:')
            for title in changed:
                print(f'    - {title}')

    # Refresh parameters.html only on a full run (it depends on code, not JSON).
    if not HTML_ONLY:
        try:
            from oculomotor.reports import gen_parameters
            gen_parameters.main()
        except Exception as e:
            print(f'Warning: parameters.html regeneration failed: {e}')


if __name__ == '__main__':
    main()
