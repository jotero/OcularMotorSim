"""Generate outputs/admin.html — standalone, works from file://.

Data is embedded as a JS constant so no HTTP server is needed.
Images reference server_figures/<run_id>.png via relative paths, which
browsers allow from file://.

Usage
-----
    python scripts/gen_admin.py            # regenerate from current CSV
"""

import csv
import json
import os
from pathlib import Path

# Mirror app.py: resolve the same data/ dir (OCULOMOTOR_DATA, else this checkout's
# repo-root data/) so the admin reads/writes the server's database.
_REPO_ROOT   = Path(__file__).resolve().parents[3]   # server → oculomotor → src → repo
_DATA_ROOT   = Path(os.environ.get('OCULOMOTOR_DATA') or (_REPO_ROOT / 'server_data'))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)
_LOG_FILE    = _DATA_ROOT / 'simulation_log.csv'
_OUT_FILE    = _DATA_ROOT / 'admin.html'

_LOG_COLUMNS = [
    'timestamp', 'run_id', 'version', 'prompt', 'mode', 'title',
    'figure_file', 'looks_correct', 'feedback',
    'favorite', 'featured', 'note', 'ms_total', 'ms_llm', 'ms_sim',
]


def _fig_rel(figure_file: str) -> str:
    """Convert absolute figure path to relative path from outputs/."""
    if not figure_file:
        return ''
    base = figure_file.replace('\\', '/').split('/')[-1]
    return f'server_figures/{base}'


def load_rows() -> list[dict]:
    if not _LOG_FILE.exists():
        return []
    with open(_LOG_FILE, newline='', encoding='utf-8') as f:
        rows = [dict(r) for r in csv.DictReader(f) if r.get('run_id')]
    for r in rows:
        r['figure_rel'] = _fig_rel(r.get('figure_file', ''))
    return rows


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ViSiOMlab — Simulation Log</title>
<style>
* {{ box-sizing: border-box; margin: 0; padding: 0; }}
body {{
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  background: #f4f5f7; color: #1c2230; height: 100vh; display: flex;
  flex-direction: column; overflow: hidden;
}}

/* ── Top bar ── */
header {{
  background: #ffffff; border-bottom: 1px solid #e3e6ec;
  padding: 0 18px; display: flex; align-items: center; gap: 16px;
  flex-shrink: 0; height: 46px;
}}
header .title {{ font-size: 0.82rem; font-weight: 600; color: #475569;
                letter-spacing: 0.06em; text-transform: uppercase; }}
header .count {{ font-size: 0.75rem; color: #94a3b8; margin-left: auto; }}
header input {{
  background: #fff; border: 1px solid #d3d7e0; border-radius: 6px;
  color: #333; font-size: 0.8rem; padding: 5px 10px; width: 260px; outline: none;
}}
header input:focus {{ border-color: #4a6cf7; }}
header input::placeholder {{ color: #aab; }}
.wide-btn {{
  background: #eef0f4; border: 1px solid #d3d7e0; color: #475569;
  font-size: 0.72rem; padding: 5px 11px; border-radius: 6px; cursor: pointer;
}}
.wide-btn.on {{ background: #2563eb; border-color: #2563eb; color: #fff; }}

/* ── Main layout ── */
.layout {{ display: flex; flex: 1; overflow: hidden; }}

/* ── Table panel ── */
.table-panel {{
  width: 34%; min-width: 0; border-right: 1px solid #e3e6ec;
  overflow-y: auto; flex-shrink: 0; transition: width 0.15s ease;
}}
.table-panel.collapsed {{ width: 0; border-right: none; overflow: hidden; }}
table {{ width: 100%; border-collapse: collapse; font-size: 0.75rem; }}
thead th {{
  background: #eef0f4; color: #6b7280; font-size: 0.68rem; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.05em;
  padding: 8px 10px; text-align: left; position: sticky; top: 0; z-index: 1;
  border-bottom: 1px solid #e3e6ec;
}}
tbody tr {{
  border-bottom: 1px solid #edeff3; cursor: pointer;
  transition: background 0.1s;
}}
tbody tr:hover {{ background: #eef3ff; }}
tbody tr.active {{ background: #e4ecff; border-left: 3px solid #4a6cf7; }}
tbody tr.active td:first-child {{ padding-left: 7px; }}
td {{ padding: 7px 10px; vertical-align: top; color: #475569; }}
td.ts    {{ color: #94a3b8; white-space: nowrap; font-size: 0.7rem; }}
td.prompt-cell {{
  color: #1f2937; max-width: 260px; overflow: hidden;
  text-overflow: ellipsis; white-space: nowrap;
}}
td.title-cell {{ color: #64748b; font-size: 0.7rem; max-width: 180px;
                overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
.mode-badge {{
  display: inline-block; font-size: 0.62rem; font-weight: 700; padding: 1px 6px;
  border-radius: 8px; white-space: nowrap; text-transform: uppercase;
  letter-spacing: 0.04em;
}}
.mode-single     {{ background: #dbeafe; color: #1d4ed8; }}
.mode-comparison {{ background: #dcfce7; color: #15803d; }}
.ok-dot {{
  display: inline-block; width: 7px; height: 7px; border-radius: 50%; margin-top: 4px;
}}
.ok-true  {{ background: #22c55e; }}
.ok-false {{ background: #ef4444; }}
.ok-null  {{ background: #cbd0d8; }}
tr.hidden {{ display: none; }}
td.fav-cell, td.feat-cell {{ text-align: center; cursor: pointer; font-size: 0.92rem;
  user-select: none; width: 30px; }}
td.fav-cell:hover, td.feat-cell:hover {{ background: #fff7e6; }}
.fav-on  {{ color: #f0b429; }}
.feat-on {{ color: #2563eb; }}
.fav-off, .feat-off {{ color: #cbd0d8; }}

/* ── Multi-select ── */
td.sel-cell, th.sel-cell {{ text-align: center; width: 26px; padding-left: 6px;
  padding-right: 0; cursor: default; }}
td.sel-cell input, th.sel-cell input {{ cursor: pointer; }}
tbody tr.selected {{ background: #eaf1ff; }}
tbody tr.selected:hover {{ background: #e0eaff; }}
tbody tr.selected.active {{ background: #dbe6ff; }}
#bulkbar {{
  display: none; align-items: center; gap: 8px; flex-wrap: wrap;
  background: #1f2937; color: #e5e7eb; padding: 7px 16px; flex-shrink: 0;
  border-bottom: 1px solid #111827; font-size: 0.76rem;
}}
#bulkbar.show {{ display: flex; }}
#bulkbar .bulk-count {{ font-weight: 600; margin-right: 6px; }}
#bulkbar .bulk-msg {{ margin-left: auto; color: #9ca3af; font-size: 0.72rem; }}
.bulk-btn {{
  background: #374151; border: 1px solid #4b5563; color: #e5e7eb;
  font-size: 0.74rem; padding: 4px 10px; border-radius: 6px; cursor: pointer;
}}
.bulk-btn:hover {{ border-color: #9ca3af; }}
.bulk-btn.danger:hover {{ border-color: #f87171; color: #fecaca; background: #3f2222; }}
.bulk-btn.warn {{ border-color: #b45309; color: #fcd34d; }}
.bulk-btn.warn:hover {{ border-color: #f59e0b; color: #fde68a; background: #3a2c10; }}
.bulk-btn:disabled {{ opacity: 0.5; cursor: not-allowed; }}
.bulk-sep {{ width: 1px; height: 18px; background: #4b5563; margin: 0 4px; }}

/* ── Figure panel ── */
.figure-panel {{
  flex: 1; min-width: 0; overflow-y: auto; padding: 14px 16px;
  display: flex; flex-direction: column; gap: 14px;
}}
.no-selection {{
  flex: 1; display: flex; align-items: center; justify-content: center;
  color: #aab; font-size: 0.85rem; text-align: center; line-height: 1.8;
}}
.figure-panel img {{
  width: 100%; border-radius: 6px; border: 1px solid #e3e6ec;
  display: block; background: #fff;
}}
.detail-grid {{
  display: grid; grid-template-columns: 100px 1fr; gap: 4px 12px;
  font-size: 0.75rem;
}}
.detail-grid .lbl {{ color: #94a3b8; text-transform: uppercase;
                    font-size: 0.65rem; letter-spacing: 0.05em; padding-top: 2px; }}
.detail-grid .val {{ color: #334155; word-break: break-word; }}
.detail-grid .val.prompt-val {{ color: #0f172a; font-size: 0.8rem; line-height: 1.5; }}
.feedback-val {{ color: #64748b; font-style: italic; }}

/* ── Interactive figure ── */
.fig-wrap {{ display: flex; flex-direction: column; gap: 8px; }}
.fig-toolbar {{ display: flex; gap: 6px; align-items: center; }}
.fig-btn {{
  background: #eef0f4; border: 1px solid #d3d7e0; color: #475569;
  font-size: 0.68rem; padding: 3px 10px; border-radius: 5px; cursor: pointer;
}}
.fig-btn.on {{ background: #2563eb; border-color: #2563eb; color: #fff; }}
.fig-btn:disabled {{ opacity: 0.4; cursor: not-allowed; }}
.plot-host {{
  width: 100%; background: #ffffff; border: 1px solid #e3e6ec;
  border-radius: 6px; padding: 6px 4px;
}}
.plot-host .u-legend {{ color: #475569; }}

/* ── Admin controls (favorite / note / delete / timing) ── */
.admin-bar {{ display: flex; gap: 8px; align-items: center; flex-wrap: wrap; }}
.adm-btn {{
  background: #eef0f4; border: 1px solid #d3d7e0; color: #475569;
  font-size: 0.74rem; padding: 5px 12px; border-radius: 6px; cursor: pointer;
}}
.adm-btn:hover {{ border-color: #2563eb; color: #1c2230; }}
.adm-btn.fav.on {{ background: #fff7e6; border-color: #f0b429; color: #8a6d1a; }}
.adm-btn.danger:hover {{ border-color: #dc2626; color: #dc2626; background: #fef2f2; }}
.note-box {{
  width: 100%; background: #fff; border: 1px solid #d3d7e0; border-radius: 6px;
  color: #1c2230; font-size: 0.82rem; padding: 8px 10px; resize: vertical;
  min-height: 48px; outline: none; font-family: inherit; margin-top: 4px;
}}
.note-box:focus {{ border-color: #2563eb; }}
.adm-saved {{ font-size: 0.72rem; color: #16a34a; }}
.adm-warn {{ font-size: 0.72rem; color: #b45309; background: #fffbeb;
  border: 1px solid #fde68a; border-radius: 6px; padding: 6px 9px; }}
.fav-star {{ color: #f0b429; }}
.timing-val {{ font-family: monospace; font-size: 0.72rem; color: #475569; }}
.adm-narrative {{ font-size: 0.84rem; color: #334155; line-height: 1.55;
  background: #f6f8fa; border: 1px solid #e3e6ec; border-radius: 8px; padding: 10px 12px; }}
.adm-narr-label {{ font-size: 0.62rem; text-transform: uppercase; letter-spacing: 0.05em;
  color: #94a3b8; margin-bottom: 4px; }}
header .token-input {{
  background: #fff; border: 1px solid #d3d7e0; border-radius: 6px;
  color: #333; font-size: 0.72rem; padding: 4px 8px; width: 150px; outline: none;
}}
</style>
<link rel="stylesheet" href="/vendor/uPlot.min.css">
<script src="/vendor/uPlot.iife.min.js"></script>
<script src="/plotspec.js"></script>
</head>
<body>

<header>
  <span class="title">Simulation Log</span>
  <input id="search" placeholder="Filter by prompt…" oninput="applyFilter()">
  <button id="favFilterBtn" class="wide-btn" onclick="toggleFavFilter()" title="Show only favorites">★ Favorites</button>
  <button id="featFilterBtn" class="wide-btn" onclick="toggleFeatFilter()" title="Show only featured">◆ Featured</button>
  <button id="wideBtn" class="wide-btn" onclick="toggleWide()" title="Hide the list to give plots full width">⤢ Wide plots</button>
  <a class="wide-btn" href="/collections.html" style="text-decoration:none" title="Curated lists of simulations (lectures, disease sets)">▤ Collections</a>
  <input id="adminToken" class="token-input" placeholder="admin token (if set)"
         oninput="localStorage.setItem('oculomotor_admin_token', this.value)">
  <span class="count" id="count"></span>
</header>

<div id="bulkbar">
  <span class="bulk-count" id="bulkCount">0 selected</span>
  <button class="bulk-btn" onclick="bulkRerun('sim')"
          title="Re-simulate from each stored scenario — no LLM, free/local">↻ Rerun sim</button>
  <button class="bulk-btn warn" onclick="bulkRerun('prompt')"
          title="Re-run each prompt through the LLM, then simulate — costs API tokens">↻ Rerun prompt+sim</button>
  <button class="bulk-btn" onclick="bulkAddToCollection()"
          title="Add the selected runs to a curated collection / list">➕ Add to collection</button>
  <span class="bulk-sep"></span>
  <button class="bulk-btn" onclick="bulkFavorite(true)">★ Favorite</button>
  <button class="bulk-btn" onclick="bulkFavorite(false)">☆ Unfavorite</button>
  <button class="bulk-btn" onclick="bulkFeatured(true)">◆ Feature</button>
  <button class="bulk-btn" onclick="bulkFeatured(false)">◇ Unfeature</button>
  <button class="bulk-btn danger" onclick="bulkDelete()">🗑 Delete</button>
  <button class="bulk-btn" onclick="clearSelection()">Clear</button>
  <span class="bulk-msg" id="bulkMsg"></span>
</div>

<div class="layout">
  <div class="table-panel">
    <table id="log">
      <thead>
        <tr>
          <th class="sel-cell"><input type="checkbox" id="selAll" title="Select all visible" onclick="toggleSelectAll(this)"></th>
          <th style="width:90px">Date</th>
          <th style="width:76px">Mode</th>
          <th style="width:30px" title="Favorite — shows in the gallery">★</th>
          <th style="width:30px" title="Featured — front-page example">◆</th>
          <th>Prompt</th>
          <th style="width:130px">Title</th>
          <th style="width:22px"></th>
        </tr>
      </thead>
      <tbody id="tbody"></tbody>
    </table>
  </div>

  <div class="figure-panel" id="panel">
    <div class="no-selection">← select a row to preview the figure</div>
  </div>
</div>

<script>
// ── Embedded data ─────────────────────────────────────────────────────────────
const ALL_ROWS = {rows_json};

// ── Format timestamp ─────────────────────────────────────────────────────────
function fmtTs(ts) {{
  if (!ts) return '—';
  const m = ts.match(/(\\d{{4}})-(\\d{{2}})-(\\d{{2}})T(\\d{{2}}):(\\d{{2}})/);
  if (!m) return ts.slice(0, 10);
  return `${{m[1]}}-${{m[2]}}-${{m[3]}} ${{m[4]}}:${{m[5]}}`;
}}

// ── State ────────────────────────────────────────────────────────────────────
let rows = [];
let activeIdx = null;
const selected = new Set();   // run_ids checked for bulk actions

// ── Render table ─────────────────────────────────────────────────────────────
function renderTable(data) {{
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  rows = [...data].reverse();  // newest first
  rows.forEach((r, i) => {{
    const tr = document.createElement('tr');
    tr.dataset.idx = i;
    tr.innerHTML = `
      <td class="sel-cell" onclick="event.stopPropagation()"><input type="checkbox" class="row-check" onclick="onRowCheck(event, ${{i}})"></td>
      <td class="ts">${{fmtTs(r.timestamp)}}</td>
      <td><span class="mode-badge mode-${{r.mode}}">${{r.mode || '—'}}</span></td>
      <td class="fav-cell" title="Toggle favorite" onclick="event.stopPropagation(); toggleFavorite('${{r.run_id}}')">${{isFav(r) ? '<span class="fav-on">★</span>' : '<span class="fav-off">☆</span>'}}</td>
      <td class="feat-cell" title="Toggle featured" onclick="event.stopPropagation(); toggleFeatured('${{r.run_id}}')">${{isFeatured(r) ? '<span class="feat-on">◆</span>' : '<span class="feat-off">◇</span>'}}</td>
      <td class="prompt-cell" title="${{esc(r.prompt)}}">${{esc(r.prompt)}}</td>
      <td class="title-cell" title="${{esc(r.title)}}">${{esc(r.title)}}</td>
      <td><span class="ok-dot ok-${{r.looks_correct === 'True' ? 'true' : r.looks_correct === 'False' ? 'false' : 'null'}}"></span></td>
    `;
    tr.addEventListener('click', () => selectRow(i, tr));
    tbody.appendChild(tr);
  }});
  updateCount();
  // Re-apply multi-select highlight for selected rows that still exist
  document.querySelectorAll('#tbody tr').forEach((tr, i) => {{
    if (rows[i] && selected.has(rows[i].run_id)) {{
      tr.classList.add('selected');
      const cb = tr.querySelector('.row-check'); if (cb) cb.checked = true;
    }}
  }});
  if (typeof updateBulkBar === 'function') updateBulkBar();
}}

function esc(s) {{
  return (s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}}

// ── Admin helpers (favorite / note / delete) ──────────────────────────────────
function adminToken() {{ return localStorage.getItem('oculomotor_admin_token') || ''; }}
function isFav(r) {{
  const v = (r.favorite || '').toString().toLowerCase();
  return v === 'true' || v === '1' || v === 'yes';
}}
function isFeatured(r) {{
  const v = (r.featured || '').toString().toLowerCase();
  return v === 'true' || v === '1' || v === 'yes';
}}
function fmtTiming(r) {{
  if (!r.ms_total) return '—';
  const s = x => (Number(x) / 1000).toFixed(1);
  return `${{s(r.ms_total)}}s total · LLM ${{s(r.ms_llm)}}s · sim ${{s(r.ms_sim)}}s`;
}}
function adminPost(url, body) {{
  return fetch(url, {{
    method: 'POST',
    headers: {{ 'Content-Type': 'application/json', 'X-Admin-Token': adminToken() }},
    body: JSON.stringify(body),
  }});
}}
function admMsg(t, ok=true) {{
  const e = document.getElementById('admSaved');
  if (e) {{ e.textContent = t; e.style.color = ok ? '#16a34a' : '#dc2626'; }}
}}
function updateRowMarks(runId) {{
  const i = rows.findIndex(x => x.run_id === runId);
  if (i < 0) return;
  const tr = document.querySelectorAll('#tbody tr')[i];
  if (!tr) return;
  const favC = tr.querySelector('.fav-cell');
  if (favC)  favC.innerHTML  = isFav(rows[i])      ? '<span class="fav-on">★</span>'  : '<span class="fav-off">☆</span>';
  const featC = tr.querySelector('.feat-cell');
  if (featC) featC.innerHTML = isFeatured(rows[i]) ? '<span class="feat-on">◆</span>' : '<span class="feat-off">◇</span>';
}}
async function toggleFavorite(runId) {{
  const r = rows.find(x => x.run_id === runId); if (!r) return;
  const fav = !isFav(r);
  try {{
    const resp = await adminPost('/admin/favorite', {{ run_id: runId, favorite: fav }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    r.favorite = fav ? 'True' : '';
    const ar = ALL_ROWS.find(x => x.run_id === runId); if (ar) ar.favorite = r.favorite;
    const b = document.getElementById('favBtn');
    if (b) {{ b.classList.toggle('on', fav); b.textContent = fav ? '★ Favorited' : '☆ Favorite'; }}
    updateRowMarks(runId);
    applyFilter();
    admMsg(fav ? 'Marked favorite — will appear in the gallery' : 'Removed from favorites');
  }} catch (e) {{ admMsg('Error: ' + e.message, false); }}
}}
async function toggleFeatured(runId) {{
  const r = rows.find(x => x.run_id === runId); if (!r) return;
  const feat = !isFeatured(r);
  try {{
    const resp = await adminPost('/admin/featured', {{ run_id: runId, featured: feat }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    r.featured = feat ? 'True' : '';
    const ar = ALL_ROWS.find(x => x.run_id === runId); if (ar) ar.featured = r.featured;
    const b = document.getElementById('featBtn');
    if (b) {{ b.classList.toggle('on', feat); b.textContent = feat ? '◆ Featured' : '◇ Feature'; }}
    updateRowMarks(runId);
    applyFilter();
    admMsg(feat ? 'Featured — will appear as a front-page example' : 'Removed from front-page examples');
  }} catch (e) {{ admMsg('Error: ' + e.message, false); }}
}}
async function saveNote(runId) {{
  const box = document.getElementById('noteBox'); if (!box) return;
  try {{
    const resp = await adminPost('/admin/note', {{ run_id: runId, note: box.value }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const r = rows.find(x => x.run_id === runId); if (r) r.note = box.value;
    const ar = ALL_ROWS.find(x => x.run_id === runId); if (ar) ar.note = box.value;
    admMsg('Note saved');
  }} catch (e) {{ admMsg('Error: ' + e.message, false); }}
}}
async function deleteRun(runId) {{
  if (!confirm('Delete this run permanently? Removes its data, figure, and log entry.')) return;
  try {{
    const resp = await fetch('/runs/' + runId, {{
      method: 'DELETE', headers: {{ 'X-Admin-Token': adminToken() }},
    }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const ai = ALL_ROWS.findIndex(x => x.run_id === runId);
    if (ai >= 0) ALL_ROWS.splice(ai, 1);
    renderTable(ALL_ROWS);
    document.getElementById('panel').innerHTML =
      '<div class="no-selection">Deleted. ← select another run</div>';
  }} catch (e) {{ alert('Delete failed: ' + e.message); }}
}}

// ── Select row → show figure ─────────────────────────────────────────────────
function selectRow(idx, tr) {{
  document.querySelectorAll('#tbody tr').forEach(r => r.classList.remove('active'));
  tr.classList.add('active');
  activeIdx = idx;
  const r = rows[idx];
  const panel = document.getElementById('panel');
  panel.innerHTML = '';

  const figWrap = document.createElement('div');
  figWrap.className = 'fig-wrap';
  panel.appendChild(figWrap);

  const narr = document.createElement('div');
  narr.className = 'adm-narrative';
  narr.style.display = 'none';
  panel.appendChild(narr);

  // Admin controls: favorite / delete / note
  const admin = document.createElement('div');
  admin.style.cssText = 'display:flex;flex-direction:column;gap:8px;';
  admin.innerHTML = `
    <div class="admin-bar">
      <button class="adm-btn fav ${{isFav(r) ? 'on' : ''}}" id="favBtn"
              onclick="toggleFavorite('${{r.run_id}}')">${{isFav(r) ? '★ Favorited' : '☆ Favorite'}}</button>
      <button class="adm-btn fav ${{isFeatured(r) ? 'on' : ''}}" id="featBtn"
              title="Featured runs appear as curated examples on the front page (a subset of favorites)"
              onclick="toggleFeatured('${{r.run_id}}')">${{isFeatured(r) ? '◆ Featured' : '◇ Feature'}}</button>
      <button class="adm-btn danger" onclick="deleteRun('${{r.run_id}}')">🗑 Delete</button>
      <span class="adm-saved" id="admSaved"></span>
    </div>
    ${{r.has_sidecar ? '' : '<div class="adm-warn">No interactive data (older run) - favoriting will NOT show it in the gallery. Re-run it on this server to generate data.</div>'}}
    <textarea class="note-box" id="noteBox" placeholder="Note / tag…">${{esc(r.note || '')}}</textarea>
    <div><button class="adm-btn" onclick="saveNote('${{r.run_id}}')">Save note</button></div>
  `;
  panel.appendChild(admin);

  const details = document.createElement('div');
  details.className = 'detail-grid';
  const ok = r.looks_correct === 'True'  ? '✓ looks correct'
           : r.looks_correct === 'False' ? '✗ incorrect' : '—';
  details.innerHTML = `
    <span class="lbl">Prompt</span>
    <span class="val prompt-val">${{esc(r.prompt)}}</span>
    <span class="lbl">Title</span>
    <span class="val">${{esc(r.title)}}</span>
    <span class="lbl">Mode</span>
    <span class="val"><span class="mode-badge mode-${{r.mode}}">${{r.mode||'—'}}</span></span>
    <span class="lbl">Date</span>
    <span class="val">${{fmtTs(r.timestamp)}}</span>
    <span class="lbl">Timing</span>
    <span class="val timing-val">${{fmtTiming(r)}}</span>
    <span class="lbl">Version</span>
    <span class="val">${{esc(r.version)}}</span>
    <span class="lbl">Run ID</span>
    <span class="val" style="font-family:monospace;font-size:0.68rem;color:#555">${{esc(r.run_id)}}</span>
    <span class="lbl">Correct</span>
    <span class="val">${{ok}}</span>
    ${{r.feedback ? `<span class="lbl">Feedback</span><span class="val feedback-val">${{esc(r.feedback)}}</span>` : ''}}
  `;
  panel.appendChild(details);

  loadSidecarInto(r, figWrap, narr);
}}

// ── Figure + narrative: one sidecar fetch feeds both ──────────────────────────
let figHandle = null;

async function loadSidecarInto(r, figWrap, narrEl) {{
  let payload = null;
  if (r.has_sidecar) {{
    try {{
      const resp = await fetch(`data/${{r.run_id}}.json`);
      if (resp.ok) payload = await resp.json();
    }} catch (e) {{}}
  }}
  renderFigure(r, figWrap, payload);
  const text = payload && (payload.narrative
    || (payload.detail && payload.detail.narrative));
  if (text && narrEl) {{
    narrEl.innerHTML = `<div class="adm-narr-label">Narrative</div>${{esc(text)}}`;
    narrEl.style.display = 'block';
  }}
}}

function renderFigure(r, figWrap, payload) {{
  if (figHandle) {{ try {{ figHandle.destroy(); }} catch (e) {{}} figHandle = null; }}

  const bar  = document.createElement('div'); bar.className = 'fig-toolbar';
  const host = document.createElement('div'); host.className = 'plot-host';
  const img  = document.createElement('img');
  if (r.figure_rel) img.src = r.figure_rel;
  img.alt = r.title || 'simulation figure';

  const bInter = document.createElement('button');
  bInter.className = 'fig-btn'; bInter.textContent = 'Interactive';
  const bImg = document.createElement('button');
  bImg.className = 'fig-btn'; bImg.textContent = 'Image';

  function showImg()   {{ host.style.display = 'none'; img.style.display = 'block';
                          bImg.classList.add('on'); bInter.classList.remove('on'); }}
  function showInter() {{ img.style.display = 'none'; host.style.display = 'block';
                          bInter.classList.add('on'); bImg.classList.remove('on'); }}
  bImg.onclick   = showImg;
  bInter.onclick = () => {{ if (!bInter.disabled) showInter(); }};

  bar.appendChild(bInter); bar.appendChild(bImg);
  figWrap.appendChild(bar); figWrap.appendChild(host); figWrap.appendChild(img);

  showImg();   // default until the spec loads

  const canInteractive = (typeof uPlot !== 'undefined') && window.PlotSpec
                         && payload && payload.plot_spec;
  if (!canInteractive) {{ bInter.disabled = true; return; }}

  try {{
    showInter();                                  // visible before render → correct width
    figHandle = window.PlotSpec.render(host, payload.plot_spec);
  }} catch (e) {{
    bInter.disabled = true; showImg();            // keep the PNG fallback
  }}
}}

// ── Wide-plots toggle (collapse the list) ─────────────────────────────────────
function toggleWide() {{
  const tp = document.querySelector('.table-panel');
  const btn = document.getElementById('wideBtn');
  const on = tp.classList.toggle('collapsed');
  btn.classList.toggle('on', on);
  // let the transition finish, then resize the uPlot charts to the new width
  setTimeout(() => window.dispatchEvent(new Event('resize')), 180);
}}

// ── Filter (search text + favorite/featured toggles) ──────────────────────────
let favOnly = false, featOnly = false;

function toggleFavFilter() {{
  favOnly = !favOnly;
  document.getElementById('favFilterBtn').classList.toggle('on', favOnly);
  applyFilter();
}}
function toggleFeatFilter() {{
  featOnly = !featOnly;
  document.getElementById('featFilterBtn').classList.toggle('on', featOnly);
  applyFilter();
}}

function applyFilter() {{
  const q = (document.getElementById('search').value || '').toLowerCase();
  let visible = 0;
  document.querySelectorAll('#tbody tr').forEach((tr, i) => {{
    const r = rows[i];
    const textMatch = !q
      || (r.prompt||'').toLowerCase().includes(q)
      || (r.title||'').toLowerCase().includes(q)
      || (r.mode||'').toLowerCase().includes(q);
    const match = textMatch && (!favOnly || isFav(r)) && (!featOnly || isFeatured(r));
    tr.classList.toggle('hidden', !match);
    if (match) visible++;
  }});
  document.getElementById('count').textContent = `${{visible}} runs`;
}}

function updateCount() {{
  document.getElementById('count').textContent = `${{rows.length}} runs`;
}}

// ── Multi-select (bulk favorite / feature / delete) ───────────────────────────
let lastCheckedIdx = null;

function onRowCheck(ev, idx) {{
  ev.stopPropagation();
  const on = ev.target.checked;
  if (ev.shiftKey && lastCheckedIdx !== null) {{
    const a = Math.min(lastCheckedIdx, idx), b = Math.max(lastCheckedIdx, idx);
    const trs = document.querySelectorAll('#tbody tr');
    for (let k = a; k <= b; k++) {{
      if (trs[k] && !trs[k].classList.contains('hidden')) setRowSelected(k, on);
    }}
  }} else {{
    setRowSelected(idx, on);
  }}
  lastCheckedIdx = idx;
  updateBulkBar();
}}

function setRowSelected(idx, on) {{
  const r = rows[idx]; if (!r) return;
  const tr = document.querySelectorAll('#tbody tr')[idx]; if (!tr) return;
  const cb = tr.querySelector('.row-check'); if (cb) cb.checked = on;
  tr.classList.toggle('selected', on);
  if (on) selected.add(r.run_id); else selected.delete(r.run_id);
}}

function toggleSelectAll(cb) {{
  document.querySelectorAll('#tbody tr').forEach((tr, i) => {{
    if (!tr.classList.contains('hidden')) setRowSelected(i, cb.checked);
  }});
  updateBulkBar();
}}

function clearSelection() {{
  selected.clear();
  document.querySelectorAll('#tbody tr').forEach((tr) => {{
    tr.classList.remove('selected');
    const cb = tr.querySelector('.row-check'); if (cb) cb.checked = false;
  }});
  const sa = document.getElementById('selAll'); if (sa) sa.checked = false;
  updateBulkBar();
}}

function updateBulkBar() {{
  const n = selected.size;
  document.getElementById('bulkbar').classList.toggle('show', n > 0);
  document.getElementById('bulkCount').textContent = `${{n}} selected`;
  if (!n) bulkMsg('');
}}

function bulkMsg(t) {{ const m = document.getElementById('bulkMsg'); if (m) m.textContent = t; }}

async function bulkFavorite(fav) {{
  const ids = [...selected]; if (!ids.length) return;
  bulkMsg('Working…');
  const res = await Promise.allSettled(ids.map(async (runId) => {{
    const resp = await adminPost('/admin/favorite', {{ run_id: runId, favorite: fav }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const r = rows.find(x => x.run_id === runId); if (r) r.favorite = fav ? 'True' : '';
    const ar = ALL_ROWS.find(x => x.run_id === runId); if (ar) ar.favorite = fav ? 'True' : '';
  }}));
  const ok = res.filter(x => x.status === 'fulfilled').length;
  ids.forEach(updateRowMarks);
  applyFilter();
  bulkMsg(`${{fav ? 'Favorited' : 'Unfavorited'}} ${{ok}}/${{ids.length}}`);
}}

async function bulkFeatured(feat) {{
  const ids = [...selected]; if (!ids.length) return;
  bulkMsg('Working…');
  const res = await Promise.allSettled(ids.map(async (runId) => {{
    const resp = await adminPost('/admin/featured', {{ run_id: runId, featured: feat }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const r = rows.find(x => x.run_id === runId); if (r) r.featured = feat ? 'True' : '';
    const ar = ALL_ROWS.find(x => x.run_id === runId); if (ar) ar.featured = feat ? 'True' : '';
  }}));
  const ok = res.filter(x => x.status === 'fulfilled').length;
  ids.forEach(updateRowMarks);
  applyFilter();
  bulkMsg(`${{feat ? 'Featured' : 'Unfeatured'}} ${{ok}}/${{ids.length}}`);
}}

async function bulkDelete() {{
  const ids = [...selected]; if (!ids.length) return;
  if (!confirm(`Delete ${{ids.length}} run(s) permanently? Removes their data, figures, and log entries.`)) return;
  bulkMsg('Deleting…');
  const res = await Promise.allSettled(ids.map(async (runId) => {{
    const resp = await fetch('/runs/' + runId, {{ method: 'DELETE', headers: {{ 'X-Admin-Token': adminToken() }} }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    const ai = ALL_ROWS.findIndex(x => x.run_id === runId); if (ai >= 0) ALL_ROWS.splice(ai, 1);
  }}));
  const ok = res.filter(x => x.status === 'fulfilled').length;
  selected.clear();
  renderTable(ALL_ROWS);
  applyFilter();
  document.getElementById('panel').innerHTML =
    `<div class="no-selection">Deleted ${{ok}} run(s). ← select another</div>`;
}}

// ── Bulk rerun (sequential — the server runs one sim at a time) ───────────────
// kind='sim'    : re-simulate from each stored scenario (POST /rerun/{{id}}), no LLM, free.
// kind='prompt' : re-run each prompt through the LLM then simulate (POST /run), costs tokens.
let _rerunning = false;
async function bulkRerun(kind) {{
  if (_rerunning) return;
  const ids = [...selected];
  if (!ids.length) return;

  if (kind === 'prompt' && ids.length > 1) {{
    const okGo = confirm(
      `Re-run the PROMPT through the LLM for ${{ids.length}} runs?\\n\\n` +
      `That makes ${{ids.length}} LLM API calls — this COSTS TOKENS and the results ` +
      `may differ from the originals.\\n\\n"Rerun sim" re-simulates for free without the LLM.\\n\\nProceed?`);
    if (!okGo) return;
  }}

  _rerunning = true;
  document.querySelectorAll('#bulkbar .bulk-btn').forEach(b => b.disabled = true);
  const label = kind === 'prompt' ? 'prompt+sim' : 'sim';
  const token = adminToken();
  let ok = 0, fail = 0;
  for (let i = 0; i < ids.length; i++) {{
    bulkMsg(`Re-running ${{label}} ${{i + 1}}/${{ids.length}}…` + (fail ? ` (${{fail}} failed)` : ''));
    try {{
      let resp;
      if (kind === 'prompt') {{
        const r = rows.find(x => x.run_id === ids[i]) || ALL_ROWS.find(x => x.run_id === ids[i]);
        const prompt = r && r.prompt;
        if (!prompt) throw new Error('no stored prompt');
        resp = await fetch('/run', {{
          method: 'POST', headers: {{ 'Content-Type': 'application/json' }},
          body: JSON.stringify({{ description: prompt }}),
        }});
      }} else {{
        resp = await fetch('/rerun/' + ids[i], {{ method: 'POST', headers: {{ 'X-Admin-Token': token }} }});
      }}
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const body = await resp.json().catch(() => ({{}}));
      if (body && body.error) throw new Error(body.error);
      ok++;
    }} catch (e) {{ fail++; }}
  }}
  bulkMsg(`Done: ${{ok}} ok${{fail ? ', ' + fail + ' failed' : ''}} — reloading to show new runs…`);
  setTimeout(() => window.location.reload(), 1000);
}}

// ── Add selected runs to a curated collection ─────────────────────────────────
async function bulkAddToCollection() {{
  const ids = [...selected]; if (!ids.length) return;
  let cols = [];
  try {{ cols = ((await (await fetch('/collections')).json()).collections) || []; }}
  catch (e) {{ bulkMsg('Could not load collections: ' + e.message); return; }}
  const menu = cols.map((c, i) => `${{i + 1}}. ${{c.title}}`).join('\\n');
  const pick = prompt(
    `Add ${{ids.length}} run(s) to which collection?\\n\\n${{menu || '(no collections yet)'}}\\n\\n0 = create a new collection`,
    cols.length ? '1' : '0');
  if (pick === null) return;
  const n = parseInt(pick, 10);
  let cid;
  if (!cols.length || n === 0) {{
    const title = prompt('New collection name:', ''); if (!title) return;
    try {{
      const resp = await fetch('/collections', {{ method: 'POST',
        headers: {{ 'Content-Type': 'application/json', 'X-Admin-Token': adminToken() }},
        body: JSON.stringify({{ title }}) }});
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      cid = (await resp.json()).id;
    }} catch (e) {{ bulkMsg('Create failed: ' + e.message); return; }}
  }} else if (n >= 1 && n <= cols.length) {{
    cid = cols[n - 1].id;
  }} else {{ return; }}
  const section = (prompt('Section name (leave blank for top level):', '') || '').trim() || null;
  try {{
    const resp = await fetch('/collections/' + cid + '/add', {{ method: 'POST',
      headers: {{ 'Content-Type': 'application/json', 'X-Admin-Token': adminToken() }},
      body: JSON.stringify({{ run_ids: ids, section }}) }});
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    bulkMsg(`Added ${{ids.length}} run(s) to the collection — open ▤ Collections to arrange them`);
  }} catch (e) {{ bulkMsg('Add failed: ' + e.message); }}
}}

// ── Init ──────────────────────────────────────────────────────────────────────
document.getElementById('adminToken').value = localStorage.getItem('oculomotor_admin_token') || '';
renderTable(ALL_ROWS);
</script>
</body>
</html>
"""


def generate(rows: list[dict] | None = None) -> None:
    if rows is None:
        rows = load_rows()
    # Annotate each row (on a copy — never mutate the caller's log dicts, which
    # are written back to the fixed-column CSV) with whether a data sidecar
    # exists, so the page only fetches interactive data when there is some.
    data_dir = _DATA_ROOT / 'data'
    annotated = []
    for r in rows:
        rr = dict(r)
        rr['has_sidecar'] = (data_dir / f"{rr.get('run_id', '')}.json").exists()
        annotated.append(rr)
    rows_json = json.dumps(annotated, ensure_ascii=False, indent=None)
    html = _HTML_TEMPLATE.format(rows_json=rows_json)
    _OUT_FILE.write_text(html, encoding='utf-8')


if __name__ == '__main__':
    rows = load_rows()
    generate(rows)
    print(f"Written {len(rows)} rows → {_OUT_FILE}")
