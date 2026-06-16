"""Local FastAPI server — natural-language oculomotor simulation.

Usage
-----
    python -X utf8 scripts/server.py
    python -X utf8 scripts/server.py --host 0.0.0.0 --port 8000

Then open http://localhost:8000 in your browser.
"""

import os
import io
import csv
import json
import time
import uuid
import base64
import argparse
import mimetypes
import traceback
import datetime
from pathlib import Path

# app.py is at src/oculomotor/server/ → the checkout's repo root is 3 dirs up.
# dev runs from the main checkout, stable from the om-stable worktree (own venv), so this
# self-locates web/ + data/ to whichever checkout's package is imported.
_REPO_ROOT = Path(__file__).resolve().parents[3]

# Serve .js (incl. ES modules like avatar.js) with a JavaScript MIME type.
# On Windows the registry often maps .js → text/plain, which browsers REJECT
# for `<script type="module">` (strict MIME checking) — breaking the 3D avatar.
mimetypes.add_type('text/javascript', '.js')
mimetypes.add_type('text/javascript', '.mjs')


from dotenv import load_dotenv
load_dotenv(_REPO_ROOT / '.env')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fastapi import FastAPI, HTTPException, Header
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from oculomotor import __version__ as _SIM_VERSION
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison
from oculomotor.llm_pipeline.run import run_scenario, run_comparison
from oculomotor.llm_pipeline.interpret import call_llm
from oculomotor.llm_pipeline.patient_builder import Patient as _PatientCls
from oculomotor.server.admin import generate as _gen_admin

# YAML schema used to enrich patient-change diffs with anatomy / disorders.
import yaml as _yaml
from importlib.resources import files as _files
_SCHEMA_PATH = _files('oculomotor.schema') / 'parameters_schema.yaml'
with _SCHEMA_PATH.open(encoding='utf-8') as _f:
    _PARAM_SCHEMA = _yaml.safe_load(_f) or {}


# ── Paths ─────────────────────────────────────────────────────────────────────

# Request database — ONE folder per checkout: server_data/ (figures, sidecars, log, admin.html).
# Each checkout owns its own, so dev (main) and stable (om-stable worktree) get SEPARATE DBs.
# Override with OCULOMOTOR_DATA (e.g. to a non-OneDrive path — OneDrive syncing an
# actively-written log can corrupt it: orphaned sidecars, vanished CSV rows).
_DATA_ROOT = Path(os.environ.get('OCULOMOTOR_DATA') or (_REPO_ROOT / 'server_data'))
_DATA_ROOT.mkdir(parents=True, exist_ok=True)

_FIGURES_DIR = _DATA_ROOT / 'server_figures'
_FIGURES_DIR.mkdir(exist_ok=True)

# Per-run JSON sidecars: metadata + library-agnostic plot spec for client-side
# rendering.  The CSV below remains the lightweight queryable index.
_DATA_DIR = _DATA_ROOT / 'data'
_DATA_DIR.mkdir(exist_ok=True)

_LOG_FILE = _DATA_ROOT / 'simulation_log.csv'

_LOG_COLUMNS = [
    'timestamp', 'run_id', 'version', 'prompt', 'mode', 'title',
    'figure_file', 'looks_correct', 'feedback',
    'favorite', 'featured', 'note',        # admin: gallery (favorite) + front-page examples (featured) + tag
    'ms_total', 'ms_llm', 'ms_sim',        # timing (debug): whole request / LLM / sim
]

# Optional shared secret guarding the admin mutation endpoints (delete / favorite
# / note).  If ADMIN_TOKEN is set in the environment, those endpoints require a
# matching X-Admin-Token header; if unset, they are open (local-dev convenience).
_ADMIN_TOKEN = os.environ.get('ADMIN_TOKEN', '').strip()


# ── In-memory state ───────────────────────────────────────────────────────────

_log_entries: dict[str, dict] = {}   # run_id → CSV row dict
_sim_cache:   dict[str, dict] = {}   # run_id → sim data arrays (numpy)


def _load_log() -> None:
    """Load existing log CSV into _log_entries on startup."""
    if _LOG_FILE.exists():
        with open(_LOG_FILE, newline='', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                _log_entries[row['run_id']] = dict(row)


def _append_log(row: dict) -> None:
    """Append one row to the log CSV and in-memory dict."""
    _log_entries[row['run_id']] = row
    write_header = not _LOG_FILE.exists()
    with open(_LOG_FILE, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        if write_header:
            w.writeheader()
        w.writerow(row)
    _gen_admin(list(_log_entries.values()))


def _rewrite_log() -> None:
    """Rewrite the full CSV (used after feedback updates)."""
    with open(_LOG_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=_LOG_COLUMNS)
        w.writeheader()
        for row in _log_entries.values():
            w.writerow(row)
    _gen_admin(list(_log_entries.values()))


# ── Per-run JSON sidecars (metadata + plot spec) ──────────────────────────────

def _data_path(run_id: str) -> Path:
    return _DATA_DIR / f'{run_id}.json'


def _write_run_json(run_id: str, payload: dict) -> None:
    """Persist a run's metadata + plot spec to outputs/data/<run_id>.json."""
    with open(_data_path(run_id), 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)


def _read_run_json(run_id: str) -> dict | None:
    """Load a persisted run sidecar, overlaying the latest feedback from the log."""
    p = _data_path(run_id)
    if not p.exists():
        return None
    with open(p, encoding='utf-8') as f:
        payload = json.load(f)
    # Keep mutable fields live: prefer the current log values over the snapshot.
    row = _log_entries.get(run_id)
    if row:
        payload['looks_correct'] = row.get('looks_correct', payload.get('looks_correct', ''))
        payload['feedback']      = row.get('feedback', payload.get('feedback', ''))
        payload['favorite']      = _truthy(row.get('favorite'))
        payload['featured']      = _truthy(row.get('featured'))
        payload['note']          = row.get('note', payload.get('note', ''))
    return payload


def _truthy(v) -> bool:
    """Interpret a CSV/string flag as a boolean."""
    return str(v).strip().lower() in ('1', 'true', 'yes', 'on')


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(title='OculomotorSim')

# Allow requests from GitHub Pages (and any other origin) so the static
# frontend can call this local backend across origins.
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['GET', 'POST', 'DELETE'],
    allow_headers=['Content-Type', 'X-Admin-Token', 'ngrok-skip-browser-warning'],
)


@app.middleware('http')
async def _force_js_mime(request, call_next):
    """Force a JS MIME type on .js/.mjs responses.

    Belt-and-suspenders over mimetypes.add_type: browsers reject
    `<script type="module">` served as text/plain (which Windows' registry
    often returns for .js), which silently breaks the 3D avatar module.
    """
    response = await call_next(request)
    path = request.url.path
    if path.endswith('.js') or path.endswith('.mjs'):
        response.headers['content-type'] = 'text/javascript; charset=utf-8'
    return response

# Load any existing log at startup, normalize it to the current columns
# (older logs predate favorite/note/timing), and write a fresh admin page.
_load_log()
_rewrite_log()


# ── Request / response models ─────────────────────────────────────────────────

class RunRequest(BaseModel):
    description: str
    model: str = 'claude-opus-4-8'


class RunResponse(BaseModel):
    image_b64:        str
    mode:             str        # 'single' or 'comparison'
    title:            str
    detail_json:      dict
    run_id:           str
    version:          str
    eye_trajectory:   dict | None = None   # single mode: one trajectory
    eye_trajectories: list | None = None   # comparison mode: one per scenario (with 'label' field)
    patient_changes:  list | None = None   # single mode: list of changed parameters w/ metadata
    plot_spec:        dict | None = None   # library-agnostic spec for client-side rendering


class FeedbackRequest(BaseModel):
    run_id:        str
    looks_correct: str | None = None   # 'correct', 'incorrect', or None (not rated)
    comment:       str = ''


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_eye_trajectory(sim_data: dict, fps: int = 60) -> dict | None:
    """Downsample per-eye position arrays for web animation.

    Returns a dict with 'fps', 'n_frames', 'duration_s', 'left', 'right'
    where left/right are lists of [yaw, pitch, roll] in degrees, rounded to
    2 decimal places.  Kept small enough to embed in the JSON response.
    """
    if sim_data is None:
        return None
    t  = np.array(sim_data['t'])
    dt = float(t[1] - t[0]) if len(t) > 1 else 0.001
    step = max(1, round(1.0 / (fps * dt)))

    L = np.array(sim_data.get('eye_pos_L', sim_data['eye_pos']))[::step]
    R = np.array(sim_data.get('eye_pos_R', sim_data['eye_pos']))[::step]
    t_ds = t[::step]

    # Integrate head velocity → head orientation (deg), then downsample
    dt_orig  = float(t[1] - t[0]) if len(t) > 1 else 0.001
    hv       = np.array(sim_data['head_vel'])            # (T, 3) deg/s
    head_pos = np.cumsum(hv * dt_orig, axis=0)           # (T, 3) deg
    head_ds  = head_pos[::step]

    ones = np.ones(len(t))
    spL = np.array(sim_data.get('scene_present_L',  ones))
    spR = np.array(sim_data.get('scene_present_R',  ones))
    tpL = np.array(sim_data.get('target_present_L', ones))
    tpR = np.array(sim_data.get('target_present_R', ones))
    # Monocular cover: prefer the explicit cover flags carried in the data (the
    # scenario's high-level intent). Fall back to inference (eye in total darkness
    # while the fellow eye sees) only for older runs / direct simulate() calls that
    # predate the explicit flags. Binocular darkness is never a cover.
    if 'cover_L' in sim_data and sim_data['cover_L'] is not None:
        cover_L = (np.array(sim_data['cover_L']) > 0.5).astype(int)[::step]
        cover_R = (np.array(sim_data['cover_R']) > 0.5).astype(int)[::step]
    else:
        cover_L = ((spL < 0.5) & (tpL < 0.5) & ((spR > 0.5) | (tpR > 0.5))).astype(int)[::step]
        cover_R = ((spR < 0.5) & (tpR < 0.5) & ((spL > 0.5) | (tpL > 0.5))).astype(int)[::step]

    # Target: world-Cartesian position relative to the head (m), + a per-frame
    # presence flag (either eye seeing the foveal target). The avatar draws a red
    # sphere at this point; presence toggles its visibility (e.g. off during OKN).
    out = dict(
        fps        = fps,
        n_frames   = int(len(t_ds)),
        duration_s = round(float(t_ds[-1]), 3),
        left    = [[round(float(v), 2) for v in row] for row in L.tolist()],
        right   = [[round(float(v), 2) for v in row] for row in R.tolist()],
        head    = [[round(float(v), 2) for v in row] for row in head_ds.tolist()],
        cover_L = cover_L.tolist(),
        cover_R = cover_R.tolist(),
    )

    if 'p_target' in sim_data and sim_data['p_target'] is not None:
        tgt = np.array(sim_data['p_target'])[::step]
        target_present = ((tpL > 0.5) | (tpR > 0.5)).astype(int)[::step]
        out['target']         = [[round(float(v), 4) for v in row] for row in tgt.tolist()]
        out['target_present'] = target_present.tolist()

    # Scene angular position (integrate scene velocity) + presence flag, for the
    # world dot-cloud: it rotates with the scene (OKN) and hides when the scene
    # is off (e.g. VOR in the dark).
    if 'scene_vel' in sim_data and sim_data['scene_vel'] is not None:
        sv = np.array(sim_data['scene_vel'])                 # (T, 3) deg/s
        scene_pos = np.cumsum(sv * dt_orig, axis=0)[::step]  # (n, 3) deg
        out['scene_pos']     = [[round(float(v), 2) for v in row] for row in scene_pos.tolist()]
        out['scene_present'] = ((spL > 0.5) | (spR > 0.5)).astype(int)[::step].tolist()

    # Head linear displacement (m, relative to start) for translational optic flow
    # of the dot-cloud as the head walks/translates through the world.
    if 'head_lin_pos' in sim_data and sim_data['head_lin_pos'] is not None:
        hlp = np.array(sim_data['head_lin_pos'])
        hlp = (hlp - hlp[0])[::step]                          # (n, 3) m from start
        out['head_lin_pos'] = [[round(float(v), 4) for v in row] for row in hlp.tolist()]

    # Per-eye prism deviation [yaw, pitch, roll] deg — carried for the avatar to
    # draw a prism/wedge later. Emitted only when a prism is actually present.
    for side in ('L', 'R'):
        key = f'prism_{side}'
        if key in sim_data and sim_data[key] is not None:
            pr = np.array(sim_data[key])
            if pr.ndim == 2 and np.any(np.abs(pr) > 1e-6):
                out[key] = [[round(float(v), 3) for v in row] for row in pr[::step].tolist()]

    return out


# ── Patient-changes diff (LLM overrides vs healthy defaults) ──────────────────

def _looks_changed(value, default) -> bool:
    """Robust equality test for scalars + lists, tolerating float drift."""
    if isinstance(value, list) or isinstance(default, list):
        v = list(value) if isinstance(value, (list, tuple)) else [value]
        d = list(default) if isinstance(default, (list, tuple)) else [default]
        if len(v) != len(d):
            return True
        return any(abs(float(a) - float(b)) > 1e-9 for a, b in zip(v, d))
    try:
        return abs(float(value) - float(default)) > 1e-9
    except (TypeError, ValueError):
        return value != default


def _format_disorders(disorders):
    if not disorders:
        return []
    return [
        {'name': d.get('name', '?'), 'value': d.get('value', ''), 'tag': d.get('tag', 'other')}
        for d in disorders
    ]


def _find_schema_entry(field_name: str) -> dict:
    """Locate the YAML entry for a Patient field, trying brain/sensory/plant."""
    for prefix in ('brain', 'sensory', 'plant'):
        key = f'{prefix}.{field_name}'
        if key in _PARAM_SCHEMA:
            return _PARAM_SCHEMA[key]
    return {}


def _build_patient_changes(patient) -> list[dict]:
    """Diff the LLM-set Patient against defaults; enrich with YAML metadata.

    Returns a list of dicts (only changed parameters) for the avatar page to
    render, mirroring the parameters.html column structure but with explicit
    default→value diff so users see exactly what the LLM tweaked.
    """
    default_patient = _PatientCls()
    changes = []
    for fname in _PatientCls.model_fields:
        try:
            value   = getattr(patient, fname)
            default = getattr(default_patient, fname)
        except AttributeError:
            continue
        if not _looks_changed(value, default):
            continue
        entry = _find_schema_entry(fname)
        changes.append({
            'name':        fname,
            'default':     default,
            'value':       value,
            'units':       entry.get('units', ''),
            'description': (entry.get('description') or '').strip(),
            'anatomy':     entry.get('anatomy', ''),
            'disorders':   _format_disorders(entry.get('disorders', [])),
            'group':       entry.get('group', 'ungrouped'),
        })
    return changes


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.get('/version')
async def version_endpoint():
    """Return the running simulator version and git commit."""
    return JSONResponse({'version': _SIM_VERSION})


@app.post('/run', response_model=RunResponse)
async def run_endpoint(req: RunRequest):
    """LLM decides single simulation or comparison; runs and returns the figure."""
    run_id = str(uuid.uuid4())
    try:
        _t0 = time.perf_counter()
        result = call_llm(req.description, model=req.model)
        ms_llm = (time.perf_counter() - _t0) * 1000.0
        _t1 = time.perf_counter()

        if isinstance(result, SimulationComparison):
            fig, cmp_sim_data_list, plot_spec = run_comparison(
                result, return_data=True, return_spec=True)
            title = result.title
            mode  = 'comparison'
            detail = result.model_dump()
            sim_data = None    # comparison CSV download not supported yet
            # Build per-scenario trajectories with short labels for avatar tabs
            eye_trajectories = []
            for scenario, sd in zip(result.scenarios, cmp_sim_data_list):
                traj = _build_eye_trajectory(sd)
                if traj is not None:
                    traj['label'] = scenario.description
                    traj['patient_changes'] = _build_patient_changes(scenario.patient)
                    eye_trajectories.append(traj)
            patient_changes = None   # per-scenario, attached on each entry
        else:
            fig, sim_data, plot_spec = run_scenario(
                result, return_data=True, return_spec=True)
            title            = result.description
            mode             = 'single'
            detail           = result.model_dump()
            eye_trajectories = []
            patient_changes  = _build_patient_changes(result.patient)

        ms_sim = (time.perf_counter() - _t1) * 1000.0   # sim + plotting + spec

        # Save figure to disk
        fig_name = f'{run_id}.png'
        fig_path = _FIGURES_DIR / fig_name
        fig.savefig(fig_path, dpi=130, bbox_inches='tight')

        # Encode for inline display
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=130, bbox_inches='tight')
        plt.close(fig)
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode('utf-8')

        # Cache sim data for download
        if sim_data is not None:
            _sim_cache[run_id] = sim_data

        ms_total = (time.perf_counter() - _t0) * 1000.0
        timing = {'total_ms': round(ms_total), 'llm_ms': round(ms_llm),
                  'sim_ms': round(ms_sim)}

        timestamp = datetime.datetime.utcnow().isoformat(timespec='seconds') + 'Z'
        eye_traj  = _build_eye_trajectory(sim_data) if mode == 'single' else None

        # Persist the full record (metadata + plot spec) FIRST, so the data
        # sidecar exists on disk BEFORE _append_log regenerates admin.html
        # (gen_admin flags runs whose sidecar is missing). Writing the log first
        # made every fresh run show as "no data" until the next mutation.
        _write_run_json(run_id, {
            'run_id':          run_id,
            'timestamp':       timestamp,
            'version':         _SIM_VERSION,
            'prompt':          req.description,
            'mode':            mode,
            'title':           title,
            'narrative':       detail.get('narrative', ''),
            'figure_rel':      f'server_figures/{fig_name}',
            'looks_correct':   '',
            'feedback':        '',
            'favorite':        False,
            'featured':        False,
            'note':            '',
            'timing':          timing,
            'patient_changes': patient_changes,
            'eye_trajectory':  eye_traj,
            'eye_trajectories': eye_trajectories if mode == 'comparison' else None,
            'plot_spec':       plot_spec,
            # Full LLM output (the structured scenario/comparison the model
            # produced, incl. narrative + every stimulus/patient/plot field).
            'detail':          detail,
        })

        # Append the log row (this regenerates admin.html) AFTER the sidecar
        # exists, so the new run is correctly shown as having data.
        _append_log({
            'timestamp':   timestamp,
            'run_id':      run_id,
            'version':     _SIM_VERSION,
            'prompt':      req.description,
            'mode':        mode,
            'title':       title,
            'figure_file': str(fig_path),
            'looks_correct': '',
            'feedback':    '',
            'favorite':    '',
            'featured':    '',
            'note':        '',
            'ms_total':    timing['total_ms'],
            'ms_llm':      timing['llm_ms'],
            'ms_sim':      timing['sim_ms'],
        })

        return RunResponse(
            image_b64        = img_b64,
            mode             = mode,
            title            = title,
            detail_json      = detail,
            run_id           = run_id,
            version          = _SIM_VERSION,
            eye_trajectory   = eye_traj,
            eye_trajectories = eye_trajectories if mode == 'comparison' else None,
            patient_changes  = patient_changes,
            plot_spec        = plot_spec,
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={'error': str(e)})


@app.post('/feedback')
async def feedback_endpoint(req: FeedbackRequest):
    """Record user feedback (looks_correct + comment) for a run."""
    if req.run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries[req.run_id]['looks_correct'] = req.looks_correct or ''
    _log_entries[req.run_id]['feedback']      = req.comment
    _rewrite_log()
    return {'status': 'ok'}


# ── Admin: curate the database (favorite / note / delete) ─────────────────────
#
# These mutate the request database.  If ADMIN_TOKEN is set in the server's
# environment, they require a matching X-Admin-Token header; otherwise they are
# open (local-dev convenience).  The admin page is intended to be used locally,
# not from the public site.

class FavoriteRequest(BaseModel):
    run_id:   str
    favorite: bool


class FeaturedRequest(BaseModel):
    run_id:   str
    featured: bool


class NoteRequest(BaseModel):
    run_id: str
    note:   str = ''


def _check_admin(token: str | None) -> None:
    if _ADMIN_TOKEN and (token or '') != _ADMIN_TOKEN:
        raise HTTPException(status_code=403, detail='Admin token required.')


def _patch_run_json(run_id: str, **fields) -> None:
    """Update selected fields in a run's sidecar (keeps it consistent with the log)."""
    p = _data_path(run_id)
    if not p.exists():
        return
    with open(p, encoding='utf-8') as f:
        payload = json.load(f)
    payload.update(fields)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)


@app.post('/admin/favorite')
async def admin_favorite(req: FavoriteRequest,
                         x_admin_token: str | None = Header(default=None)):
    """Mark/unmark a run as a favorite (favorites are what the gallery shows)."""
    _check_admin(x_admin_token)
    if req.run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries[req.run_id]['favorite'] = 'True' if req.favorite else ''
    _rewrite_log()
    _patch_run_json(req.run_id, favorite=req.favorite)
    return {'status': 'ok', 'favorite': req.favorite}


@app.post('/admin/featured')
async def admin_featured(req: FeaturedRequest,
                         x_admin_token: str | None = Header(default=None)):
    """Mark/unmark a run as FEATURED (featured runs appear as front-page examples).

    Featured is a curated subset of favorites — the paradigm-spanning examples shown
    under the prompt box. The full favorites set is the 'see more' gallery.
    """
    _check_admin(x_admin_token)
    if req.run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries[req.run_id]['featured'] = 'True' if req.featured else ''
    _rewrite_log()
    _patch_run_json(req.run_id, featured=req.featured)
    return {'status': 'ok', 'featured': req.featured}


@app.post('/admin/note')
async def admin_note(req: NoteRequest,
                     x_admin_token: str | None = Header(default=None)):
    """Attach a free-text note/tag to a run."""
    _check_admin(x_admin_token)
    if req.run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries[req.run_id]['note'] = req.note
    _rewrite_log()
    _patch_run_json(req.run_id, note=req.note)
    return {'status': 'ok', 'note': req.note}


@app.delete('/runs/{run_id}')
async def admin_delete(run_id: str,
                       x_admin_token: str | None = Header(default=None)):
    """Delete a run entirely: log row + data sidecar + figure + cached data."""
    _check_admin(x_admin_token)
    if run_id not in _log_entries:
        raise HTTPException(status_code=404, detail='run_id not found')
    _log_entries.pop(run_id, None)
    _sim_cache.pop(run_id, None)
    try:
        _data_path(run_id).unlink(missing_ok=True)
    except Exception:
        pass
    try:
        (_FIGURES_DIR / f'{run_id}.png').unlink(missing_ok=True)
    except Exception:
        pass
    _rewrite_log()
    return {'status': 'deleted', 'run_id': run_id}


@app.get('/runs')
async def runs_index_endpoint(correct_only: bool = False, favorites_only: bool = False,
                              featured_only: bool = False):
    """Return the index of past runs (newest first) for browsing.

    Only runs with a persisted data sidecar are listed — those can be
    re-rendered client-side via /run/{run_id}/data.  Same backend the public
    site already calls for /run, so the static frontend can browse the database
    cross-origin (CORS is open).

    The public gallery calls this with ``favorites_only=true`` so only curated
    runs are shown; the admin calls it without filters to see everything.
    """
    rows = []
    for run_id, row in _log_entries.items():
        if not _data_path(run_id).exists():
            continue
        lc   = row.get('looks_correct', '')
        fav  = _truthy(row.get('favorite'))
        feat = _truthy(row.get('featured'))
        if correct_only and lc not in ('True', 'correct'):
            continue
        if favorites_only and not fav:
            continue
        if featured_only and not feat:
            continue
        rows.append({
            'run_id':        run_id,
            'timestamp':     row.get('timestamp', ''),
            'prompt':        row.get('prompt', ''),
            'title':         row.get('title', ''),
            'mode':          row.get('mode', ''),
            'version':       row.get('version', ''),
            'looks_correct': lc,
            'favorite':      fav,
            'featured':      feat,
            'note':          row.get('note', ''),
            'ms_total':      row.get('ms_total', ''),
            'ms_llm':        row.get('ms_llm', ''),
            'ms_sim':        row.get('ms_sim', ''),
        })
    rows.sort(key=lambda r: r['timestamp'], reverse=True)
    return JSONResponse(rows)


@app.get('/run/{run_id}/data')
async def run_data_endpoint(run_id: str):
    """Return a persisted run (metadata + library-agnostic plot spec).

    Used by the website/admin to re-render a past request client-side without
    the matplotlib PNG.  404 if the run has no sidecar (e.g. pre-dates this
    feature — fall back to the stored figure).
    """
    payload = _read_run_json(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail='No data for this run_id.')
    return JSONResponse(payload)


@app.get('/download/{run_id}')
async def download_endpoint(run_id: str):
    """Return simulation data as a CSV file (eye + stimulus arrays)."""
    if run_id not in _sim_cache:
        raise HTTPException(status_code=404,
                            detail='Data not available (comparison runs or expired cache).')
    data = _sim_cache[run_id]

    # Build CSV in memory
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        't_s',
        'eye_pos_yaw_deg', 'eye_pos_pitch_deg', 'eye_pos_roll_deg',
        'eye_vel_yaw_degs', 'eye_vel_pitch_degs', 'eye_vel_roll_degs',
        'head_vel_yaw_degs', 'head_vel_pitch_degs', 'head_vel_roll_degs',
        'scene_vel_yaw_degs', 'scene_vel_pitch_degs', 'scene_vel_roll_degs',
        'target_vel_yaw_degs', 'target_vel_pitch_degs', 'target_vel_roll_degs',
    ])

    t          = np.array(data['t'])
    eye_pos    = np.array(data['eye_pos'])
    eye_vel    = np.array(data['eye_vel'])
    head_vel   = np.array(data['head_vel'])
    scene_vel  = np.array(data['scene_vel'])
    target_vel = np.array(data['target_vel'])

    for i in range(len(t)):
        writer.writerow([
            f'{t[i]:.4f}',
            *[f'{v:.4f}' for v in eye_pos[i]],
            *[f'{v:.4f}' for v in eye_vel[i]],
            *[f'{v:.4f}' for v in head_vel[i]],
            *[f'{v:.4f}' for v in scene_vel[i]],
            *[f'{v:.4f}' for v in target_vel[i]],
        ])

    csv_bytes = buf.getvalue().encode('utf-8')
    filename  = f'oculomotorsim_{run_id[:8]}.csv'

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{filename}"'},
    )


# ── Admin page ───────────────────────────────────────────────────────────────

from fastapi.responses import RedirectResponse as _Redirect

@app.get('/admin')
async def admin_redirect():
    """Redirect /admin → /outputs/admin.html (served via HTTP, no file:// issues)."""
    return _Redirect('/outputs/admin.html')


# ── Static file mounts (specific paths before the catch-all /) ────────────────

app.mount('/outputs', StaticFiles(directory=str(_DATA_ROOT)), name='outputs')

# Frontend (web/): this checkout's web/ by default; override with OCULOMOTOR_WEB.
_WEB_DIR = Path(os.environ.get('OCULOMOTOR_WEB') or (_REPO_ROOT / 'web'))
app.mount('/', StaticFiles(directory=str(_WEB_DIR), html=True), name='frontend')


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='0.0.0.0',
                        help='Bind host (default 0.0.0.0 = all interfaces)')
    parser.add_argument('--port', type=int, default=8000)
    args = parser.parse_args()

    print(f"\n  OculomotorSim {_SIM_VERSION} running at http://localhost:{args.port}")
    print(f"  Local network:  http://<your-ip>:{args.port}")
    print(f"  Web:            {_WEB_DIR}")
    print(f"  Data:           {_DATA_ROOT}")
    print(f"  Ctrl+C to stop\n")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
