"""One-off migration: re-run every stored simulation through the CURRENT model.

Old runs stored the patient as a FULL model_dump, baking in gen-time defaults. Two
defaults have since changed (K_phasic_verg 3->12, tonic_verg), so re-running as-is
would (a) mis-apply the stale vergence gain to the SIM and (b) show drift. This
recovers the genuine per-scenario overrides:

  * auto-detect "changed defaults" = fields whose drifted value is a single baked
    value shared across many runs (--min-count), and strip a field only when its
    value equals that baked default (a genuine override to a different value is kept);
  * also drop fields already equal to the current default (no-op).

Each run is re-simulated with the stripped patient and its sidecar rewritten
(detail w/ exclude_unset patient, plot_spec, eye_trajectory, patient_changes,
version), preserving run_id/timestamp/prompt/favorite/featured/note/feedback.

    python -m ...  (run from repo)   ->  see _diag; use:
    ./.venv/Scripts/python.exe -X utf8 scratch/_migrate_rerun_db.py --dry-run
    ./.venv/Scripts/python.exe -X utf8 scratch/_migrate_rerun_db.py            # real
"""
from __future__ import annotations
import argparse, copy, json
from collections import defaultdict, Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from oculomotor import __version__ as SIM_VERSION
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison
from oculomotor.llm_pipeline.patient_builder import Patient
from oculomotor.llm_pipeline.run import run_scenario, run_comparison
from oculomotor.server.app import (
    _DATA_ROOT, _FIGURES_DIR, _data_path,
    _build_eye_trajectory, _build_patient_changes, _looks_changed,
)

DATA_DIR = _DATA_ROOT / 'data'
DP = Patient()   # current defaults


def _vkey(v):
    return round(float(v), 9) if isinstance(v, (int, float)) else None


def load_runs():
    runs = []
    for f in sorted(DATA_DIR.glob('*.json')):
        try:
            pl = json.loads(f.read_text(encoding='utf-8'))
        except Exception:
            continue
        if pl.get('detail'):
            runs.append(pl)
    return runs


def patient_dicts(pl):
    d = pl['detail']
    if pl.get('mode') == 'comparison':
        return [sc.get('patient', {}) for sc in d.get('scenarios', [])]
    return [d.get('patient', {})]


def detect_changed_defaults(runs, min_count):
    """{field: baked_default_value} for scalar fields drifting to one shared value
    across >= min_count runs (i.e. a default that changed, not a per-scenario override)."""
    vals = defaultdict(Counter)
    for pl in runs:
        for pd in patient_dicts(pl):
            for k, v in pd.items():
                try:
                    dv = getattr(DP, k)
                except AttributeError:
                    continue
                if _looks_changed(v, dv) and _vkey(v) is not None:
                    vals[k][_vkey(v)] += 1
    out = {}
    for k, cnt in vals.items():
        val, n = cnt.most_common(1)[0]
        if n >= min_count:
            out[k] = val
    return out


def strip(pd, changed):
    """Keep only genuine overrides: differ from current default AND not the baked
    changed-default value."""
    keep = {}
    for k, v in pd.items():
        try:
            dv = getattr(DP, k)
        except AttributeError:
            continue
        if not _looks_changed(v, dv):
            continue                              # already at current default
        if k in changed and _vkey(v) == changed[k]:
            continue                              # baked stale default
        keep[k] = v                               # genuine per-scenario override
    return keep


def stripped_detail(pl, changed):
    d = copy.deepcopy(pl['detail'])
    if pl.get('mode') == 'comparison':
        for sc in d.get('scenarios', []):
            sc['patient'] = strip(sc.get('patient', {}), changed)
    else:
        d['patient'] = strip(d.get('patient', {}), changed)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dry-run', action='store_true')
    ap.add_argument('--figure', action='store_true', help='also regenerate thumbnail PNG')
    ap.add_argument('--min-count', type=int, default=25, help='runs sharing a baked value => changed default')
    args = ap.parse_args()

    runs = load_runs()
    changed = detect_changed_defaults(runs, args.min_count)
    print(f'DB: {_DATA_ROOT}   runs={len(runs)}')
    print('Changed-default fields stripped (field=baked_value):',
          {k: changed[k] for k in sorted(changed)})

    if args.dry_run:
        for pl in runs:
            ovs = [sorted(strip(pd, changed).keys()) for pd in patient_dicts(pl)]
            print(f"  {pl['run_id'][:8]} {pl.get('mode','single'):10} keep={ovs}  {pl.get('title','')[:38]}")
        return

    ok = 0
    for pl in runs:
        rid = pl['run_id']
        try:
            detail = stripped_detail(pl, changed)
            if pl.get('mode') == 'comparison':
                scn = SimulationComparison.model_validate(detail)
                fig, sim_list, spec = run_comparison(scn, return_data=True, return_spec=True, make_figure=args.figure)
                eye_trajectories = []
                for scenario, sd in zip(scn.scenarios, sim_list):
                    tr = _build_eye_trajectory(sd)
                    if tr is not None:
                        tr['label'] = scenario.description
                        tr['patient_changes'] = _build_patient_changes(scenario.patient)
                        eye_trajectories.append(tr)
                title, narrative = scn.title, getattr(scn, 'narrative', '') or ''
                eye_traj, patient_changes = None, None
                det = scn.model_dump()
                for i, sc in enumerate(scn.scenarios):
                    det['scenarios'][i]['patient'] = sc.patient.model_dump(exclude_unset=True)
            else:
                scn = SimulationScenario.model_validate(detail)
                fig, sim_data, spec = run_scenario(scn, return_data=True, return_spec=True, make_figure=args.figure)
                eye_traj = _build_eye_trajectory(sim_data)
                eye_trajectories = None
                title, narrative = scn.description, getattr(scn, 'narrative', '') or ''
                patient_changes = _build_patient_changes(scn.patient)
                det = scn.model_dump()
                det['patient'] = scn.patient.model_dump(exclude_unset=True)
            if fig is not None:
                fig.savefig(_FIGURES_DIR / f'{rid}.png', dpi=130, bbox_inches='tight'); plt.close(fig)
            pl.update(version=SIM_VERSION, title=title, narrative=narrative, detail=det,
                      plot_spec=spec, eye_trajectory=eye_traj,
                      eye_trajectories=eye_trajectories, patient_changes=patient_changes)
            _data_path(rid).write_text(json.dumps(pl, ensure_ascii=False), encoding='utf-8')
            ok += 1
            print(f'  ok  {rid[:8]}  {title[:44]}')
        except Exception as e:
            print(f'  FAIL {rid[:8]}  {type(e).__name__}: {e}')
    print(f'\nRe-ran {ok}/{len(runs)}. Restart the server to pick up rewritten sidecars.')


if __name__ == '__main__':
    main()
