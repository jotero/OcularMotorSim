"""Re-run curated runs (featured, optionally favorites) through the CURRENT
pipeline so their interactive plots + avatar data reflect the latest model and
visualisations — without going through the LLM (deterministic).

Each run's sidecar stores the exact scenario the LLM produced (``detail``); this
re-runs that scenario via run_scenario / run_comparison and rewrites the sidecar
(plot_spec, eye_trajectory, narrative, version, patient_changes), preserving
run_id / timestamp / prompt / favorite / featured / note / feedback.

Thumbnails: by default the existing figure PNG is left as-is (the gallery/featured
thumbnails keep working). Pass --figure to also regenerate a matplotlib thumbnail.

Usage
-----
    python -m oculomotor.reports.rerun_featured            # featured only
    python -m oculomotor.reports.rerun_featured --favorites
    python -m oculomotor.reports.rerun_featured --figure   # also refresh thumbnails
    python -m oculomotor.reports.rerun_featured --dry-run

Operates on the server DB resolved by OCULOMOTOR_DATA (default <checkout>/server_data),
i.e. the same DB the dev server serves. Restart the server afterwards (or it will
pick up the rewritten sidecars on the next request).
"""
from __future__ import annotations

import argparse
import json

import matplotlib.pyplot as plt

from oculomotor import __version__ as _SIM_VERSION
from oculomotor.llm_pipeline.scenario import SimulationScenario, SimulationComparison
from oculomotor.llm_pipeline.run import run_scenario, run_comparison
# Reuse the server's DB paths + builders (import has no server-start side effects).
from oculomotor.server.app import (
    _DATA_ROOT, _FIGURES_DIR, _data_path,
    _build_eye_trajectory, _build_patient_changes,
)


def _truthy(v) -> bool:
    return str(v).strip().lower() in ('true', '1', 'yes')


def _rerun_one(payload: dict, make_figure: bool) -> dict:
    """Re-run a single run's stored scenario and return the updated sidecar."""
    mode   = payload.get('mode', 'single')
    detail = payload['detail']

    if mode == 'comparison':
        scn = SimulationComparison.model_validate(detail)
        fig, sim_list, spec = run_comparison(
            scn, return_data=True, return_spec=True, make_figure=make_figure)
        eye_trajectories = []
        for scenario, sd in zip(scn.scenarios, sim_list):
            tr = _build_eye_trajectory(sd)
            if tr is not None:
                tr['label'] = scenario.description
                tr['patient_changes'] = _build_patient_changes(scenario.patient)
                eye_trajectories.append(tr)
        title, narrative = scn.title, getattr(scn, 'narrative', '') or ''
        eye_traj, patient_changes = None, None
    else:
        scn = SimulationScenario.model_validate(detail)
        fig, sim_data, spec = run_scenario(
            scn, return_data=True, return_spec=True, make_figure=make_figure)
        eye_traj = _build_eye_trajectory(sim_data)
        eye_trajectories = None
        title, narrative = scn.description, getattr(scn, 'narrative', '') or ''
        patient_changes = _build_patient_changes(scn.patient)

    if fig is not None:
        fig.savefig(_FIGURES_DIR / f"{payload['run_id']}.png", dpi=130, bbox_inches='tight')
        plt.close(fig)

    payload.update(
        version         = _SIM_VERSION,
        title           = title,
        narrative       = narrative,
        detail          = scn.model_dump(),
        plot_spec       = spec,
        eye_trajectory  = eye_traj,
        eye_trajectories= eye_trajectories,
        patient_changes = patient_changes,
    )
    return payload


def main() -> None:
    ap = argparse.ArgumentParser(description="Re-run featured/favorite runs through the current pipeline.")
    ap.add_argument('--favorites', action='store_true', help="Also re-run favorited runs, not just featured.")
    ap.add_argument('--figure', action='store_true', help="Also regenerate the matplotlib thumbnail PNG.")
    ap.add_argument('--dry-run', action='store_true', help="List what would be re-run; don't write.")
    args = ap.parse_args()

    data_dir = _DATA_ROOT / 'data'
    if not data_dir.exists():
        raise SystemExit(f"No sidecars in {data_dir}")

    targets = []
    for p in sorted(data_dir.glob('*.json')):
        try:
            payload = json.loads(p.read_text(encoding='utf-8'))
        except Exception:
            continue
        if not payload.get('detail'):
            continue
        if _truthy(payload.get('featured')) or (args.favorites and _truthy(payload.get('favorite'))):
            targets.append(payload)

    print(f"DB: {_DATA_ROOT}\n{len(targets)} run(s) to re-run"
          + (" (featured + favorites)" if args.favorites else " (featured)"))
    if args.dry_run:
        for pl in targets:
            print(f"  [dry] {pl['run_id'][:10]}  {pl.get('title', '')[:50]}")
        return

    ok = 0
    for pl in targets:
        rid = pl['run_id']
        try:
            updated = _rerun_one(pl, make_figure=args.figure)
            _data_path(rid).write_text(
                json.dumps(updated, ensure_ascii=False), encoding='utf-8')
            ok += 1
            print(f"  ✓ {rid[:10]}  {updated.get('title', '')[:50]}")
        except Exception as e:
            print(f"  ✗ {rid[:10]}  FAILED: {e}")

    print(f"\nRe-ran {ok}/{len(targets)}. Restart the server (or it picks up the "
          f"rewritten sidecars on the next request).")


if __name__ == '__main__':
    main()
