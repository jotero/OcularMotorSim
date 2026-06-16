"""Copy curated runs (featured, and optionally favorites) from one server
database to another — e.g. promote dev's featured examples to the stable gallery.

Dev and stable keep SEPARATE databases (each checkout's ``server_data/``), so
featuring a run in dev never touches stable. This script copies, for each
featured run, the three artifacts the gallery needs into the target DB:

  * ``data/{run_id}.json``        — the run sidecar (interactive plot + metadata)
  * ``server_figures/{run_id}.png`` — the thumbnail (skipped if absent)
  * its row in ``simulation_log.csv`` (merged; ``featured``/``favorite`` set)

It is additive and idempotent: existing target rows/files are preserved, and a
run already present in the target just has its featured/favorite flags updated.
Nothing is deleted. Restart the target server afterwards (it reloads the log and
regenerates admin.html on startup).

Usage
-----
    python -m oculomotor.reports.copy_featured              # dev -> ../om-stable
    python -m oculomotor.reports.copy_featured --to PATH    # explicit target DB
    python -m oculomotor.reports.copy_featured --from PATH --to PATH
    python -m oculomotor.reports.copy_featured --favorites  # also copy favorites
    python -m oculomotor.reports.copy_featured --dry-run    # show, don't write
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

_LOG_COLUMNS = [
    'timestamp', 'run_id', 'version', 'prompt', 'mode', 'title',
    'figure_file', 'looks_correct', 'feedback',
    'favorite', 'featured', 'note',
    'ms_total', 'ms_llm', 'ms_sim',
]

_REPO_ROOT = Path(__file__).resolve().parents[3]   # reports → oculomotor → src → repo


def _truthy(v) -> bool:
    return str(v).strip().lower() in ('true', '1', 'yes')


def _read_log(db: Path) -> list[dict]:
    f = db / 'simulation_log.csv'
    if not f.exists():
        return []
    with open(f, newline='', encoding='utf-8') as fh:
        return [dict(r) for r in csv.DictReader(fh) if r.get('run_id')]


def _write_log(db: Path, rows: list[dict]) -> None:
    f = db / 'simulation_log.csv'
    with open(f, 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=_LOG_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in _LOG_COLUMNS})


def copy_featured(src: Path, dst: Path, include_favorites: bool = False,
                  dry_run: bool = False) -> None:
    if not (src / 'simulation_log.csv').exists():
        raise SystemExit(f"No simulation_log.csv in source DB: {src}")
    (dst / 'data').mkdir(parents=True, exist_ok=True)
    (dst / 'server_figures').mkdir(parents=True, exist_ok=True)

    src_rows = _read_log(src)
    selected = [r for r in src_rows
                if _truthy(r.get('featured'))
                or (include_favorites and _truthy(r.get('favorite')))]
    if not selected:
        print("Nothing to copy (no featured"
              + ("/favorite" if include_favorites else "") + " runs in source).")
        return

    dst_rows = _read_log(dst)
    dst_by_id = {r['run_id']: r for r in dst_rows}

    copied_files, new_rows, updated_rows, missing = 0, 0, 0, []
    for r in selected:
        rid = r['run_id']
        sidecar = src / 'data' / f'{rid}.json'
        if not sidecar.exists():
            missing.append(rid)
            continue   # no sidecar → gallery can't render it; skip
        figure = src / 'server_figures' / f'{rid}.png'

        if not dry_run:
            shutil.copy2(sidecar, dst / 'data' / f'{rid}.json')
            if figure.exists():
                shutil.copy2(figure, dst / 'server_figures' / f'{rid}.png')
        copied_files += 1

        if rid in dst_by_id:                       # already there → just flag it
            dst_by_id[rid]['featured'] = 'True'
            if _truthy(r.get('favorite')):
                dst_by_id[rid]['favorite'] = 'True'
            updated_rows += 1
        else:                                      # bring the row across
            row = {k: r.get(k, '') for k in _LOG_COLUMNS}
            # Repoint figure_file at the target DB's figures dir.
            row['figure_file'] = str(dst / 'server_figures' / f'{rid}.png')
            dst_rows.append(row)
            dst_by_id[rid] = row
            new_rows += 1

    if missing:
        print(f"Skipped {len(missing)} featured run(s) with no sidecar "
              f"(re-run them on the source server to generate data): "
              + ", ".join(missing[:8]) + ("…" if len(missing) > 8 else ""))

    if dry_run:
        print(f"[dry-run] would copy {copied_files} run(s): "
              f"{new_rows} new + {updated_rows} re-flagged in {dst}")
        return

    _write_log(dst, dst_rows)
    print(f"Copied {copied_files} run(s) → {dst}  "
          f"({new_rows} new, {updated_rows} re-flagged).")
    print("Restart the target server to reload the log and rebuild admin.html.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy featured runs between server databases.")
    ap.add_argument('--from', dest='src', default=str(_REPO_ROOT / 'server_data'),
                    help="Source server_data dir (default: this checkout).")
    ap.add_argument('--to', dest='dst',
                    default=str(_REPO_ROOT.parent / 'om-stable' / 'server_data'),
                    help="Target server_data dir (default: ../om-stable/server_data).")
    ap.add_argument('--favorites', action='store_true',
                    help="Also copy favorited runs, not just featured.")
    ap.add_argument('--dry-run', action='store_true', help="Show what would happen.")
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    print(f"Source: {src}\nTarget: {dst}\n")
    copy_featured(src, dst, include_favorites=args.favorites, dry_run=args.dry_run)


if __name__ == '__main__':
    main()
