"""Re-run a single run by id through the CURRENT pipeline (regenerates its sidecar:
plot_spec, labels, eye_trajectory, version). Usage: _rerun_id.py <run_id>"""
import sys, json
from oculomotor.reports.rerun_featured import _rerun_one
from oculomotor.server.app import _data_path

rid = sys.argv[1]
payload = json.loads(_data_path(rid).read_text(encoding='utf-8'))
updated = _rerun_one(payload, make_figure=False)
_data_path(rid).write_text(json.dumps(updated, ensure_ascii=False), encoding='utf-8')
labels = [p['ylabel'] for p in updated['plot_spec']['panels'] if 'eye_position' in p['name']]
print("re-ran", rid, "-> eye-position labels:", labels)
