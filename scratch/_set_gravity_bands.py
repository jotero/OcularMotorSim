"""One-off: set physiological bands for OVAR / tilt-suppression / tVOR metrics.

Physiology @ 60 deg/s OVAR + the tilt-suppression protocol. The two solid
patterns (OVAR modulation grows with sin(tilt); tilt-suppression TC shrinks with
tilt) are band-enforced; absolute magnitudes are provisional, anchored to the
protocol + the model's current sane scale, pending calibration vs the Laurens ref.
"""
import json

P = r'd:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\web\benchmarks\metrics_ranges.json'
r = json.load(open(P, encoding='utf-8'))

upd = {
    # OVAR modulation MUST grow with sin(tilt); small-tilt bands tight so the
    # current backwards trend flags.
    'gravity_ovar_mod_10': (0.2, 1.5),
    'gravity_ovar_mod_30': (0.6, 3.0),
    'gravity_ovar_mod_60': (1.2, 4.5),
    'gravity_ovar_mod_90': (1.5, 5.0),
    # OVAR bias: consistent sign (neg, this rotation dir), |bias| < spin, grows with tilt.
    'gravity_ovar_bias_10': (-35.0, 0.0),
    'gravity_ovar_bias_30': (-40.0, -5.0),
    'gravity_ovar_bias_60': (-45.0, -10.0),
    'gravity_ovar_bias_90': (-50.0, -12.0),
    # Tilt-suppression TC MUST decrease with tilt. Upright anchored to VS TC (~35 s).
    'gravity_tilt_tc_0':  (18.0, 45.0),
    'gravity_tilt_tc_30': (8.0, 22.0),
    'gravity_tilt_tc_60': (4.0, 15.0),
    'gravity_tilt_tc_90': (3.0, 12.0),
    # Reversal (model diagnostic): late post-rotatory SPV should stay small.
    'gravity_tilt_reversal_60': (-3.0, 3.0),
    'gravity_tilt_reversal_90': (-3.0, 3.0),
    # tVOR cross-axis torsion during PURE lateral translation should stay small.
    'tvor_torsion_cross_dark':         (None, 1.5),
    'tvor_torsion_cross_scene':        (None, 1.5),
    'tvor_torsion_cross_scene_target': (None, 1.5),
}
for k, (lo, hi) in upd.items():
    r[k]['lo'] = lo
    r[k]['hi'] = hi

json.dump(r, open(P, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
print('updated', len(upd), 'band entries')
for k in upd:
    print('  {:32s} [{} .. {}]'.format(k, r[k]['lo'], r[k]['hi']))
