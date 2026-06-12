"""Smoke test: a vertical saccade must produce vertical (V) traces in the spec."""
from oculomotor.llm_pipeline.scenario import SimulationScenario
from oculomotor.llm_pipeline.runner import run_scenario

# 20 deg UPWARD saccade: target steps in +pitch (lin_y up), lit room.
scn = SimulationScenario(
    description='20 deg vertical (up) saccade',
    duration_s=1.2,
    head=[{'duration_s': 1.2}],
    target=[{'duration_s': 0.3, 'lin_z_0': 1.0, 'lin_y_0': 0.0},
            {'duration_s': 0.9, 'lin_y_0': 0.36}],   # ~20 deg up at 1 m
    scene=[{'duration_s': 1.2}],
    visual=[{'duration_s': 1.2}],
    plot={'panels': ['eye_position', 'eye_velocity', 'saccade_burst', 'neural_integrator']},
)

fig, data, spec = run_scenario(scn, return_data=True, return_spec=True)
print('panels:', [p['name'] for p in spec['panels']])
for p in spec['panels']:
    labels = [tr['label'] for tr in p.get('traces', [])]
    has_v = any(' V' in lbl or lbl.endswith('V') for lbl in labels)
    print(f"  {p['name']:18s} V-trace={has_v}  traces={labels}")
ep = next(p for p in spec['panels'] if p['name'] == 'eye_position')
assert any(' V' in tr['label'] for tr in ep['traces']), 'eye_position has NO vertical trace!'
print('\nOK: vertical saccade now shows vertical eye-position traces.')
