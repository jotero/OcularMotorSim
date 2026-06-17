"""Does a plain fixation produce an eye-position offset, and do the eyes land on
target once it's applied?  Prints the spec offsets + displayed means."""
import numpy as np
from oculomotor.llm_pipeline.scenario import (
    SimulationScenario, BodySegment, VisualFlagsSegment, PlotConfig)
from oculomotor.llm_pipeline.run import run_scenario

fix = SimulationScenario(
    description="fixation, target straight ahead 1 m",
    head=[BodySegment(duration_s=5.0)],
    target=[BodySegment(duration_s=5.0, lin_x_0=0.0, lin_y_0=0.0, lin_z_0=1.0)],
    scene=[BodySegment(duration_s=5.0)],
    visual=[VisualFlagsSegment(duration_s=5.0)],
    plot=PlotConfig(panels=['eye_position', 'vergence']),
)
_, spec = run_scenario(fix, make_figure=False, return_spec=True)
p = next(pp for pp in spec['panels'] if pp['name'] == 'eye_position_h')
print("ylabel:", p['ylabel'])
t = np.array(spec['t'])
last = t > (t[-1] - 1.0)
for tr in p['traces']:
    y = np.array([np.nan if v is None else v for v in tr['y']], float)
    off = tr.get('offset', 0.0)
    raw_mean = np.nanmean(y[last])
    disp_mean = np.nanmean((y + off)[last])
    print(f"  {tr['label']:8s} offset={off:+.3f}  raw_mean(last1s)={raw_mean:+.3f}  "
          f"displayed_mean={disp_mean:+.3f}")
print("\n-> if offsets are nonzero and displayed L/R ~ target ~ 0, the code is correct;")
print("   then a still-straddling plot means the dev server wasn't restarted or the")
print("   page wasn't hard-refreshed (old plotspec.js / old spec).")
