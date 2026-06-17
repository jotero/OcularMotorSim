"""End-to-end check: run a cover test, render the eye_position panel from the
SPEC (raw y + client-side offset) to confirm eyes sit on target and the covered
eye peels off — proving the raw-data + offset design works."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from oculomotor.llm_pipeline.scenario import (
    SimulationScenario, BodySegment, VisualFlagsSegment, PlotConfig)
from oculomotor.llm_pipeline.run import run_scenario


def get_spec(sc):
    _, spec = run_scenario(sc, make_figure=False, return_spec=True)
    return spec


def panel(spec, name):
    return next((p for p in spec['panels'] if p['name'] == name), None)


def draw_eyepos(ax, spec, t, title):
    p = panel(spec, 'eye_position_h')
    if p is None:
        ax.set_title(title + " (no eye_position_h)"); return
    for tr in p['traces']:
        y = np.array([np.nan if v is None else v for v in tr['y']], float)
        off = tr.get('offset', 0.0)
        ax.plot(t[:len(y)], y + off, label=f"{tr['label']}"
                + (f" (+{off:.2f})" if off else ""),
                color=tr['color'], ls={'-': '-', '--': '--', ':': ':'}.get(tr['style'], '-'))
    ax.set_title(f"{title}\n{p['ylabel']}", fontsize=9)
    ax.legend(fontsize=7); ax.axhline(0, color='#bbb', lw=.6, ls='--')
    ax.set_xlabel("time (s)")


# Cover test: near target 0.5 m straight ahead; cover RIGHT eye 2-4 s.
cover = SimulationScenario(
    description="cover test, near target 0.5 m",
    head=[BodySegment(duration_s=6.0)],
    target=[BodySegment(duration_s=6.0, lin_x_0=0.0, lin_y_0=0.0, lin_z_0=0.5)],
    scene=[BodySegment(duration_s=6.0)],
    visual=[VisualFlagsSegment(duration_s=2.0),
            VisualFlagsSegment(duration_s=2.0, cover_R=True),
            VisualFlagsSegment(duration_s=2.0)],
    plot=PlotConfig(panels=['visual_flags', 'eye_position', 'vergence']),
)

# Saccade to an eccentric near target (still 0.5 m), to check eyes ride the target.
sacc = SimulationScenario(
    description="near saccade 0.5 m, 15 deg",
    head=[BodySegment(duration_s=4.0)],
    target=[BodySegment(duration_s=1.5, lin_x_0=0.0, lin_z_0=0.5),
            BodySegment(duration_s=2.5, lin_x_0=0.134, lin_z_0=0.5)],  # ~15 deg at 0.5 m
    scene=[BodySegment(duration_s=4.0)],
    visual=[VisualFlagsSegment(duration_s=4.0)],
    plot=PlotConfig(panels=['eye_position', 'vergence']),
)

specs = {}
for name, sc in (("cover_R @0.5m", cover), ("near saccade @0.5m", sacc)):
    spec = get_spec(sc)
    specs[name] = spec
    ep = panel(spec, 'eye_position_h')
    print(f"\n=== {name} ===  ylabel={ep['ylabel']!r}")
    for tr in ep['traces']:
        print(f"   trace {tr['label']:8s} offset={tr.get('offset', 0.0):+.3f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2), dpi=130)
for ax, (name, spec) in zip(axes, specs.items()):
    draw_eyepos(ax, spec, np.array(spec['t']), name)
fig.tight_layout()
out = r"d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_verify_binoc.png"
fig.savefig(out); print("\nwrote", out)

# CLI matplotlib path (parity check — must not error, label must carry the ref).
cli_out = r"d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_verify_binoc_cli.png"
run_scenario(cover, output_path=cli_out, make_figure=True)
print("wrote CLI figure", cli_out)
