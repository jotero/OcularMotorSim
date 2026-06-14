"""Command-line entry point: natural-language description -> simulation figure.

Usage
-----
    python -X utf8 scripts/simulate.py "healthy subject making a 20 deg saccade to the right"
    python -X utf8 scripts/simulate.py --show "OKN 30 deg/s for 20 s, then OKAN"
    python -X utf8 scripts/simulate.py --dry-run "..."
    python -X utf8 scripts/simulate.py --json scenario.json

Calls interpret._call_llm to turn the description into a SimulationScenario,
then run.run_scenario to simulate + plot. Requires ANTHROPIC_API_KEY.
"""

import sys
import os
import json
import argparse
import re

from dotenv import load_dotenv
load_dotenv()  # searches upward from cwd — finds .env at project root

from oculomotor.llm_pipeline.scenario import SimulationScenario
from oculomotor.llm_pipeline.interpret import _call_llm
from oculomotor.llm_pipeline.run import run_scenario

# ── Helpers ───────────────────────────────────────────────────────────────────

def _slugify(text: str) -> str:
    """Convert description to a filename-safe slug."""
    slug = re.sub(r'[^a-z0-9]+', '_', text.lower()).strip('_')
    return slug[:60]


def _default_output_path(description: str) -> str:
    # cli.py is at src/oculomotor/llm_pipeline/cli.py → repo root is 4 levels up.
    here = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(here))))
    out_dir = os.path.join(project_root, 'outputs')
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, _slugify(description) + '.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Convert a natural-language scenario description to an oculomotor simulation.')
    parser.add_argument('description', nargs='?', default=None,
                        help='Plain-English scenario description.')
    parser.add_argument('--show', action='store_true',
                        help='Display the figure interactively.')
    parser.add_argument('--out', default=None,
                        help='Output path for the figure (PNG/SVG). '
                             'Default: outputs/<slug>.png')
    parser.add_argument('--dry-run', action='store_true',
                        help='Print the generated scenario JSON without running.')
    parser.add_argument('--model', default='claude-opus-4-8',
                        help='Claude model to use (default: claude-opus-4-8).')
    parser.add_argument('--json', default=None,
                        help='Load scenario from a JSON file instead of calling the LLM.')
    args = parser.parse_args()

    # ── Load scenario ─────────────────────────────────────────────────────────
    if args.json:
        with open(args.json) as f:
            scenario = SimulationScenario.model_validate(json.load(f))
        description = scenario.description
    elif args.description:
        description = args.description
        scenario = _call_llm(description, args.model)
    else:
        parser.print_help()
        sys.exit(1)

    # ── Print scenario ────────────────────────────────────────────────────────
    print("\n── Generated scenario ──────────────────────────────────────────")
    print(scenario.model_dump_json(indent=2))
    print("────────────────────────────────────────────────────────────────\n")

    if args.dry_run:
        return

    # ── Run simulation ────────────────────────────────────────────────────────
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_path = args.out or _default_output_path(scenario.description)
    print(f"Running simulation: {scenario.description}")

    fig = run_scenario(scenario, output_path=out_path)

    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print("Done.")


if __name__ == '__main__':
    main()
