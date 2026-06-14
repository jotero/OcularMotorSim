"""Interpret a natural-language description -> SimulationScenario via the Claude API.

The LLM-call layer: sends the user's description plus the system prompt
(``prompt.py``) to Claude with forced tool_use, and returns a SimulationScenario
or SimulationComparison.

Requires: ANTHROPIC_API_KEY.
"""

import anthropic

from oculomotor.llm_pipeline.scenario import (
    SimulationScenario, SimulationComparison, json_schema, comparison_json_schema)
from oculomotor.llm_pipeline.prompt import SYSTEM_PROMPT

# Cache the large static system prompt (tools + system render before it) so repeated
# calls read it at ~0.1x cost instead of re-sending ~6K tokens every time. 1-hour TTL
# (vs the 5-min default) keeps it warm across an interactive exploration session — worth
# the higher write cost (2x vs 1.25x) since users fire several scenarios minutes apart.
_SYSTEM = [{"type": "text", "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}}]

# ── LLM call ──────────────────────────────────────────────────────────────────

def call_llm(description: str, model: str) -> SimulationScenario | SimulationComparison:
    """Call the Claude API with both tools; LLM picks single or comparison.

    Returns either a SimulationScenario or a SimulationComparison depending
    on what the description calls for.
    """
    client = anthropic.Anthropic()

    tools = [
        {
            "name": "generate_scenario",
            "description": (
                "Use for a single simulation: one patient, one stimulus. "
                "Use when the user describes one condition or paradigm without asking to compare."
            ),
            "input_schema": json_schema(),
        },
        {
            "name": "generate_comparison",
            "description": (
                "Use when the user wants to compare 2–4 conditions on the same stimulus "
                "(e.g. 'healthy vs neuritis', 'compare X and Y', 'show the difference between'). "
                "All scenarios MUST share identical stimulus and differ ONLY in patient parameters. "
                "Each scenario.description becomes its legend label — keep it short (≤5 words)."
            ),
            "input_schema": comparison_json_schema(),
        },
    ]

    print(f"Calling {model}...")
    response = client.messages.create(
        model=model,
        max_tokens=4096,
        system=_SYSTEM,
        tools=tools,
        tool_choice={"type": "any"},   # force tool use — never a plain-text reply
        messages=[{"role": "user", "content": description}],
    )

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        text = " ".join(b.text for b in response.content if hasattr(b, "text"))
        raise ValueError(f"LLM did not call a tool (stop_reason={response.stop_reason!r}). "
                         f"Response text: {text[:300]!r}")
    print(f"  → LLM chose: {tool_block.name}")

    if tool_block.name == "generate_comparison":
        return SimulationComparison.model_validate(tool_block.input)
    else:
        return SimulationScenario.model_validate(tool_block.input)


# Keep old names as aliases for any direct CLI use
def _call_llm(description: str, model: str) -> SimulationScenario:
    result = call_llm(description, model)
    if isinstance(result, SimulationComparison):
        raise ValueError("Description implies a comparison — use call_llm() instead.")
    return result


def _call_llm_comparison(description: str, model: str) -> SimulationComparison:
    """Call the Claude API to generate a SimulationComparison from a description."""
    client = anthropic.Anthropic()

    tool_schema = {
        "name": "generate_comparison",
        "description": (
            "Generate a SimulationComparison from the user's natural-language description. "
            "All scenarios MUST share identical stimulus (head_motion, target, visual, duration_s) "
            "and differ ONLY in patient parameters. "
            "Each scenario.description becomes its legend label — keep it short (≤5 words). "
            "Choose panels that best show the difference between conditions."
        ),
        "input_schema": comparison_json_schema(),
    }

    print(f"Calling {model} to generate comparison...")
    response = client.messages.create(
        model=model,
        max_tokens=4096,  # comparisons are larger
        system=_SYSTEM,
        tools=[tool_schema],
        tool_choice={"type": "tool", "name": "generate_comparison"},
        messages=[{"role": "user", "content": description}],
    )

    tool_block = next(b for b in response.content if b.type == "tool_use")
    return SimulationComparison.model_validate(tool_block.input)
