"""LLM simulation pipeline — natural-language → SimulationScenario → Figure.

Flow: ``cli`` → ``interpret`` (sends ``prompt`` to Claude) → ``scenario`` →
``patient_builder`` → ``run`` → matplotlib Figure.

Submodules
----------
prompt          The system prompt sent to Claude (edit here to tune interpretation)
interpret       NL description → SimulationScenario via the Claude API (call_llm)
scenario        Pydantic schema (SimulationScenario, SimulationComparison, Patient)
patient_builder Patient params built from schema/parameters_schema.yaml
run             SimulationScenario → stimulus + simulator wiring + figure (run_scenario)
cli             Command-line entry point (main)
"""
