"""Entry point — delegates to oculomotor.llm_pipeline.cli."""
from oculomotor.llm_pipeline.cli import main
from oculomotor.llm_pipeline.interpret import call_llm, _call_llm, _call_llm_comparison  # noqa: F401

if __name__ == '__main__':
    main()
