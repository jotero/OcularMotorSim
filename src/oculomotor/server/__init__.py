"""FastAPI web server for the oculomotor LLM pipeline.

Run:  python -m oculomotor.server [--port 8001]
The frontend (web/) and database (data/) directories are resolved from the package's
repo root, overridable via OCULOMOTOR_WEB / OCULOMOTOR_DATA. dev and stable each run
from their own checkout (own venv) so stable serves the frozen model + its own data.
"""
from oculomotor.server.app import app, main

__all__ = ['app', 'main']
