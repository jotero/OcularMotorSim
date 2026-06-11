# Shared, non-OneDrive data dir (must match start_stable_server.ps1) so the dev
# (8001) and stable (8000) servers use ONE database.
$env:OCULOMOTOR_OUTPUTS = Join-Path $env:USERPROFILE 'oculomotor_outputs'

$python = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
Write-Host 'Starting dev server on http://localhost:8001'
Write-Host "Data dir: $env:OCULOMOTOR_OUTPUTS"
& $python -X utf8 (Join-Path $PSScriptRoot 'scripts\server.py') --port 8001
