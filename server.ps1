<#
=====================================================================================
 server.ps1 - Oculomotor simulator server management (one script, several commands)
=====================================================================================

 USAGE:   .\server.ps1 <command>
          .\server.ps1            (no command shows help)

 COMMANDS:
   dev           Start the DEV server on http://localhost:8001 using THIS checkout's
                 venv (live code). Serves this checkout's web/ + data/.
   stable        Start the STABLE server on http://localhost:8000 from the ..\om-stable
                 worktree using ITS OWN venv = the FROZEN snapshot (frozen model, web, data).
                 Run 'make-stable' first to create/refresh it.
   make-stable   Snapshot HEAD onto the 'stable' branch in the worktree, (re)install the
                 frozen package into the worktree's own venv, copy .env, then optionally
                 deploy web/ to the public website repo.
   help          Show this help.

 ALIASES:        start_dev=dev | start_stable=stable | make_stable=make-stable

 NOTES:
   * dev and stable have SEPARATE databases: each serves its own checkout's data/
     (the package self-locates via OCULOMOTOR_DATA / OCULOMOTOR_WEB, default <checkout>/{data,web}).
     So dev experiments never pollute the public stable gallery.
   * stable runs the worktree's OWN venv, so the frozen model code is truly isolated from dev.
   * Website deploy copies the frozen web/ into ..\om-lab-website\sim, then you commit + push there.
=====================================================================================
#>

param([Parameter(Position = 0)][string]$Command = 'help')

$ErrorActionPreference = 'Stop'

# ── Shared paths / config ────────────────────────────────────────────────────
$root            = $PSScriptRoot
$mainPython      = Join-Path $root '.venv\Scripts\python.exe'
$stableWorktree  = Join-Path (Split-Path $root -Parent) 'om-stable'
$stablePython    = Join-Path $stableWorktree '.venv\Scripts\python.exe'
$websiteDest     = 'D:\OneDrive\UC Berkeley\OMlab - JOM\Code\om-lab-website\sim'

function Show-Usage {
    Write-Host @'

server.ps1 - Oculomotor simulator server management

  USAGE:  .\server.ps1 <command>      (no command shows this help)

  COMMANDS
    dev           DEV server, http://localhost:8001 (this checkout's venv = live code).
    stable        STABLE server, http://localhost:8000 (..\om-stable worktree's OWN venv =
                  frozen snapshot: frozen model + web + data). Run make-stable first.
    make-stable   Snapshot HEAD -> stable branch in the worktree, (re)install the frozen
                  package into the worktree venv, copy .env, optionally copy curated
                  gallery examples (featured/favorites) from dev, optionally deploy web/.
    help          Show this help.

  ALIASES   start_dev=dev | start_stable=stable | make_stable=make-stable

  NOTES
    * dev and stable have SEPARATE databases (each checkout's data/). Override with
      $env:OCULOMOTOR_DATA / $env:OCULOMOTOR_WEB.
    * stable uses the worktree's own venv -> its model code is frozen, isolated from dev.
    * Website deploy dest: ..\om-lab-website\sim

'@
}

function Start-Dev {
    Write-Host 'Starting DEV server (live code) on http://localhost:8001'
    & $mainPython -X utf8 -m oculomotor.server --port 8001
}

function Start-Stable {
    if (-not (git branch --list stable)) {
        Write-Host "No stable snapshot yet. Run '.\server.ps1 make-stable' first." -ForegroundColor Yellow
        exit 1
    }
    if (-not (Test-Path $stablePython)) {
        Write-Host "Stable worktree venv missing. Run '.\server.ps1 make-stable' first." -ForegroundColor Yellow
        exit 1
    }
    $ver = git -C $stableWorktree rev-parse --short HEAD
    Write-Host "Starting STABLE server (frozen commit $ver) on http://localhost:8000"
    Write-Host 'Ctrl-C to stop.'
    & $stablePython -X utf8 -m oculomotor.server --port 8000
}

function Invoke-MakeStable {
    $short = git rev-parse --short HEAD
    $dirty = git status --porcelain

    if ($dirty) {
        Write-Host 'WARNING: uncommitted changes present. Proceed anyway? (y/n) ' -NoNewline
        if ((Read-Host) -ne 'y') { exit 0 }
    }

    # 1. Snapshot HEAD onto the stable branch, checked out in the worktree.
    if (Test-Path $stableWorktree) {
        Push-Location $stableWorktree
        git reset --hard "$(git -C "$root" rev-parse HEAD)"
        Pop-Location
    } else {
        git branch -f stable HEAD
        git worktree add $stableWorktree stable
    }
    Write-Host "Stable branch updated to $short"

    # 2. Ensure the worktree has its OWN venv + editable install = frozen code, isolated from dev.
    if (-not (Test-Path $stablePython)) {
        Write-Host 'Creating stable venv (one-time; downloads deps)...' -ForegroundColor Cyan
        & $mainPython -m venv (Join-Path $stableWorktree '.venv')
    }
    Write-Host 'Installing frozen package into the stable venv...'
    Push-Location $stableWorktree
    & $stablePython -m pip install -e ".[all]" --quiet
    Pop-Location

    # 3. Copy .env (gitignored, absent from the worktree) so stable has API keys.
    $mainEnv = Join-Path $root '.env'
    if (Test-Path $mainEnv) { Copy-Item $mainEnv (Join-Path $stableWorktree '.env') -Force }

    Write-Host "Run '.\server.ps1 stable' to serve the frozen snapshot on port 8000"

    # 4. Optional: promote curated examples from dev's DB into the stable gallery.
    #    dev and stable keep separate databases, so featured/favorite runs don't
    #    cross over automatically. This copy is idempotent (existing runs are
    #    re-flagged, never duplicated).
    Write-Host ''
    Write-Host 'Copy curated examples from dev into the stable gallery?' -ForegroundColor Cyan
    Write-Host '  [f] featured only   [b] featured + favorites   [n] none ' -NoNewline
    $pick = (Read-Host).ToLower()
    if ($pick -eq 'f' -or $pick -eq 'b') {
        $cfArgs = @('-X', 'utf8', '-m', 'oculomotor.reports.copy_featured',
                    '--from', (Join-Path $root 'server_data'),
                    '--to',   (Join-Path $stableWorktree 'server_data'))
        if ($pick -eq 'b') { $cfArgs += '--favorites' }
        & $mainPython @cfArgs
    }

    # 5. Optional: deploy the frozen frontend (web/) to the public website repo.
    Write-Host ''
    Write-Host 'Deploy the simulator frontend to the website repo too? (y/n) ' -NoNewline
    if ((Read-Host) -eq 'y') {
        $srcWeb = Join-Path $stableWorktree 'web'
        if (-not (Test-Path $srcWeb)) { $srcWeb = Join-Path $root 'web' }

        if (-not (Test-Path $websiteDest)) {
            Write-Host "Website folder not found: $websiteDest" -ForegroundColor Red
        } else {
            Write-Host "Deploying $srcWeb -> $websiteDest"
            # /E = copy subtree (incremental, keeps dest-only files); quiet per-file output.
            robocopy $srcWeb $websiteDest /E /NFL /NDL /NP /R:2 /W:1
            if ($LASTEXITCODE -ge 8) {
                Write-Host "robocopy failed (exit $LASTEXITCODE)" -ForegroundColor Red
            } else {
                $repo = Split-Path $websiteDest -Parent
                Write-Host 'Deployed. To publish, from the website repo:' -ForegroundColor Green
                Write-Host "  cd `"$repo`"; git add -A; git commit -m `"update sim`"; git push" -ForegroundColor Cyan
            }
        }
    }
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    { $_ -in 'dev', 'start_dev' }            { Start-Dev }
    { $_ -in 'stable', 'start_stable' }      { Start-Stable }
    { $_ -in 'make-stable', 'make_stable' }  { Invoke-MakeStable }
    { $_ -in 'help', '-h', '--help', '/?' }  { Show-Usage }
    default {
        Write-Host "Unknown command: '$Command'" -ForegroundColor Red
        Write-Host ''
        Show-Usage
        exit 1
    }
}
