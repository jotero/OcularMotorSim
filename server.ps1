<#
=====================================================================================
 server.ps1 - Oculomotor simulator server management (one script, several commands)
=====================================================================================

 USAGE:   .\server.ps1 <command>
          .\server.ps1            (no command → shows this help)

 COMMANDS:
   dev           Start the DEV server on http://localhost:8001 from THIS working
                 directory (live, uncommitted code). Use while developing.

   stable        Start the STABLE server on http://localhost:8000 from the
                 'om-stable' git worktree - i.e. whatever you last snapshotted
                 with 'make-stable'. Creates/refreshes the worktree automatically.
                 Requires a 'stable' branch (run 'make-stable' first).

   make-stable   Snapshot the current HEAD onto the 'stable' branch (this is what
                 'stable' will serve), then OPTIONALLY deploy web/ (the frontend)
                 to the public website repo via robocopy.

   help          Show this help. (Default when no command is given.)

 ALIASES:        start_dev = dev   |   start_stable = stable   |   make_stable = make-stable

 NOTES:
   * Dev (8001) and stable (8000) share ONE database via
       $env:OCULOMOTOR_OUTPUTS = <USERPROFILE>\oculomotor_outputs
     This is deliberately OUTSIDE OneDrive - OneDrive syncing an actively-written
     log corrupts it (orphaned sidecars, vanished CSV rows).
   * The stable worktree lives next to this repo at ..\om-stable.
   * The website deploy copies web/ into ..\om-lab-website\sim, then you commit
     + push from that repo to publish.
=====================================================================================
#>

param([Parameter(Position = 0)][string]$Command = 'help')

$ErrorActionPreference = 'Stop'

# ── Shared paths / config ────────────────────────────────────────────────────
$root            = $PSScriptRoot
$python          = Join-Path $root '.venv\Scripts\python.exe'
$serverPy        = 'scripts\server.py'
$dataDir         = Join-Path $env:USERPROFILE 'oculomotor_outputs'   # shared, non-OneDrive
$stableWorktree  = Join-Path (Split-Path $root -Parent) 'om-stable'
$websiteDest     = 'D:\OneDrive\UC Berkeley\OMlab - JOM\Code\om-lab-website\sim'

function Show-Usage {
    Write-Host @'

server.ps1 - Oculomotor simulator server management

  USAGE:  .\server.ps1 <command>      (no command shows this help)

  COMMANDS
    dev           Start the DEV server on http://localhost:8001 from THIS working
                  directory (live, uncommitted code). Use while developing.
    stable        Start the STABLE server on http://localhost:8000 from the
                  ..\om-stable worktree (whatever you last saved with make-stable).
                  Creates/refreshes the worktree. Needs a 'stable' branch first.
    make-stable   Snapshot current HEAD onto the 'stable' branch, then optionally
                  deploy web/ (the frontend) to ..\om-lab-website\sim via robocopy.
    help          Show this help.

  ALIASES   start_dev=dev | start_stable=stable | make_stable=make-stable

  NOTES
    * Dev (8001) + stable (8000) share ONE database via
        $env:OCULOMOTOR_OUTPUTS = <USERPROFILE>\oculomotor_outputs
      (deliberately OUTSIDE OneDrive - OneDrive corrupts the live log).
    * Stable worktree: ..\om-stable   Website deploy dest: ..\om-lab-website\sim

'@
}

function Start-Dev {
    $env:OCULOMOTOR_OUTPUTS = $dataDir
    Write-Host 'Starting DEV server on http://localhost:8001'
    Write-Host "Data dir: $env:OCULOMOTOR_OUTPUTS"
    & $python -X utf8 (Join-Path $root $serverPy) --port 8001
}

function Start-Stable {
    $env:OCULOMOTOR_OUTPUTS = $dataDir

    if (-not (git branch --list stable)) {
        Write-Host "No stable version saved yet. Run '.\server.ps1 make-stable' first." -ForegroundColor Yellow
        exit 1
    }

    if (Test-Path $stableWorktree) {
        Write-Host 'Updating stable worktree...'
        git -C $stableWorktree reset --hard stable 2>$null
    } else {
        Write-Host "Creating stable worktree at $stableWorktree ..."
        git worktree add $stableWorktree stable
    }

    $ver = git -C $stableWorktree rev-parse --short HEAD
    Write-Host "Starting STABLE server (commit $ver) on http://localhost:8000"
    Write-Host 'Ctrl-C to stop.'
    & $python -X utf8 (Join-Path $stableWorktree $serverPy) --port 8000
}

function Invoke-MakeStable {
    $short = git rev-parse --short HEAD
    $dirty = git status --porcelain

    if ($dirty) {
        Write-Host 'WARNING: uncommitted changes present. Proceed anyway? (y/n) ' -NoNewline
        if ((Read-Host) -ne 'y') { exit 0 }
    }

    if (Test-Path $stableWorktree) {
        # Branch is checked out in the worktree - reset it there directly.
        Push-Location $stableWorktree
        git reset --hard "$(git -C "$root" rev-parse HEAD)"
        Pop-Location
    } else {
        git branch -f stable HEAD
    }

    Write-Host "Stable branch updated to $short"
    Write-Host "Run '.\server.ps1 stable' to serve it on port 8000"

    # ── Optional: deploy the frontend (web/) to the public website repo ──────
    Write-Host ''
    Write-Host 'Deploy the simulator frontend to the website repo too? (y/n) ' -NoNewline
    if ((Read-Host) -eq 'y') {
        # Prefer the stable worktree's web/ (the snapshot just made); fall back to local web/.
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
