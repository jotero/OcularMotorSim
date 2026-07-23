<#
=====================================================================================
 server.ps1 - Oculomotor simulator server management (one script, several commands)
=====================================================================================

 USAGE:   .\server.ps1 <command>
          .\server.ps1            (no command shows help)

 COMMANDS:
   up            Open ONE Windows Terminal window with three tabs and launch, in each:
                 the Cloudflare tunnel, the STABLE server (8000), the DEV server (8001).
                 One-shot "bring the whole public site online" convenience.
   dev           Start the DEV server on http://localhost:8001 using THIS checkout's
                 venv (live code). Serves this checkout's web/ + data/.
   stable        Start the STABLE server on http://localhost:8000 from the ..\om-stable
                 worktree using ITS OWN venv = the FROZEN snapshot (frozen model, web, data).
                 Run 'make-stable' first to create/refresh it.
   make-stable   Snapshot HEAD onto the 'stable' branch in the worktree, (re)install the
                 frozen package into stable's own venv, copy .env, then optionally
                 deploy web/ to the public website repo.
   help          Show this help.

 ALIASES:        start_dev=dev | start_stable=stable | make_stable=make-stable | tunnel=cloudflare

 NOTES:
   * dev and stable have SEPARATE databases: each serves its own checkout's server_data/
     (safe to run BOTH at once - no shared file to clobber). Use 'make-stable' to promote
     curated content (featured/favorite runs + collections) from dev into stable.
   * stable runs its OWN venv, so the frozen model code is truly isolated from dev.
   * Venvs live outside OneDrive, under %LOCALAPPDATA%\om-venvs\ (see $venvRoot).
   * Website deploy copies the frozen web/ into ..\om-lab-website\sim, then you commit + push there.
   * The public site (sim.oteromillan.com) needs BOTH the tunnel AND the stable server on
     8000 running. 'up' starts all three at once; each is also runnable on its own tab.
=====================================================================================
#>

param([Parameter(Position = 0)][string]$Command = 'help')

$ErrorActionPreference = 'Stop'

# ── Shared paths / config ────────────────────────────────────────────────────
# Everything is derived from $PSScriptRoot, so moving or renaming this checkout does
# not break the script. The sibling repos (om-stable, om-data, om-lab-website) all
# live next to it under Code\.
$root            = $PSScriptRoot
$codeRoot        = Split-Path $root -Parent
$stableWorktree  = Join-Path $codeRoot 'om-stable'
$websiteDest     = Join-Path $codeRoot 'om-lab-website\sim'

# Venvs live OUTSIDE OneDrive: syncing hundreds of MB of wheels is slow, and a synced
# venv breaks whenever the checkout moves (absolute paths in the editable-install .pth
# and the console-script shims). Override with $env:OCULOMOTOR_VENV_ROOT if needed.
$venvRoot        = if ($env:OCULOMOTOR_VENV_ROOT) { $env:OCULOMOTOR_VENV_ROOT }
                   else { Join-Path $env:LOCALAPPDATA 'om-venvs' }
$mainVenv        = Join-Path $venvRoot 'OcularMotorSim'
$stableVenv      = Join-Path $venvRoot 'OcularMotorSim-stable'
$mainPython      = Join-Path $mainVenv 'Scripts\python.exe'
$stablePython    = Join-Path $stableVenv 'Scripts\python.exe'

# Data dirs are PINNED relative to Code\ so every launch reads the same place, never
# depending on the CWD. stable-data = public runs, dev-data = throwaway dev testing.
# Set OCULOMOTOR_DATA before launching.
$dataStable      = Join-Path $codeRoot 'om-data\stable-data'
$dataDev         = Join-Path $codeRoot 'om-data\dev-data'

function Show-Usage {
    Write-Host @'

server.ps1 - Oculomotor simulator server management

  USAGE:  .\server.ps1 <command>      (no command shows this help)

  COMMANDS
    up            Open a Windows Terminal window with 3 tabs: cloudflare tunnel + stable
                  (8000) + dev (8001). One command to bring the whole public site online.
    cloudflare    Run the Cloudflare tunnel (sim.oteromillan.com -> localhost:8000) in this
                  window. The stable server must be running on 8000 for the site to answer.
    dev           DEV server, http://localhost:8001 (this checkout's venv = live code).
    stable        STABLE server, http://localhost:8000 (..\om-stable worktree, its OWN venv =
                  frozen snapshot: frozen model + web + data). Run make-stable first.
    make-stable   Snapshot HEAD -> stable branch in the worktree, (re)install the frozen
                  package into the worktree venv, copy .env, optionally copy curated
                  gallery examples (featured/favorites) from dev, optionally deploy web/.
    help          Show this help.

  ALIASES   start_dev=dev | start_stable=stable | make_stable=make-stable | tunnel=cloudflare

  NOTES
    * dev and stable have SEPARATE databases, both PINNED under om-data\ (dev-data and
      stable-data) via OCULOMOTOR_DATA - safe to run both at once, and never depends on
      the launch folder. make-stable promotes featured/favorite runs + collections
      from dev-data to stable-data.
    * stable uses its own venv -> its model code is frozen, isolated from dev.
    * Venvs live outside OneDrive, under %LOCALAPPDATA%\om-venvs\.
    * Website deploy dest: ..\om-lab-website\sim

'@
}

function Start-Dev {
    $env:OCULOMOTOR_DATA = $dataDev
    Write-Host 'Starting DEV server (live code) on http://localhost:8001'
    Write-Host "  data: $dataDev"
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
    $env:OCULOMOTOR_DATA = $dataStable
    $ver = git -C $stableWorktree rev-parse --short HEAD
    Write-Host "Starting STABLE server (frozen commit $ver) on http://localhost:8000"
    Write-Host "  data: $dataStable"
    Write-Host 'Ctrl-C to stop.'
    & $stablePython -X utf8 -m oculomotor.server --port 8000
}

function Start-Cloudflare {
    # Runs the named tunnel from ~/.cloudflared/config.yml (ingress -> localhost:8000).
    # The tunnel just forwards; the STABLE server must be up on 8000 for the site to answer.
    $cf = Get-Command cloudflared -ErrorAction SilentlyContinue
    if (-not $cf) {
        Write-Host "cloudflared not found on PATH. Install it or add it to PATH first." -ForegroundColor Red
        exit 1
    }
    Write-Host 'Starting Cloudflare tunnel (sim.oteromillan.com -> localhost:8000)'
    Write-Host '  NOTE: the STABLE server must be running on 8000 or the site returns errors.'
    Write-Host 'Ctrl-C to stop.'
    & $cf.Source tunnel run
}

function Start-All {
    # Open ONE Windows Terminal window with three tabs, each running one long-lived process.
    # We shell out to '.\server.ps1 <cmd>' in each tab so every tab reuses THIS script's path
    # logic (venvs, OCULOMOTOR_DATA, tunnel config) instead of duplicating it here.
    $wt = Get-Command wt -ErrorAction SilentlyContinue
    if (-not $wt) {
        Write-Host 'Windows Terminal (wt.exe) not found. Falling back to 3 separate windows.' -ForegroundColor Yellow
        foreach ($c in 'cloudflare', 'stable', 'dev') {
            Start-Process powershell -ArgumentList @(
                '-NoExit', '-Command', "Set-Location `"$root`"; .\server.ps1 $c")
        }
        return
    }

    # wt parses ';' as its own tab delimiter, so it must be a BARE array element (splatting
    # puts spaces around it). '-w new' forces a fresh window even if launched from inside wt.
    # -d sets each tab's starting directory to this checkout so '.\server.ps1' resolves.
    $tab = {
        param($title, $cmd)
        @('new-tab', '--title', $title, '-d', $root,
          'powershell', '-NoExit', '-Command', ".\server.ps1 $cmd")
    }
    $wtArgs = @('-w', 'new') +
              (& $tab 'cloudflare' 'cloudflare') + @(';') +
              (& $tab 'stable'     'stable')     + @(';') +
              (& $tab 'dev'        'dev')

    Write-Host 'Opening Windows Terminal: [cloudflare | stable | dev]'
    & $wt.Source @wtArgs
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

    # 2. Ensure stable has its OWN venv + editable install = frozen code, isolated from dev.
    if (-not (Test-Path $stablePython)) {
        Write-Host 'Creating stable venv (one-time; downloads deps)...' -ForegroundColor Cyan
        & $mainPython -m venv $stableVenv
    }
    Write-Host 'Installing frozen package into the stable venv...'
    Push-Location $stableWorktree
    & $stablePython -m pip install -e ".[all]" --quiet
    Pop-Location

    # 3. Copy .env (gitignored, absent from the worktree) so stable has API keys.
    $mainEnv = Join-Path $root '.env'
    if (Test-Path $mainEnv) { Copy-Item $mainEnv (Join-Path $stableWorktree '.env') -Force }

    Write-Host "Run '.\server.ps1 stable' to serve the frozen snapshot on port 8000"

    # 4. Promote curated content from dev's DB into the stable gallery. dev and stable
    #    keep SEPARATE databases (safe to run both at once), so this copies featured /
    #    favorite runs AND collections (with the runs they reference). Additive + idempotent.
    Write-Host ''
    Write-Host 'Copy curated content from dev into the stable gallery?' -ForegroundColor Cyan
    Write-Host '  [f] featured only   [b] featured + favorites   [n] none ' -NoNewline
    $pick = (Read-Host).ToLower()
    if ($pick -eq 'f' -or $pick -eq 'b') {
        $cfArgs = @('-X', 'utf8', '-m', 'oculomotor.reports.copy_featured',
                    '--from', $dataDev,
                    '--to',   $dataStable,
                    '--collections')
        if ($pick -eq 'b') { $cfArgs += '--favorites' }
        & $mainPython @cfArgs
    }

    # 5. Optional: deploy the frozen frontend (web/) to the public website repo.
    $publishRepo = $null
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
                $publishRepo = Split-Path $websiteDest -Parent
            }
        }
    }

    # 6. Final reminder. Copying web/ into the website repo only STAGES the files on
    #    disk; the public main page updates ONLY after you commit + push that repo.
    #    That step is manual, so print a loud reminder as the LAST thing on screen.
    if ($publishRepo) {
        Write-Host ''
        Write-Host '============================================================' -ForegroundColor Yellow
        Write-Host '  REMINDER: the website is NOT published yet!' -ForegroundColor Yellow
        Write-Host '  Copying web/ only staged the files locally. The public' -ForegroundColor Yellow
        Write-Host '  main page updates ONLY after you commit + push the repo:' -ForegroundColor Yellow
        Write-Host ''
        Write-Host "    cd `"$publishRepo`"" -ForegroundColor Cyan
        Write-Host '    git add -A; git commit -m "update sim"; git push' -ForegroundColor Cyan
        Write-Host '============================================================' -ForegroundColor Yellow
    }
}

# ── Dispatch ─────────────────────────────────────────────────────────────────
switch ($Command.ToLower()) {
    { $_ -in 'up', 'all', 'servers' }        { Start-All }
    { $_ -in 'cloudflare', 'tunnel' }        { Start-Cloudflare }
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
