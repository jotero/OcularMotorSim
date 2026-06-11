$ErrorActionPreference = 'Stop'

$short = git rev-parse --short HEAD
$dirty = git status --porcelain

if ($dirty) {
    Write-Host 'WARNING: uncommitted changes present. Proceed anyway? (y/n)' -NoNewline
    $ans = Read-Host
    if ($ans -ne 'y') { exit 0 }
}

$worktree = 'D:/OneDrive/UC Berkeley/OMlab - JOM/Code/om-stable'

if (Test-Path $worktree) {
    # Branch is checked out in worktree — reset it there directly
    Push-Location $worktree
    git reset --hard "$(git -C "$PSScriptRoot" rev-parse HEAD)"
    Pop-Location
} else {
    git branch -f stable HEAD
}

Write-Host "Stable branch updated to $short"
Write-Host 'Run start_stable_server.ps1 to serve it on port 8000'

# Optionally deploy the simulator frontend (docs/) to the public website repo.
Write-Host ''
Write-Host 'Deploy the simulator frontend to the website repo too? (y/n) ' -NoNewline
$deploy = Read-Host
if ($deploy -eq 'y') {
    $dest = 'D:\OneDrive\UC Berkeley\OMlab - JOM\Code\om-lab-website\sim'
    # Prefer the stable worktree's docs (the snapshot just made); fall back to local docs.
    $srcDocs = Join-Path $worktree 'docs'
    if (-not (Test-Path $srcDocs)) { $srcDocs = Join-Path $PSScriptRoot 'docs' }

    if (-not (Test-Path $dest)) {
        Write-Host "Website folder not found: $dest" -ForegroundColor Red
    } else {
        Write-Host "Deploying $srcDocs -> $dest"
        # /E = copy subtree (incremental, keeps dest-only files); quiet per-file output.
        robocopy $srcDocs $dest /E /NFL /NDL /NP /R:2 /W:1
        if ($LASTEXITCODE -ge 8) {
            Write-Host "robocopy failed (exit $LASTEXITCODE)" -ForegroundColor Red
        } else {
            $repo = Split-Path $dest -Parent
            Write-Host 'Deployed. To publish, from the website repo:' -ForegroundColor Green
            Write-Host "  cd `"$repo`"; git add -A; git commit -m `"update sim`"; git push" -ForegroundColor Cyan
        }
    }
}
