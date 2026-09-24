# SETUP — running OcularMotorSim on a new machine

Everything you need to go from a bare machine to running simulations, benchmarks and the
web servers. Works on macOS, Linux and Windows; where they differ, both are given.

> **Architecture and conventions live in [CLAUDE.md](CLAUDE.md); this file is only about
> getting it to run.** If you want to know *what* the model does, start at [MAP.md](MAP.md).

---

## 0. The one rule about folder layout

The repo can live anywhere, **but the virtualenv must live outside any synced folder**
(OneDrive, Dropbox, iCloud). Two reasons, both learned the hard way: syncing hundreds of
megabytes of wheels is slow, and a synced venv breaks the moment the checkout moves,
because editable installs and console-script shims hardcode absolute paths.

The sibling folders the server scripts expect, all next to each other under one `Code/`
directory:

```
Code/
├── OcularMotorSim/      ← this repo (may be synced)
├── om-stable/           ← git worktree, frozen snapshot for the public site (optional)
├── om-data/             ← request databases (dev-data/ and stable-data/)
└── om-lab-website/      ← public website repo (optional, only for deploying)
```

Only `OcularMotorSim/` is needed to run simulations. The rest are for the servers.

---

## 1. Prerequisites

| | |
|---|---|
| **Python** | ≥ 3.10 (3.12 is what's in use) |
| **git** | any recent version |
| **C compiler** | not needed — JAX ships wheels for macOS arm64, Linux x86-64 and Windows |

**Apple Silicon note.** JAX runs on the **CPU** by default and that is the supported path
here. `jax-metal` exists but is experimental and does not cover everything Diffrax needs —
don't install it expecting a speedup. Verify what you got in step 4; it should say
`CpuDevice`.

---

## 2. Get the code

**Preferred — clone:**

```bash
git clone https://github.com/jotero/OcularMotorSim.git
cd OcularMotorSim
git checkout sensory-periphery-refactor      # or main
```

**Alternative — folder sync (OneDrive et al.).** Works, but set this *before* running any
git command, or the first `git status` reports every file as modified:

```bash
git config core.autocrlf true
```

The repo is committed with LF blobs and checked out CRLF on Windows (`core.autocrlf=true`
there, no `.gitattributes`). macOS git defaults to `false`, so it compares CRLF on disk
against LF in the index and sees the whole tree as changed. Matching the setting on both
machines avoids it.

Also, if you sync `.git` itself: **only one machine may touch the repo at a time.** Cloud
sync has no file locking, and two machines writing `.git` concurrently produces conflicted
copies of `index`, stray `index.lock` files, or half-written packfiles.

---

## 3. Create the venv and install

**macOS / Linux**

```bash
python3 -m venv ~/om-venvs/OcularMotorSim
source ~/om-venvs/OcularMotorSim/bin/activate
pip install -U pip
pip install -e ".[all]"
```

**Windows (PowerShell)**

```powershell
python -m venv "$env:LOCALAPPDATA\om-venvs\OcularMotorSim"
& "$env:LOCALAPPDATA\om-venvs\OcularMotorSim\Scripts\Activate.ps1"
pip install -U pip
pip install -e ".[all]"
```

`[all]` pulls the LLM pipeline and web server extras on top of the core model. For the
model alone (`models`, `sim`, `analysis`), plain `pip install -e .` is enough.

The venv root is configurable via `OCULOMOTOR_VENV_ROOT` if you want it elsewhere; the
Windows `server.ps1` reads it.

---

## 4. Verify the install

```bash
python -X utf8 scratch/perf_baseline.py
```

`-X utf8` prevents cp1252 crashes on Windows when scripts print Greek letters; harmless
elsewhere, so it's used everywhere for consistency.

Expect something like:

```
machine   : arm64  Darwin 24.x
devices   : [CpuDevice(id=0)]          ← CPU is correct, see §1
oculomotor: <git-describe string>
compile+first run :    25.60 s
solve (best of 6) :     1.011 s   (1,978 ODE steps/s)
```

Those two timings are the reference numbers from an Intel Xeon (Skylake-SP) box — use them
to compare machines. **Compile and solve are timed separately on purpose:** JAX traces this
~400-state ODE once per array shape, and that one-off cost dominates short runs. Only the
solve figure predicts how fast a real parameter sweep will be.

Second check, this one needs no simulation at all (~2 s):

```bash
python -X utf8 -m oculomotor.benchmarks.bench_metrics
```

It scores the stored benchmark data and prints a pass/fail scoreboard. If it renders, the
package imports and the data files are intact.

---

## 5. Run simulations and benchmarks

```bash
# one validation area (figures land in web/benchmarks/figures/)
python -X utf8 -m oculomotor.benchmarks.bench_<area>

# The 11 areas in the scored suite (BENCH_MODULES in bench_metrics.py):
#   saccades  vor_okr  gravity  pursuit  vergence  fixation
#   gaze_holding  listing  fcp  pupil  eyelid
#     - gravity also renders the T-VOR figures (merged section)
#
# Runnable but NOT in the scored suite (figures only, no metrics):
#   clinical  clinical_cerebellum  clinical_cn_palsies  clinical_ni_vs
#   clinical_pupil  clinical_saccades  clinical_vergence  clinical_vestibular
#   experiments  tvor
#   accommodation   <- orphaned; superseded by bench_vergence's near_response

# score everything offline (fast, no ODE solves)
python -X utf8 -m oculomotor.benchmarks.bench_metrics
python -X utf8 -m oculomotor.benchmarks.bench_metrics --fails
python -X utf8 -m oculomotor.benchmarks.bench_metrics --section saccades

# re-measure, then score  (one section, or all with a bare --run)
python -X utf8 -m oculomotor.benchmarks.bench_metrics --run saccades
python -X utf8 -m oculomotor.benchmarks.bench_metrics --run          # ~60 min
```

**The full suite takes about an hour** (`vor_okr` alone is ~23 min), so re-measure only the
section you changed. See CLAUDE.md § "Evaluating the benchmarks" for what PASS / FAIL /
DRIFT / NEW mean and how staleness is judged.

Regenerating the docs pages:

```bash
python -X utf8 -m oculomotor.reports.gen_parameters      # web/parameters.html
python -X utf8 -m oculomotor.reports.gen_states          # web/states.html
python -X utf8 -m oculomotor.reports.run_benchmarks --html-only   # bench gallery index
```

---

## 6. The web servers

Two servers, deliberately separate:

| | port | code | data |
|---|---|---|---|
| **dev** | 8001 | this checkout, live | `om-data/dev-data` |
| **stable** | 8000 | `om-stable/` worktree, its own venv, frozen | `om-data/stable-data` |

They have separate databases, pinned via `OCULOMOTOR_DATA`, so both can run at once and
neither depends on the launch directory.

### macOS / Linux

There is no shell-script equivalent of `server.ps1` yet — run the module directly:

```bash
# dev server (live code)
export OCULOMOTOR_DATA=../om-data/dev-data       # absolute path is safer
python -X utf8 -m oculomotor.server --port 8001

# stable server, from the frozen worktree with its own venv
cd ../om-stable
export OCULOMOTOR_DATA=../om-data/stable-data
~/om-venvs/OcularMotorSim-stable/bin/python -X utf8 -m oculomotor.server --port 8000
```

Create `om-data/dev-data` and `om-data/stable-data` first — `mkdir -p ../om-data/dev-data`.

### Windows

```powershell
.\server.ps1 dev          # http://localhost:8001, live code
.\server.ps1 stable       # http://localhost:8000, frozen worktree
.\server.ps1 make-stable  # snapshot HEAD onto the stable branch + reinstall its venv
.\server.ps1 up           # tunnel + both servers, each in its own terminal tab
.\server.ps1 help
```

### What's served

- `/` — the simulator front end
- `/benchmarks/` — the benchmark gallery (static files from `web/`, *not* live simulation;
  it shows whatever the last bench run wrote to disk)
- `/parameters.html`, `/states.html` — generated reference pages
- `/download/{run_id}` — CSV of a run

Environment variables the server reads: `OCULOMOTOR_DATA` (request DB location),
`OCULOMOTOR_WEB` (override the `web/` directory), `ADMIN_TOKEN` (admin endpoints).

### Public site (optional)

`sim.oteromillan.com` needs **both** the stable server on 8000 and a Cloudflare tunnel
pointing at it. On Windows, `.\server.ps1 cloudflare` runs the tunnel; elsewhere run
`cloudflared` yourself against `localhost:8000`.

---

## 7. LLM pipeline

Needs an Anthropic API key, in the environment or in a `.env` file at the repo root:

```
ANTHROPIC_API_KEY=sk-ant-...
```

`.env` is gitignored and does **not** travel with a clone — copy it across by hand, or
recreate it. Then:

```bash
python -X utf8 -m oculomotor.llm_pipeline.cli "healthy subject making a 20 deg saccade right"
python -X utf8 -m oculomotor.llm_pipeline.cli --dry-run "..."   # no API call
python -X utf8 -m oculomotor.llm_pipeline.cli --json scenario.json
# or the console script:  oculomotor-simulate "..."
```

---

## 8. Claude Code working memory (optional)

If you use Claude Code on this project, the per-project memory notes are **not** in the
repo — they live under your home directory and do not sync:

```
~/.claude/projects/<slug>/memory/
```

The `<slug>` is derived from the absolute project path, so it differs per machine. To carry
the accumulated context over, copy `MEMORY.md` and the `*.md` notes from the old machine's
slug directory into the new one. Transcripts are pruned after `cleanupPeriodDays` (default
30) — raise it in `~/.claude/settings.json` if you want them kept:

```json
{ "cleanupPeriodDays": 365 }
```

---

## 9. Gotchas

| Symptom | Cause / fix |
|---|---|
| Every file shows as modified after a sync | CRLF mismatch — `git config core.autocrlf true` (§2) |
| `UnicodeEncodeError` printing Greek letters | Windows cp1252 — always pass `-X utf8` |
| Venv breaks after moving the checkout | Editable installs hardcode paths — recreate the venv (§0) |
| First simulation takes ~25 s | JAX tracing, once per array shape. Normal, not a hang |
| `devices: [MetalDevice]` on a Mac | `jax-metal` got installed; uninstall it (§1) |
| Bench gallery shows stale figures | It's static files — re-run the bench, then `run_benchmarks --html-only` |
| Server 404s on everything | Wrong `OCULOMOTOR_WEB`, or launched from a directory with no `web/` |
| dev and stable fighting over data | `OCULOMOTOR_DATA` not set — both default to `server_data/` |
| `.git` corruption after sync | Two machines touched the repo at once (§2) |
