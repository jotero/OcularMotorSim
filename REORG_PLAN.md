# Repo reorganization plan (v2)

> The single destination to build toward. v2 (2026-06-13) replaces the v1 "separate
> `webapp/`" sketch after we locked a cleaner spine: **all important Python lives in the
> package; only debug lives outside.** Check items off as they land.

---

## The spine (locked)

**Everything important is Python-in-`src`, organized into layered subpackages. Only debug code
lives outside `src`. The served website (client + its generated cache) is the one non-Python tree.**

Four layers: **(1) model → (2) model-adjacent tools → (3) server → (4) client**, and the client is a
*dynamic site with a cache* (server computes the interactive LLM sim per request; the parameter/state/
benchmark pages are pre-rendered and cached).

---

## Target layout

```
ClaudeOculomotorJax/
├── src/oculomotor/            ALL important Python
│   ├── models/  sim/  params.py        L1  the differentiable model
│   ├── analysis.py  fitting/           L2  model-adjacent tools (importable)
│   ├── benchmarks/                     L2  the bench_* suite            ← from scripts/
│   ├── reports/                        L2  gen_* + run_*benchmarks → write the cache ← from scripts/
│   ├── llm_pipeline/                   L3  LLM interface (prompt/interpret/scenario/
│   │                                       patient_builder/run/cli)  [renamed inside ✓]
│   ├── server/                         L3  FastAPI backend             ← from scripts/server.py
│   └── schema/                         config/schemas as PACKAGE-DATA   ← from top-level schema/
├── web/                       L4  served site: client shell + generated cache, TOGETHER, tracked
├── scratch/                   debug-only scripts (all diag_* and _*)    ← from scripts/
├── data/                      request DB — gitignored (OneDrive risk accepted; via OCULOMOTOR_DATA)
├── outputs/                   other generated run artifacts (gitignored)
├── tests/  manuscript/  references/   (references = renamed papers/)
├── docs/                      (optional) real human docs — design notes, MAP
└── root: pyproject.toml LICENSE CLAUDE.md MAP.md REORG_PLAN.md server.ps1
```

`scripts/` dissolves entirely → into `src/` subpackages (entry points become console scripts)
and `scratch/` (debug).

---

## §1 — DONE (uncommitted)

- [x] `MAP.md` created; pointer atop `CLAUDE.md`.
- [x] Deleted stale `PLAN_orbital_limits.md`; deleted stray tracked `src/outputs/…png`.
- [x] **`docs/` → `web/`** — it's the served frontend. ~16 source files repointed; verified.
- [x] **`server.ps1`** dispatcher (`dev`/`stable`/`make-stable`, no-arg = help); deleted the 3 old `.ps1`.
- [x] **`schema/`** created top-level (interim) + 4 readers repointed. *(Will move again → §2.2 package-data.)*
- [x] **`pipeline` internals renamed** — `prompt` / `interpret` / `scenario` / `patient_builder` / `run` / `cli`
      (was `runner.py` + `simulate.py`); all refs + `CLAUDE.md` updated; verified at runtime.
- [x] **Fixed cli.py output-path bug** — was writing to `src/outputs/` (3 levels vs 4 under `src/` layout);
      now resolves to repo-root `outputs/`. Partly closes the old §4 outputs leak.

**→ Commit this as a checkpoint before §2.**

---

## §2 — move all Python into the package

Order matters (imports). Each move: `git mv`, then repoint imports, then verify.

- [x] **2.1 `scripts/server.py` → `src/oculomotor/server/`** (`app.py` + `admin.py` + `__init__`/`__main__`). DONE.
      - `gen_admin` → `server/admin.py` (server-coupled, not reports). Env-driven paths: `OCULOMOTOR_DATA`
        (default `<checkout>/data`), `OCULOMOTOR_WEB` (default `<checkout>/web`); self-locate via `parents[3]`.
      - Run `python -m oculomotor.server` / console `oculomotor-server`. Verified: dev serves frontend (200).
- [x] **2.2 `schema/*.yaml` → `src/oculomotor/schema/`** as package-data (`importlib.resources`). DONE.
      - Repointed readers `patient_builder`, `server`, `gen_parameters`, `gen_states` via `importlib.resources.files`.
      - **Removed the `patient_builder` parent-walk leak** (reads its own package data); verified.
      - `pyproject.toml`: added `[tool.setuptools.package-data] "oculomotor.schema"=["*.yaml"]` + `pyyaml` to extras.
- [x] **2.3 `scripts/bench_*` → `src/oculomotor/benchmarks/`** (19 files incl. `bench_utils`, `bench_clinical_utils`). DONE.
      - Rewired `import bench_utils as utils` → `from oculomotor.benchmarks import ...`; killed the `sys.path` hack;
        repo-root path now 3-up; `run_benchmarks`/`run_clinical`/`sweep_occlusion` import via `oculomotor.benchmarks.*`.
      - Run via `python -m oculomotor.benchmarks.bench_<area>`; figures still → `web/` cache. Verified imports.
- [x] **2.4 web-cache builders → `src/oculomotor/reports/`** (7 files). DONE.
      - Moved `gen_parameters`, `gen_states`, `run_benchmarks`, `run_clinical_benchmarks`,
        `freeze_reference`, `sweep_occlusion`, `block_diagram`. Repo-root paths now 3-up; `sys.path`
        hacks removed; `run_benchmarks` imports `gen_parameters` via `oculomotor.reports`. Verified imports.
      - **Excluded:** `gen_admin` (server-coupled → moves with `server`); `run_recovery` (a fitting
        experiment, not a cache builder → `scratch`/fitting later). Both remain in `scripts/` for now.
- [x] **2.5 retire `scripts/`** — DONE. Removed `simulate.py` shim (use `oculomotor-simulate` /
      `python -m oculomotor.llm_pipeline.cli`); moved `run_recovery.py` → `scratch/`. `scripts/` deleted.
      Console scripts added to `pyproject.toml`: `oculomotor-server`, `oculomotor-simulate`.
- [x] **Deploy wiring** — `server.ps1` rewritten: `dev` = main venv / live; `stable` = `om-stable`
      worktree's OWN venv (frozen model, isolated from dev); `make-stable` snapshots + (re)installs the
      worktree venv (`pip install -e ".[all]"`) + copies `.env` + deploys `web/`. Stable is now *truly* frozen.

---

## §3 — debug + small cleanups

- [x] **`scratch/`** — moved all `diag_*` + `_*` (22 files), tracked, + `scratch/README.md`. DONE.
      (Kept the dead `*2/*3` duplicates for now — prune later.)
- [x] **`data/`** — DONE. The request-DB home (gitignored). Server reads `OCULOMOTOR_DATA`
      (default `<checkout>/data`). Dev and stable get separate DBs by construction (separate checkouts);
      override `OCULOMOTOR_DATA` to relocate (e.g. off OneDrive) or unify. **NOTE:** the old shared
      `~/oculomotor_outputs` DB is no longer read — copy it into `data/` if you want the prior history.
- [x] **`papers/` → `references/`** — DONE (pure rename, no code refs).
- [ ] **`docs/`** now free — optional human-docs home (could absorb `OVAR_DIAGNOSIS_NOTES.md`,
      `EXPERIMENTS.md`, `web/cerebellum.md`, `web/plant_compensation.md` — the last is referenced by
      `final_common_pathway.py:100`, update if moved).

---

## §4 — packaging (encodes "one package, ship-subset gated by extras")

- [ ] `pyproject.toml`:
      ```toml
      dependencies = ["jax","jaxlib","diffrax","optax","numpy","matplotlib"]   # model core
      [project.optional-dependencies]
      llm    = ["anthropic","pydantic","python-dotenv"]   # llm_pipeline
      server = ["fastapi","uvicorn"]                      # server
      [project.scripts]
      oculomotor-server   = "oculomotor.server:main"
      oculomotor-simulate = "oculomotor.llm_pipeline.cli:main"
      ```
      - Optional-dep modules (`llm_pipeline`, `server`) guard their imports so the bare model installs clean.
      - The "pure model" = `oculomotor.models` + `oculomotor.sim` + `oculomotor.analysis` with core deps only.
      - **DONE:** deps split into core + `[llm]`/`[server]`/`[all]` extras; build verified. Console
        `[project.scripts]` deferred until `server`/`cli` modules move (§2). Dev install: `pip install -e ".[all]"`.
- [x] Keep static `version = "0.1.0"` (the PEP 440 *build* version). NOT dead — `oculomotor.__version__`
      (git-describe string, e.g. `5030944-dirty`) is the *runtime* version and is NOT PEP 440, so it
      can't be the package version. They legitimately coexist. (Confirmed: dynamic attr build fails on it.)
- [ ] Remaining leak: `llm_pipeline/cli.py` still parent-walks to `outputs/`. Optional: make `out_dir` an arg.
      (`__init__.py` `git describe` — benign, leave.)

---

## Execution order

1. **Commit §1** (checkpoint).
2. **§4 pyproject** extras + dead-version — cheap, no moves, encodes the boundary.
3. **§3 small** — `data/` config + `papers→references` + `scratch/` for debug.
4. **§2 the package consolidation** — server → schema → benchmarks → reports → cli/console-scripts.
   Verify imports after each.
5. Update `MAP.md` landscape + memory to the final layout.
```
