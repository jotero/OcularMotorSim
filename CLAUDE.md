# ClaudeOculomotorJax — Project Context for Claude

> **Lost? Start at [MAP.md](MAP.md)** — the project atlas: subsystem router (code ↔ design note ↔
> manuscript ↔ bench), a live status dashboard, and a mirror of Claude's private memory.
> This file (CLAUDE.md) is the deep architecture reference; MAP.md is the index over it.

## Shared analysis utilities — no redundant helpers

`src/oculomotor/analysis.py` is the single source of truth for post-hoc signal extraction and plotting
helpers. **Never redefine these locally in a demo script or notebook cell:**

| Function | What it gives you |
|---|---|
| `vs_net(states)` | VS net signal x_L − x_R, (T, 3), deg/s |
| `ni_net(states)` | NI net signal x_L − x_R, (T, 3), deg |
| `vs_null(states)` | VS null-adaptation state, (T, 3) |
| `ni_null(states)` | NI null-adaptation state, (T, 3) |
| `extract_burst(states, theta)` | u_burst via vmap, (T, 3) |
| `extract_sg(states, theta)` | All SG sub-states dict |
| `extract_canal(states)` | Canal yaw estimate, (T,) |
| `extract_spv(t, ev, burst)` | Slow-phase velocity via burst mask |
| `fit_tc(t, y, t_start, t_end)` | Exponential TC fit |
| `ax_fmt(ax, ylabel, xlabel, ylim)` | Standard axes formatting |

When you need a yaw-only scalar in a notebook, use a thin one-liner wrapper:
```python
def vs_net_yaw(states): return _vs_net3(states)[:, 0]
```
Do **not** reimplement the logic.

## Running scripts

Always use `-X utf8` to avoid Windows cp1252 encoding errors (Greek letters in print statements crash otherwise):

```bash
"d:/OneDrive/UC Berkeley/OMlab - JOM/Code/ClaudeOculomotorJax/.venv/Scripts/python.exe" -X utf8 -m oculomotor.benchmarks.bench_vor_okr
```

Or from PowerShell:

```powershell
& "d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\.venv\Scripts\python.exe" -X utf8 -m oculomotor.benchmarks.bench_vor_okr
```

(Benchmarks live in `src/oculomotor/benchmarks/` and run as modules: `-m oculomotor.benchmarks.bench_<area>`.)

## Stimulus conventions

Keep all pathways active by default. Only disable a pathway (e.g. `g_burst=0`, `K_vis=0`, `scene_present=0`) when the demo is explicitly testing what happens *without* that pathway — and add a comment explaining why.

In particular:
- **Saccade demos**: keep `scene_present=1` (the animal sees the target in a lit room). Do not zero out OKR gains.
- **VOR demos**: keep saccades on (`g_burst=700`) unless the demo is specifically about the canal→VS cascade without fast phases.
- **OKR demos**: keep saccades on so the nystagmus sawtooth is visible.

When a panel looks wrong, diagnose via simulation before disabling pathways. Autoscale artifacts (matplotlib scaling noise to a salient-looking signal) are a common false alarm — enforce a minimum y-axis range for velocity panels.

## What this is

A JAX-based simulation of the primate oculomotor system. The goal is a differentiable, biophysically grounded model of how the brain controls eye movements — suitable for fitting to experimental data.

## Architecture

Signal flow: `Head velocity → Canal Array → Velocity Storage → Neural Integrator → Plant → Eye position`

Visual pathway: `Per-eye retinal slip → retina.step (sharp gamma cascade) → perception_cyclopean (binocular fusion + brain LP) → VS / pursuit / SG / vergence`

Saccade loop: `Retinal position error → retina + cyclopean → SG (Robinson local-feedback) → NI + Plant`

Pursuit loop: `Retinal velocity error → retina + cyclopean → Pursuit integrator (Smith predictor) → NI + Plant`

### Folder structure (`src/oculomotor/`)

The package now lives under `src/` — imports are still `from oculomotor...`.

```
src/oculomotor/
├── __init__.py                        __version__ from git describe --tags --always --dirty
├── params.py                          Public re-export of BrainParams/SensoryParams/PlantParams
├── analysis.py                        Post-hoc extraction helpers (vs_net, ni_net, extract_sg, …)
├── models/
│   ├── sensory_models/                Peripheral sensors only — canal, otolith, retina geometry
│   │   ├── canal.py                   Canal array SSM (Steinhausen, 6 canals, 12 states)
│   │   ├── otolith.py                 Otolith SSM (bilateral LP adaptation, 6 states)
│   │   ├── retina.py                  Geometry (world_to_retina) + sensor saturation +
│   │   │                              per-eye sharp gamma cascade (90 states/eye).
│   │   │                              retina.step → RetinaOut (delayed per-eye signals).
│   │   └── sensory_model.py           Connector: canal + otolith + retina_L + retina_R →
│   │                                   SensoryOutput {canal, otolith, retina_L, retina_R}
│   │                                   (198 states)
│   ├── brain_models/                  Cortical computations — operate on already-delayed signals
│   │   ├── perception_self_motion.py  VS + GE + HE unified observer (Laurens & Angelaki 2017)
│   │   ├── perception_target.py       Target gates + working memory (FEF/dlPFC layer)
│   │   ├── perception_cyclopean.py    Binocular fusion (NPC gate, OKR, dominance) on delayed
│   │   │                              per-eye signals + brain LP smoothing (43 states).
│   │   ├── neural_integrator.py       NI leaky integrator + bilateral push-pull + null adapt
│   │   │                              (12 states: x_L + x_R + x_null + u_lp)
│   │   ├── saccade_generator.py       Robinson local-feedback burst (20 states: e_held +
│   │   │                              z_opn + z_acc + z_trig + z_fac/z_dep + EBN_L/R + IBN_L/R)
│   │   ├── pursuit.py                 Smooth pursuit + Smith predictor (6 states)
│   │   ├── tvor.py                    Translational VOR (stateless)
│   │   ├── vergence_accommodation.py  Vergence + accommodation, AC/A + CA/C cross-links (11 states)
│   │   ├── listing.py                 Listing's-law torsion corrections (smooth pathways)
│   │   ├── final_common_pathway.py    14-state FCP: 12 MN dynamic states (tau_mn ~5 ms) +
│   │   │                              MLF AIN→CN3_MR pathway. Separable lesions: g_nucleus,
│   │   │                              g_mlf, g_nerve. Replaces old algebraic relu+clip.
│   │   ├── cerebellum.py              EC delay cascades + flocculus (FL) NI extension +
│   │   │                              paraflocculus (VPF) pursuit forward model.
│   │   │                              Replaces the deleted efference_copy.py.
│   │   ├── unified_brain.py           Experimental matrix-form rewrite of brain_model.step
│   │   │                              (active experiment, not canonical).
│   │   └── brain_model.py             Connector: cyclopean → perception → SG/pursuit/TVOR/NI
│   │                                   → vergence/acc → FCP → cerebellum (EC).
│   │                                   BrainState is a NamedTuple of sub-state NamedTuples —
│   │                                   no flat array, no _IDX_* constants.
│   ├── plant_models/
│   │   ├── plant_model_first_order.py First-order plant (Robinson 1964) (3 states/eye)
│   │   ├── accommodation_plant.py     Lens / ciliary muscle LP (1 state, tau_acc_plant ~0.156 s)
│   │   ├── muscle_geometry.py         12-muscle pulling-direction matrix + M_NUCLEUS / M_NERVE_PROJ
│   │   └── readout.py                 Eye position readout + rotation_matrix()
│   ├── fitting/                       Gradient-based fitting (loss + optimize) — ⚠️ OBSOLETE (VOR-only,
│   │                                   pre-NamedTuple; rewrite before use — see fitting/__init__.py)
│   └── llm_pipeline/                  Natural-language → simulation pipeline
│       ├── prompt.py                  The system prompt sent to Claude (edit to tune interpretation)
│       ├── interpret.py               NL description → SimulationScenario via Claude API (call_llm)
│       ├── scenario.py                Pydantic schema (SimulationScenario, Patient, …)
│       ├── patient_builder.py         YAML-driven Patient construction
│       ├── run.py                     Stimulus builder + simulator wiring + figure generator (run_scenario)
│       └── cli.py                     Command-line entry point (main)
├── benchmarks/                       Validation suite (bench_*); run: python -m oculomotor.benchmarks.bench_<area>
├── reports/                          Web-cache builders (gen_parameters/states, run_*benchmarks, freeze_reference, sweep_occlusion, block_diagram)
├── schema/                           Bundled config yaml as package-data (parameters_schema, states_schema)
└── sim/
    ├── simulator.py                   ODE wiring + simulate() entry point.
    │                                   Brain step is swappable via set_brain_step()
    │                                   (unified_brain.step is a drop-in alternative).
    ├── kinematics.py                  KinematicTrajectory / TargetTrajectory builders
    ├── stimuli.py                     Centralized stimulus generators
    └── synthetic.py                   Synthetic-data helpers for fitting
```

### Running things (scripts/ has been retired)

Everything runnable now lives in the package and runs as a module (or console script):
- **Server** → `.\server.ps1 dev` (port 8001, live) / `.\server.ps1 stable` (port 8000, frozen worktree venv);
  or directly `python -m oculomotor.server [--port N]`. (See `server.ps1` / the Web-server section below.)
- **LLM CLI** → `python -m oculomotor.llm_pipeline.cli "<scenario>"` (or the `oculomotor-simulate` console script).
- **Benchmarks** → `python -m oculomotor.benchmarks.bench_<area>`
  (vor_okr, saccades, pursuit, fixation, vergence, accommodation, gravity, listing, tvor, experiments,
  bench_clinical[_cerebellum|_cn_palsies|_ni_vs|_saccades|_vergence|_vestibular]). Figures → `web/` cache.
- **Report/cache builders** → `python -m oculomotor.reports.<name>`
  (gen_parameters, gen_states, run_benchmarks, run_clinical_benchmarks, freeze_reference, sweep_occlusion, block_diagram).

Debug/diagnostic scripts (`diag_*`, `_*`) live in **`scratch/`** (tracked, unmaintained — see `scratch/README.md`).
Reference literature PDFs are in **`references/`** (was `papers/`); your own writing is in `manuscripts/`.

### State structure (binocular)

`SimState` is a NamedTuple of three groups. The brain group is itself a `BrainState`
NamedTuple of per-subsystem `State` NamedTuples — there is **no flat brain array
and no `_IDX_*` slice constants** any more. Diffrax handles arbitrary PyTrees
natively, so subsystems are accessed as attributes.

```
SimState(
    sensory   : SensoryState   (198 states)
    brain     : BrainState     (≈170 states; see breakdown)
    plant     : PlantState     (6 states, plus accommodation 1 state)
)
```

**Sensory** (198 states; same shape as before):
```
sensory: [x_c (12) | x_oto (6) | x_retina_L (90) | x_retina_R (90)]
```
`x_retina_<L|R>` (per eye, sharp gamma cascade, N=6 stages each):
`[scene_angular_vel(18) | scene_linear_vel(18) | target_pos(18) | target_vel(18) |
  scene_visible(6) | target_visible(6) | defocus(6)]`

**Brain** (nested NamedTuple — `BrainState`):
```python
BrainState(
    pc:   pc.State    # 43  perception_cyclopean (binocular fusion + brain LP)
    sm:   sm.State    # 21  self-motion observer (VS bilateral + GE + HE).
                      #     VS pops are CANAL-PLANE [H, LARP, RALP] (H→MVN, LARP/RALP→SVN)
    pt:   pt.State    #  4  target working memory  (x_mem(3) + trust(1))
    sg:   sg.State    # 20  saccade generator (see saccade_generator.py)
    pu:   pu.State    #  6  bilateral pursuit pops
    va:   va.State    # 11  vergence (9) + accommodation (2)
    ni:   ni.State    # 12  bilateral NI (x_L + x_R + x_null + u_lp), CANAL-PLANE
                      #     [H, LARP, RALP]: H→NPH, LARP/RALP→INC
    fcp:  fcp.State   # 14  12 MN dynamic states + MLF AIN→CN3_MR
    cb:   cb.State    # ≈56 EC scene + target delay cascades + sat-flag delays +
                      #     near-response accom + verg-H EC cascades (Smith forward models)
)
```
Subsystems are read directly as `brain_state.<sub>.<field>` — never via index
slicing. `brain_model.N_STATES` is computed from sub-sums and kept only for
legacy info; do not rely on a specific total.

**Cross-subsystem reads in `brain_model.step` MUST go through the registries**
(`Activations`, `Decoded`, `Weights` returned by `read_activations`,
`decode_activations`, `read_weights`) — not via raw state field access from a
different subsystem.

**Plant** (6 eye states + 1 accommodation):
```
plant: [x_p_L (3) | x_p_R (3) | x_acc_plant (1)]
```

Cyclopean delayed signals are read via `perception_cyclopean.C_*` matrices on
`brain_state.pc` (NOT on sensory state).

### Params structure

Parameters are nested NamedTuples — not dicts. Access via attribute path:

```python
class SensoryParams(NamedTuple):
    # Sensor-side only — peripheral physiology.
    tau_c, tau_s, canal_gains, canal_floor, canal_v_max, tau_oto,         # canals + otolith
    tau_vis_sharp,                                                        # retina sharp delay
    v_max_target_vel, v_max_scene_vel,                                    # MT/MST + NOT/AOS ceilings
    visual_field_limit, k_visual_field,                                   # eccentricity gate
    sigma_canal, sigma_slip, sigma_pos, sigma_vel, tau_*_drift,           # noise (OU)
    ipd                                                                   # binocular geometry

class PlantParams(NamedTuple):
    tau_p

class BrainParams(NamedTuple):
    # Cortical + brainstem parameters (selected highlights — full list in brain_model.py):
    tau_vs, tau_vs_vert_frac, g_vor, b_vs, tau_vs_adapt,   # VS (H + one vertical/torsional frac)
    tau_i, b_ni, tau_ni_adapt,                                    # NI
    tau_vis_sharp, tau_vis_smooth_motion, tau_vis_smooth_target_vel,
    tau_vis_smooth_disparity, tau_vis_smooth_defocus,             # brain-side LP TCs
    npc, div_max, vert_max, tors_max, eye_dominant,               # binocular fusion policy
    g_burst, e_sat_sac, k_sac, threshold_sac,                     # saccade generator
    tau_fac, tau_dep, alpha_fac, alpha_dep, g_ibn_opn,            # BN facilitation/depression + IBN→OPN
    saccadic_suppression_threshold, saccadic_suppression_steepness,  # cerebellar EC saccade gate
    tau_mn,                                                       # motor neurons
    g_nucleus, g_mlf_L, g_mlf_R, g_nerve,                         # FCP lesion knobs
    K_cereb_fl, K_cereb_pu,                                       # cerebellar gains (FL + VPF)
    K_pursuit, K_phasic_pursuit, tau_pursuit,                     # pursuit
    tonic_verg, K_verg, K_verg_tonic, K_phasic_verg,              # vergence
    aca_ratio, cac_ratio, tau_acc_fast, tau_acc_slow,             # accommodation + cross-coupling
    K_grav, K_gd, g_ocr, orbital_limit                            # gravity / OCR

class Params(NamedTuple):
    sensory: SensoryParams = SensoryParams()
    plant:   PlantParams   = PlantParams()
    brain:   BrainParams   = BrainParams()
```

For a fully-defaulted Params (with `tonic_verg` derived from IPD), use
`default_params()` from `oculomotor.sim.simulator`.

Use `with_sensory(params, sigma_canal=2.0)` / `with_brain(params, tau_vs=15.0)` / `with_plant(params, tau_p=0.2)` to create modified copies.

### Solver

`diffrax.Heun()` fixed step, `dt = 0.001 s`. Must satisfy `dt < 2 * tau_stage_vis = 0.004 s`.

### Sensory noise

Four independent noise sources, **non-zero by default** (so a vanilla `simulate(PARAMS_DEFAULT, ...)` already produces realistic fixational drift, microsaccades, and pursuit jitter). All four are Ornstein-Uhlenbeck processes — small τ approaches band-limited white noise, longer τ produces drift-like fluctuations. Defaults from `SensoryParams`:

| σ param      | default     | τ param            | default | what it drives |
|--------------|-------------|--------------------|---------|----------------|
| `sigma_canal`  | 1.0 deg/s | `tau_canal_drift`  | 0.005 s | canal afferent noise; filtered by VS/NI/plant |
| `sigma_slip`   | 0.0 deg/s | `tau_slip_drift`   | 0.005 s | retinal slip noise (off by default); VS/OKR |
| `sigma_pos`    | 0.2 deg   | `tau_pos_drift`    | 0.2 s   | retinal position drift; **triggers microsaccades** |
| `sigma_vel`    | 1.0 deg/s | `tau_vel_drift`    | 0.005 s | retinal velocity noise; pursuit integrator |

`SG_acc` accumulator diffusion (`sigma_acc=0.2`, in `BrainParams`) adds RT variability to saccade triggering.

Noise is pre-generated as arrays before `diffeqsolve` and passed as `LinearInterpolation` inputs — ODE remains pure and differentiable.

```python
params = with_sensory(PARAMS_DEFAULT,
    sigma_canal    = 2.0,   # crank up canal noise
    sigma_pos      = 0.0,   # disable microsaccades for clean cascade traces
    ...
)
states = simulate(params, t, ..., key=jax.random.PRNGKey(42))
```

`sigma_pos` uses an Ornstein-Uhlenbeck process (not white noise) so drift accumulates slowly,
crosses the SG threshold occasionally, and triggers sparse corrective microsaccades.
White noise on `pos_delayed` would fire the SG continuously.

**Debug benches that need noiseless traces** (e.g. cascade figures, symmetric-vergence triggers, anything where you need exact bilateral cancellation) must explicitly disable the relevant noise sources via `with_sensory(...sigma_canal=0, sigma_pos=0, sigma_vel=0)` and `with_brain(...sigma_acc=0)`. By convention these go in the bench's top-level `PARAMS_*` constant alongside any other overrides, so the figure footer (params overrides line) makes them visible.

### Versioning

`oculomotor.__version__` is derived from `git describe --tags --always --dirty` at import time.
No manual version bumping required — tag a release with `git tag v1.0` and it appears automatically.
The version string is logged with every server simulation call.

### What "correct behavior" looks like

Each behavior has a corresponding demo script and output figure.

1. **VOR in the dark** — eye velocity ≈ −head velocity; gain ~0.9–1.0. Canal adaptation TC (~5 s) causes the VOR to decay during sustained rotation; velocity storage extends the effective TC to ~15–20 s.
   - Demo: `oculomotor/benchmarks/bench_vor_okr.py` → `outputs/vor_dark.png`

2. **Velocity storage / TC extension** — during constant-velocity rotation in the dark, eye velocity decays with TC ~15–20 s (not the canal TC of ~5 s). VVOR: in a stationary lit world, OKR corrects VOR slip as the canal adapts — gaze stays stable throughout.
   - Demo: `oculomotor/benchmarks/bench_vor_okr.py` → `outputs/vvor.png`

3. **OKN + OKAN** — during full-field visual motion, steady-state OKN gain ≈ 1. After scene off, OKAN persists with TC ~20 s (`tau_vs`). With saccades on, eye shows sawtooth nystagmus.
   - Demo: `oculomotor/benchmarks/bench_vor_okr.py` → `outputs/okr.png`

4. **Saccades — main sequence + refractory period** — peak velocity follows `v_peak ≈ 700·(1−exp(−A/7))`, saturating ~600–700 deg/s. Robust intersaccadic interval (~150–200 ms). Oblique saccades straight with synchronized components.
   - Demo: `oculomotor/benchmarks/bench_saccades.py` → `outputs/saccade_summary.png`

5. **Smooth pursuit** — foveal target tracking via MT/MST velocity pathway. Pursuit integrator + Smith predictor (efference copy cancels saccadic contamination). Catch-up saccades fire when position error exceeds threshold during ramp pursuit.
   - Demo: `oculomotor/benchmarks/bench_pursuit.py` → `outputs/smooth_pursuit.png`

6. **Saccades during head movement** — corrective saccades fire periodically as VOR slip accumulates; staircase toward target.
   - Demo: `oculomotor/benchmarks/bench_saccades.py` → `outputs/vor_saccade_cascade.png`

7. **Efference copy** — burst commands must not contaminate VS/OKR. Verified inside the VOR/OKR cascade plot in `bench_vor_okr.py` and the saccade cascade in `bench_saccades.py`.

8. **Fixational eye movements** — canal noise filtered by VS/NI/plant; retinal position OU drift produces sparse corrective microsaccades; retinal velocity noise drives pursuit-like slow drift.
   - Demo: `oculomotor/benchmarks/bench_fixation.py` → `outputs/fixation.png`

## Current status (2026-05-25)

- **Working well**: VOR, VVOR, OKN/OKAN, saccades (main sequence, refractory, oblique), smooth pursuit (velocity-driven + Smith predictor), otolith LP adaptation, sensory noise system, fixational eye movements, OCR, fixation hold / gaze-evoked nystagmus via NI null + flocculus, T-VOR (basic), accommodation (steps + AC/A + CA/C), binocular plant (L/R eyes), basic vergence steps.

- **Architecture changes since last status (2026-05-08 → 2026-05-25)**:
  - **Cerebellum module** ([`cerebellum.py`](src/oculomotor/models/brain_models/cerebellum.py)) — replaces the deleted `efference_copy.py`. Anatomical split: flocculus (FL) for NI gaze-holding extension, ventral paraflocculus (VPF) for pursuit forward model. Owns the EC scene + target delay cascades (they ARE the forward-model output). New gains `K_cereb_fl`, `K_cereb_pu`.
  - **Near-response forward models (2026-06)** — same cerebellar EC pattern extended to the near triad: two new delay-matched EC cascades (`accom` vs cyclopean defocus, `verg` vs disparity) feed Smith-predictor corrections in `va.step` (`acc_drive`/`disparity_for_loop` subtract the in-flight command = current neural state − delayed EC). Gains `K_cereb_acc`, `K_cereb_verg`. Fixes the Hung-1997-Fig-1 vergence/accommodation step overshoot (32%/27% → <1%) by letting a *fast* loop run without ringing; `K_phasic_verg` raised 3→12 to recover Hung peak velocity. EC must be fed the **neural** state (not the full `u_acc`/`u_verg` command) or the in-flight term leaves a steady-state offset → disparity runaway. SVBN saccadic-vergence burst still to be re-tuned to the faster loop.
  - **Motor neurons** ([`final_common_pathway.py`](src/oculomotor/models/brain_models/final_common_pathway.py)) — 14-state FCP: 12 MN dynamic states with `tau_mn ~5 ms`, MLF modelled as AIN MN→CN3_MR MN axon (frequency-selective conduction cap). Separable lesions `g_nucleus`, `g_mlf_L/R`, `g_nerve`. Models INO, ophthalmoplegia, palsies cleanly.
  - **Saccade generator** ([`saccade_generator.py`](src/oculomotor/models/brain_models/saccade_generator.py)) — now 20 states with explicit EBN_L/R, IBN_L/R, OPN, smooth-trigger `z_trig`, and burst-neuron facilitation/depression (`z_fac`, `z_dep`). IBN→OPN direct inhibition (no Schmitt trigger). See `project_saccade_design.md`.
  - **Saccadic suppression** — visual gate threshold/steepness on cerebellar EC during saccade (commit `0d09a28`). Used to tune the post-saccadic settling window.
  - **Exact plant forward model** — explored inside cerebellum (commits `b8722b7`, `9c33f47`); current code rotates predicted velocity through `ec_pos = NI_net` rather than running a separate MN/plant copy, since Robinson pulse-step already cancels the plant LP to ~5 ms residual.
  - **BrainState refactor** — brain state is now a NamedTuple of subsystem `State` NamedTuples (no flat array, no `_IDX_*` slice constants). Diffrax handles PyTrees natively.
  - **Unified brain (experimental)** — [`unified_brain.py`](src/oculomotor/models/brain_models/unified_brain.py) is a matrix-form rewrite of `brain_model.step` aligned with `manuscripts/unified_oculomotor_template.md`. Active experiment; **not the canonical brain**. Swap in via `simulator.set_brain_step(unified_brain.step)`.

- **Active debugging (2026-05-25)** — explicitly flagged by user:
  - **Post-saccadic oscillation** — small residual, cerebellum forward-model + suppression-gate combo has helped but not fully closed it. See `project_ec_pre_delay_tradeoff.md`.
  - **Vergence saccades (SVBN burst)** — needs urgent debugging. `verg_copy` is currently labelled vestigial in [`vergence_accommodation.py`](src/oculomotor/models/brain_models/vergence_accommodation.py); confirm with user before reactivating.
  - **General vergence tuning** — gains, TCs, AC/A and CA/C cross-coupling not yet validated against clinical data. Bench scripts: `bench_vergence.py`, `bench_clinical_vergence.py`.

- **Next focus**: cerebellum / forward-model tuning, including the open post-saccadic oscillation and the vergence-saccades / general-vergence work above.

## HTML docs and benchmarks — regen policy

Three generated HTML pages live under `web/`. Keep them in sync with code — but
**do not run the full bench suite casually** (slow). See `project_docs_and_benchmarks.md`.

| Page | Generator | Regen trigger |
|---|---|---|
| `web/parameters.html` | `python -m oculomotor.reports.gen_parameters` | Any field added/removed/renamed/defaulted in `BrainParams`, `SensoryParams`, or `PlantParams`. Source of truth: the Python NamedTuples; optional enrichment from `oculomotor/schema/parameters_schema.yaml` (missing entries → TODO markers in the rendered page). |
| `web/states.html` | `python -m oculomotor.reports.gen_states` | Any `State` NamedTuple gains/loses a field, any subsystem's `N_STATES` changes, or a new subsystem joins `BrainState`. |
| `web/index.html` (bench gallery) | `python -m oculomotor.reports.run_benchmarks` (figures + HTML) or `--html-only` (HTML only, reuses existing figures) | Run the **individual** `python -m oculomotor.benchmarks.bench_<area>` for the area you changed, then `--html-only` to rebuild the index. Run the full suite only at milestones. |

**For me (Claude):** if I touch any `*Params` field or any `State` / `N_STATES`, propose regenerating the matching HTML before declaring the task done — but ask before launching the full bench suite. Don't silently regenerate everything just to be tidy.

## Not yet implemented / pending (future work)

- **Pursuit position sensitivity** — pursuit should be weakly driven by `pos_delayed` (retinal position error) in addition to `vel_delayed`, to correct steady-state position offsets. Add `K_pursuit_pos` gain term in `pursuit.step()`.

- **Strabismus** — deferred. See `project_strabismus_plant.md`: needs 2nd-order biomechanical plant + nonlinear NI inverse before disconjugate misalignment can be modelled cleanly.

- **Gravity estimator + T-VOR** — partially validated (OCR benchmarks pass); torsion drift during static tilt still being investigated. T-VOR uses vergence angle for near-target compensation — should mature alongside vergence work.

- **Listing's law** — torsional constraints not yet enforced on smooth pathways.

- **Multiple plant models** — see design note below.

- **Multiple brain models** — `unified_brain.py` is the current experimental alternative to `brain_model.py`. Plug in via `set_brain_step()`.

### Design note: swappable plants and brain models

The simulator is designed so that `plant_models/` and `brain_models/` can be swapped without touching sensory machinery. The integration point is the **motor command interface** between brain and plant.

**Motor command format (current):**

The brain outputs `motor_cmd: (3,)` — the Robinson pulse-step sum in rotation-vector space (yaw/pitch/roll, deg/s equivalent). This is the NI output `x_ni + tau_p * u_vel`, which combines the tonic position hold and phasic burst feedthrough into a single vector. The plant does not need to know the decomposition.

Units: motoneuron firing rate equivalent. The plant converts to forces/torques internally.

**Plant interface contract:**

```python
def step(x_p, motor_cmd, plant_params) -> (dx_p, w_p, w_eye):
    # x_p:      (3,)  plant state (eye rotation vector, deg)
    # motor_cmd:(3,)  pulse-step motor command from NI
    # w_eye:    (3,)  instantaneous eye velocity → retina / feedback (algebraic)
```

Any plant implementing this contract (first-order, second-order, MJX-backed) is a drop-in replacement in `simulator.py`.

**Brain model interface contract:**

```python
def step(brain_state, sensory_out: SensoryOutput, brain_params, noise_acc=0.0) -> \
        (dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc):
    # brain_state : BrainState NamedTuple (PyTree of subsystem States)
    # sensory_out : SensoryOutput (canal, otolith, retina_L: RetinaOut, retina_R: RetinaOut, …)
    # nerves      : (12,) per-muscle nerve activations → plant
    # ec_vel/pos  : version efference (head frame, deg/s and deg)
    # ec_verg     : vergence efference (deg)
    # u_acc       : accommodation neural command (D)
```

Different brain architectures (Raphan-Cohen, Kalman, RL policy) swap in here via
`simulator.set_brain_step(fn)`. `unified_brain.step` is the current alternative
implementation matching this signature. The sensory model and plant remain unchanged.

## SSM module convention

Every subsystem is a **state-space model (SSM)** with a uniform interface. This is the contract — new modules must follow it.

### Equations

```
dx/dt = A(θ) @ x  +  B(θ) @ u      # state derivative
y     = C     @ x  +  D(θ) @ u      # output (feedthrough allowed)
```

### Module structure

Each module exposes:

| Symbol | Type | When θ-dependent |
|--------|------|-----------------|
| `N_STATES`, `N_INPUTS`, `N_OUTPUTS` | `int` constants | never |
| `step(x, u, theta)` → `(dx, y)` | pure function | — |
| `Activations` NamedTuple + `read_activations(x_self)` | firing-rate registry | — |
| `Decoded` NamedTuple + `decode_states(acts)` | push-pull L−R nets (only subsystems with bilateral pops) | — |
| `Weights` NamedTuple + `read_weights(x_self)` | tonic / null / setpoint registers (only when applicable) | — |
| Module-level constants (e.g. `C_slip`, `PINV_SENS`) | only when used externally | — |

### Activation / Decoded / Weights registries

Every brain subsystem exposes three local registries (where applicable):

| Registry | Holds | Built by |
|---|---|---|
| `Activations` | population firing rates (rectified pops, signed firing rates, OPN gate, …) | `read_activations(x_self)` |
| `Decoded` | push-pull L−R nets (`vs_net`, `ni_net`, `pu_net`) — what downstream pops physically read | `decode_states(acts)` |
| `Weights` | tonic / null / setpoint registers (`vs_null`, `ni_null`, `e_held`) — long-term: learned weights | `read_weights(x_self)` |

`brain_model.py` aggregates them under per-subsystem fields:

```python
acts     = brain_model.read_activations(x_brain)   # acts.ni.R, acts.sg.gate_opn, ...
decoded  = brain_model.decode_states(acts)         # decoded.ni.net, decoded.pu.net
weights  = brain_model.read_weights(x_brain)       # weights.sg.e_held, weights.ni.null
```

**Cross-subsystem reads inside `brain_model.step` MUST go through these registries** — not via raw `x_brain[_IDX_*]` slicing. A subsystem reads its own state slice as before for derivative computation; cross-subsystem signals come from `acts` / `decoded` / `weights`.

### `step()` contract

```python
def step(x, u, theta):
    A = ...   # build from theta inside step
    B = ...
    dx = A @ x + B @ u
    y  = C @ x + D @ u   # C, D omitted if identity or zero
    return dx, y
```

- **A, B, C, D are local variables inside `step()`** — not separate module-level functions.
- Identity matrices (B=I, C=I, D=I) are omitted — just use `x` directly and note `# B = I`.
- **Pure function** — no side effects, no global state. Compatible with `jax.jit` and `jax.grad`.
- Returns `(dx, y)` always — ODE integrator uses `dx`; simulator uses `y` to wire modules.
- Input/output shapes and units must be documented in the module docstring.
- **`theta` is a `Params` NamedTuple** — access via `theta.sensory.tau_c`, `theta.brain.tau_vs`, etc. Never treat it as a dict.
- Module-level constants are kept only when used by external code (e.g. `retina.C_slip`, `retina.C_target_in_vf`, `canal.PINV_SENS`).

### Nonlinear extensions

Some modules have nonlinearities that wrap the linear ABCD core:

- **Canal** (`canal.py`): `nonlinearity(x_c, gains)` applies smooth push-pull rectification to the `x2` (inertia state) to get afferent firing rates. The linear `A @ x + B @ u` drives the state derivative; only the output is nonlinear. Re-exported as `canal_nonlinearity` from `sensory_model.py`.
- **Saccade generator**: gates (`gate_err`, `gate_res`, `gate_dir`) and adaptive reset TC layered on top of linear SSM core. Target selection (orbital clip + centering saccade) is handled internally using `x_ni` as a proxy for eye position and `target_in_vf` to detect out-of-field targets.
- **Visual delay** (`retina.py` + `perception_cyclopean.py`): two-stage. (a) Per-eye sharp gamma cascade in `retina.step` (N=6 stages × τ_retina), with `velocity_saturation` and visibility gating done before cascade input. (b) Post-fusion brain LP smoothing in `perception_cyclopean.step` (channel-specific TCs: motion, target_vel, disparity, defocus, plus N-stage gamma for target_pos / visibility). The brain's `C_slip` / `C_pos` / `C_vel` / `C_target_disp` / `C_target_visible` / etc. readout matrices live in `perception_cyclopean` and read into `brain[:, _IDX_CYC_BRAIN]`.

### Connector modules

`sensory_model.py` and `brain_model.py` are **connector modules** — they import their sub-SSMs, own the combined state layout and index constants, and expose a single `step()` + output-read interface. They do not implement physics themselves.

### Example: Neural Integrator (simplest case)

```python
N_STATES = N_INPUTS = N_OUTPUTS = 3

def step(x_ni, u_vel, theta):
    A = (-1/theta.brain.tau_i) * jnp.eye(3)
    D = theta.brain.tau_p * jnp.eye(3)
    # B = C = I (identity — omitted)
    dx  = A @ x_ni + u_vel
    u_p = x_ni + D @ u_vel
    return dx, u_p
```

### Wiring in the simulator

`ODE_ocular_motor` in `sim/simulator.py` calls each module's `step()` in signal-flow order, passing outputs of one as inputs to the next. The global state is a `SimState` NamedTuple — each field is sliced by pre-computed index constants (`_IDX_C`, `_IDX_OTO`, `_IDX_VIS`, `_IDX_VS`, etc.).

Evaluation order within one ODE step:
1. `sensory_model.read_outputs()` — exposes canal, otolith, and per-eye `RetinaOut`
   from sensory state (sharp-cascade-delayed signals).
2. Apply sensory noise (canal + per-eye slip / target_vel / target_pos OU drift)
   to `sensory_out.retina_L` and `sensory_out.retina_R`.
3. `brain_model.step()`:
   - `perception_cyclopean.step` — fuse per-eye delayed signals + brain LP
     smoothing → `CyclopeanOut`.
   - `perception_target.step` — target EC sub + magnitude/directional gates +
     working memory (consumes `cyc.target_*`).
   - `perception_self_motion.step` — VS + GE + HE (consumes `cyc.scene_*`).
   - Pursuit, SG, T-VOR, NI, vergence/accommodation, final common pathway.
   - Post-delay EC cascades (`x_ec_scene`, `x_ec_target`) advanced at end.
4. `plant_model.step()` — motor_cmd → dx_plant; w_eye = dx_plant.
5. `sensory_model.step()` — canal + otolith + per-eye `retina.step` driven by
   the freshly-updated eye state (must follow plant).

## LLM simulation pipeline

The `oculomotor/llm_pipeline/` package converts a plain-English scenario description into a simulation
and figure using the Claude API. Flow: `cli` → `interpret` (sends `prompt` to Claude) → `scenario` →
`patient_builder` → `run`. The Claude prompt lives in `prompt.py`. The CLI entry is `cli.main()`
(console script `oculomotor-simulate`, or `python -m oculomotor.llm_pipeline.cli`).

### Usage

```bash
# Requires ANTHROPIC_API_KEY (in the environment or .env)
python -X utf8 -m oculomotor.llm_pipeline.cli "healthy subject making a 20 deg saccade to the right"
python -X utf8 -m oculomotor.llm_pipeline.cli "patient with left vestibular neuritis doing a head impulse test"
python -X utf8 -m oculomotor.llm_pipeline.cli --dry-run "OKN: 30 deg/s full-field scene motion for 20 s then OKAN"
python -X utf8 -m oculomotor.llm_pipeline.cli --json path/to/scenario.json   # skip LLM, load JSON directly
python -X utf8 -m oculomotor.llm_pipeline.cli --show "..."                   # display figure interactively
# (or: oculomotor-simulate "...")
```

### Web server

```bash
.\server.ps1 dev                          # http://localhost:8001 (live)  — recommended
.\server.ps1 stable                       # http://localhost:8000 (frozen worktree venv)
# or directly:
python -X utf8 -m oculomotor.server --port 8000 [--host 0.0.0.0]
```

Features:
- LLM-driven single simulation or comparison
- **CSV log**: `outputs/simulation_log.csv` — timestamp, run_id, version, prompt, mode, title, figure_file, looks_correct, feedback
- **Feedback UI**: checkbox ("looks correct") + comment box + disclaimer; `POST /feedback`
- **Data download**: `GET /download/{run_id}` — CSV of t, eye pos/vel (yaw/pitch/roll), head/scene/target velocities
- Figures saved to `outputs/server_figures/<run_id>.png`

### Architecture

```
User description (str)
    ↓  Claude API (tool_use, forced schema)
SimulationScenario  (oculomotor/llm_pipeline/scenario.py — Pydantic)
    ├── HeadMotion    → oculomotor/sim/stimuli.py → head_vel_array (T, 3)
    ├── Target        → oculomotor/sim/stimuli.py → p_target_array, v_target_array
    ├── Visual        → oculomotor/sim/stimuli.py → v_scene_array, scene/target_present arrays
    └── Patient       → with_brain() / with_sensory() → Params NamedTuple
    ↓  oculomotor/llm_pipeline/run.py
simulate(params, t, **stim_kw, return_states=True)
    ↓
matplotlib Figure  →  outputs/<slug>.png
```

### Adding new stimulus types

Add a generator to `oculomotor/sim/stimuli.py` following the existing pattern (returns
`t_array`, plus the relevant arrays). Then add the new `type` literal to the
appropriate sub-schema in `oculomotor/llm_pipeline/scenario.py` and handle it in
`oculomotor/llm_pipeline/run.py:_build_stimulus()`.

### Adding new plot panels

Add a new `Literal` value to `PlotConfig.panels` in `oculomotor/llm_pipeline/scenario.py` and a
corresponding `elif panel_name == '...'` branch in `oculomotor/llm_pipeline/run.py:_draw_panel()`.

### API key

Set `ANTHROPIC_API_KEY` in your shell or `.env` before running `simulate.py`.
The model defaults to `claude-opus-4-6`; use `--model claude-sonnet-4-6` for faster/cheaper calls.

## Tech stack

- **JAX** — core framework, autodiff, `jit`, `vmap`
- **Diffrax** — ODE integration within JAX
- **Optax** — gradient-based optimization
- **Matplotlib** — diagnostics and plotting
- **Pydantic** — SimulationScenario schema and validation
- **FastAPI + uvicorn** — web server
- **Anthropic Python SDK** — LLM scenario generation

## Fitting approach (future work)

**Current priority: get the fixed-parameter model to simulate correct behavior across all paradigms. Fitting comes later.**

The long-term goal is to fit this model to patient eye movement data — recovering parameters like `tau_vs`, `K_vs`, `canal_gains`, `tau_i` that characterize specific vestibular or cerebellar pathologies.

Planned approach when ready:
- Validate parameter recovery on synthetic data first (simulate with known θ, fit from perturbed init, check recovery)
- **Loss**: MSE between model-predicted and observed eye position/velocity, summed over stimulus conditions
- **Optimizer**: `optax.adam`, typical lr ~1e-3
- **Reparameterization**: `softplus` for positive TCs, `sigmoid` for bounded gains — ensures constraints without clamping
- **Gradients**: flow through `diffrax.diffeqsolve` via reverse-mode autodiff (already differentiable by construction)
- **Diagnostic plots**: loss curve, parameter trajectories vs. step, Bode plot (gain + phase vs. frequency), time-domain overlay (predicted vs. observed), residuals

## Conventions

- All angles in **degrees**, angular velocity in **deg/s**
- Eye position = `x_p` (plant state, 3D rotation vector)
- **World frame is LEFT-HANDED**: x=right, y=up, z=forward (x × y = −z)
- **Angular vectors** `[yaw, pitch, roll]` are NOT in xyz order — use `ypr_to_xyz` / `xyz_to_ypr` (from `retina.py` for JAX, `kinematics.py` for numpy) before/after rotation-matrix ops:
  - `ypr_to_xyz([yaw, pitch, roll]) = [−pitch, yaw, roll]`
  - yaw (idx 0): rotation about +y — left-hand: forward→right (rightward turn)
  - pitch (idx 1): rotation about −x — left-hand: forward→up (look up)
  - roll (idx 2): rotation about +z — left-hand: right→up
- Head velocity input can be 1D (horizontal only) or 3D
- Gravity / specific force axis convention: **x = up** (matches `canal.py` and `gravity_estimator.py`); at rest upright, `g_head = [9.81, 0, 0]` m/s²
- `scene_present`: scalar in [0,1] — is the visual scene physically on? (external input)
- `target_present`: scalar in [0,1] — is there a foveal target? Gates pursuit integrator; set to 0 for pure OKN
- `target_in_vf`: scalar in [0,1] — delayed visual-field gate (from retinal geometry); distinguishes fixation from out-of-field in the SG
- `pos_delayed`: (3,) delayed gated position error `target_in_vf · e_pos` — zero means fixating OR target out of field; use `target_in_vf` to disambiguate
