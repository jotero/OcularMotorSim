# MAP — project atlas

> **Start here when you can't remember where something lives.**
> A high-level router over the project. For per-module detail (state layout, SSM contract,
> signal flow) go straight to [CLAUDE.md](CLAUDE.md) — this file deliberately stays above that.
> Last refreshed: 2026-06-12

---

## 1. The landscape

| Area | Where | What it is |
|---|---|---|
| **The model** | [src/oculomotor/](src/oculomotor/) | JAX simulation: sensory → brain → plant. Per-module detail in [CLAUDE.md](CLAUDE.md). |
| **Architecture & conventions** | [CLAUDE.md](CLAUDE.md) | The deep reference: state layout, SSM contract, signal flow, units. |
| **Benches** | [scripts/](scripts/) `bench_*.py` | One script per behavior (VOR, saccades, pursuit, vergence, gravity, …). |
| **Theory / the "why"** | [manuscript/](manuscript/) + a few [docs/](docs/) notes | The scientific arguments behind the design (see §3). |
| **Benchmarks & generated docs** | [docs/](docs/) | Spec ([BENCHMARKS.md](docs/BENCHMARKS.md)), gallery + parameters/states HTML. |
| **LLM pipeline & server** | [llm_pipeline/](src/oculomotor/llm_pipeline/), [server.py](scripts/server.py) | Plain-English → simulation; web app + request DB. |
| **Active plans & logs** | root `*.md` | [REORG_PLAN.md](REORG_PLAN.md) (target repo layout + packaging model), [EXPERIMENTS.md](EXPERIMENTS.md), [OVAR_DIAGNOSIS_NOTES.md](OVAR_DIAGNOSIS_NOTES.md). |
| **Claude's working memory** | mirrored in §4 | Bug history + design rationale I carry across sessions. |

---

## 2. Status — what's done vs in-progress

> The high-level picture. Supersedes the dated "Current status" block in CLAUDE.md.

**✅ Working & validated** — VOR / VVOR / OKN / OKAN, saccades, smooth pursuit, fixational eye
movements, otolith adaptation, sensory noise, gaze holding, accommodation, binocular plant, basic vergence.

**🔧 Current focus** —
- Vergence: vergence saccades (**urgent**) + general vergence tuning → [vergence_focus](#4-claude-memory-mirror)
- Post-saccadic oscillation (cerebellum forward model) → [ec_pre_delay_tradeoff](#4-claude-memory-mirror)
- OVAR backwards modulation → [OVAR_DIAGNOSIS_NOTES.md](OVAR_DIAGNOSIS_NOTES.md)

**✅ Recently landed** — orbital limits (velocity-wall clamp in plant) + target selection / centering
(clip + `e_center` folded into the saccade generator).

**📋 Future** — strabismus (needs 2nd-order plant), Listing's-law enforcement, T-VOR maturation,
pursuit position sensitivity, swappable plant/brain models.

**📓 Running log:** [EXPERIMENTS.md](EXPERIMENTS.md) — hypothesis → result → status, most-recent-first.

---

## 3. Theory / manuscript index

The scientific spine — *why* the code is shaped the way it is.

| Note | Thesis |
|---|---|
| [manuscript.md](manuscript/manuscript.md) | **Main paper** — differentiable oculomotor model + LLM clinical interface |
| [unified_oculomotor_template.md](manuscript/unified_oculomotor_template.md) | Only **two** architectures: a saccadic decision + one continuous-control template for everything else |
| [push_pull_bayesian_readout.md](manuscript/push_pull_bayesian_readout.md) | Push-pull rectification = Bayesian MAP / soft-thresholding |
| [saccade_triggers_as_kalman_gains.md](manuscript/saccade_triggers_as_kalman_gains.md) | Saccade trigger machinery = SPRT; trigger rates are Kalman gains |
| [binocular_integrator_manuscript.md](manuscript/binocular_integrator_manuscript.md) | NI manifold imposes a Hering prior bounding strabismus compensation |
| [steady_state_vergence.md](manuscript/steady_state_vergence.md) + [aca_cac_routing.md](manuscript/aca_cac_routing.md) | Near-response equilibrium + where AC/A · CA/C cross-links inject |
| [docs/cerebellum.md](docs/cerebellum.md) | Cerebellum = one prediction-error rule; setpoint is its constant-prediction limit |
| [docs/plant_compensation.md](docs/plant_compensation.md) | Pulse-step cancels the plant LP (analytical) |
| [sfn_abstract.md](manuscript/sfn_abstract.md) | SFN abstract draft |

---

## 4. Claude memory mirror

Notes I keep in private memory (`~/.claude/.../memory/`) — normally invisible to you, mirrored so you can browse.

**Conventions:** PowerShell+venv not Bash · ramp from 0 (warmup prepends arr[0]) · new content in `bench_*.py` not notebooks · `ec_vel` stays `u_burst+u_pursuit+omega_tvor`.

**Rationale / history:** saccade design (20-state SG) · cerebellum = EC+FL+VPF (efference_copy.py deleted) · post-saccadic oscillation open · retinal-slip frame fix · apply_prism deg/rad bug (fixed) · 14-state FCP (done) · strabismus deferred · vergence focus · solver/loss choices · docs regen policy · request DB + deploy topology.

---

## 5. Where each kind of note belongs (so things stop scattering)

| Kind | Home |
|---|---|
| Architecture / conventions | `CLAUDE.md` |
| Navigation / status | `MAP.md` (this file) |
| Scientific argument | `manuscript/*.md` |
| Active design plan | root `PLAN_*.md` / `*_NOTES.md` |
| Experiment results | `EXPERIMENTS.md` |
| Bug history / Claude's memory | `~/.claude/.../memory/` (mirrored §4) |
