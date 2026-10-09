# BACKLOG — pending issues and ideas

> **Proposed single home** for open issues, parked ideas and status. Other docs should link here
> instead of keeping their own pending lists (today there are seven, and they disagree — Appendix B).
> Inventory compiled 2026-10-07 at `f67ce10` from all 26 tracked Markdown files, checked against the
> code. Scoreboard at `f67ce10`: **74 pass / 15 fail / 7 drift** (96 metrics, all sections fresh).
>
> Types: `bug` · `tuning` · `feature` · `design` · `refactor` · `doc` · `ops` · `research`.
> Status is *open* unless stated. "Sources" = where the item is written today. VISION_DRAFT parking-lot
> numbers are kept as `#n`.

---

## 1. Failing benchmarks — model problems

- **SAC-1 · Post-saccadic oscillation — regressed** `bug`
  Short/long ring 5.77 / 6.99 °/s (bands ≤ 1 / ≤ 0.8). It was 1.11 / 1.42 when
  `web/post_saccadic_motion.md` declared "Side 1 solved" (5ca489b); it broke at the FCP pull-only fold
  (3a161a2 → 3.26) and the 2nd-order plant (e07c6af → 22.95), partly recovered (d4418a0 → 6.88), and the
  doc was never updated. Leads to reopen: the velocity ECs have no plant model (post_saccadic Lead B /
  Decomp #5); the FCP fold breaks linear inversion (plant_compensation l.134–152).
  Sources: CLAUDE § Active debugging; MAP §2; web/post_saccadic_motion.md; web/plant_compensation.md.
- **SAC-2 · Post-saccadic vergence transient / saccade–vergence coordination (SVBN)** `bug`
  `sac_postsac_verg_peak_noiseless` 6.03 (≤ 1.5). SVBN still to be re-tuned to the faster near-response
  loop. `verg_copy` is commented "vestigial" (vergence_accommodation.py:88,103) but is active (l.162–195).
  Sources: CLAUDE (2026-06 bullet, Active debugging); MAP §2; post_saccadic Decomp #4.
- **SAC-3 · Main sequence, small saccades** `tuning`
  `sac_mainseq_resid_max` 0.353 (≤ 0.25). Cause (pulse_slide_step.md): EBN onset charges from the −8° OPN
  clamp toward a falling e_held. Options: `alpha_fac` 0.70 (post_saccadic l.311); question the
  700·(1−e^−A/7) reference curve for small amplitudes. Sources: EXPERIMENTS 2026-04-02;
  post_saccadic l.304–312; pulse_slide_step.md § Known-open.
- **SAC-4 · Triggering with the new SC** `design` — *deferred by the user: "what does triggering need
  with the new SC"*
  Default `fix_microsaccade_rate` 0.40/s **FAIL** (≥ 0.5; 0.60 with the SC driving);
  `sac_refractory_isi_ms` 335 ms DRIFT (band ≤ 350, golden 259); latency ~230 ms (first) / ~300 ms.
  Cause: the SG's target path now gets the predicted position, so there is no stale post-saccadic error
  pre-charging the accumulator. Options: retune (acc_burst_floor −0.5 → 0, threshold_sac_qp structural —
  saccade_triggers_as_kalman_gains §6), or build-up/triggering as SC dynamics.
  Sources: PLAN_SC step 5; manuscripts/saccade_triggers_as_kalman_gains.md §6.
- **VES-1 · VVOR gain 0.839 (≥ 0.85) and post-rotatory TC 35.9 s (10–30)** `tuning`
  OVAR notes blame the tau_vs_adapt = 600 s tail: shorten it (check PAN first) or fit only the fast decay.
  Sources: web/BENCHMARKS.md; OVAR_DIAGNOSIS_NOTES P3; CLAUDE "working well" (contradicted).
- **VES-2 · VOR Bode gain with a target 1.258 (≤ 1.2)** `tuning`
- **VES-3 · OVAR modulation backwards — K_gd tension** `research`
  mod_10 3.30, mod_30 3.50 FAIL; K_gd (2.86) gives the bias but flattens the modulation. Options: accept,
  phasic otolith afferents, compromise K_gd 0.5–1.0. Sources: MAP §2; OVAR_DIAGNOSIS_NOTES P1.
- **VES-4 · Torsion shortfall / OCR tilt drift** `bug`
  `gravity_ocr_tilt_spv` 0.94 (≤ 0.5); `listing_gain` 1.1 is scaffolding; cause unknown (null adaptation
  ruled out). Sources: VISION #17; CLAUDE gravity bullet.
- **VES-5 · T-VOR in the dark 0.052 (0.3–1.4)** `design`
  Options: phasic otolith afferents, longer translational integrator (tau_head 2 s), or document the gap.
  Sources: OVAR_DIAGNOSIS_NOTES P4; CLAUDE.
- **NR-1 · Phoria and dark-focus adaptation** `tuning`
  `verg_phoria_adapt` 8.31 (0.3–3), `verg_phoria_tc_s` 200 s (5–40), `acc_darkfocus_adapt` 1.99 D (≤ 0.8).
  Related: NR-2, CB-4. Sources: CLAUDE vergence tuning; manuscripts/aca_cac_routing.md §7.9.

## 2. Superior colliculus and target selection

Design record: PLAN_SC.md. Long-term thread: VISION #18 (target belief → SC → task).

- **SC-1 · Moving hill during saccades** `design` — decide: leave it; suppress the SC input while the
  eye-velocity EC is fast (old site fades in place, rostral bump grows at saccade end); or re-enable the
  cerebellar suppression gate (disabled by default since 5ca489b, `saccadic_suppression_steepness = 0`;
  VISION #4 "measure its effect" is answered by that). Sources: PLAN_SC Decisions; VISION #4.
- **SC-2 · Look-home staircase** `bug` — the memory fade (perception_target) is blended with the SG's
  quick-phase centring target `e_center`, so the SG aims between them. Stopgap: hard switch
  seen → remembered → home in perception_target. Real fix: source competition in the SC (SC-5).
  *Not yet written anywhere.*
- **SC-3 · Does the SC hold the latched saccade vector?** `design` — replace `e_held` with motor-site
  activity decaying with remaining motor error; quick phases would still need a brainstem loop.
  Source: PLAN_SC step 6.
- **SC-4 · Retinal glide** `design` — the 24-stage retinal position cascade halved it (spread 53 → 31 ms);
  a true place-code jump needs per-location delays / a place-coded retina. Does the glide trigger small
  saccades? (a premature 1° saccade was seen once). Source: PLAN_SC OQ4.
- **SC-5 · Competition / WTA** `design` — kernel shape and gains (fast switch without killing the global
  effect); attractor regime rejected (late, biased); try feedforward input-driven inhibition. Needed for
  SC-2, target selection, double step, antisaccades. Source: PLAN_SC OQ2, OQ5.
- **SC-6 · Shadow bench and metrics** `feature` — steps, double steps, flashes, catch-up saccades,
  microsaccades. No SC metric and no flash/memory-guided metric exists. Source: PLAN_SC step 2.
- **SC-7 · Tune σ, τ, kernel; move module constants to BrainParams** `tuning` — when the SC becomes
  default. Source: PLAN_SC step 3.
- **SC-8 · Default switch-over** `feature` — `sc_drives_sg` exists (default 0); torsion still comes from
  the cyclopean path. Blocked by SAC-4. Source: PLAN_SC step 5.
- **SC-9 · Memory remap includes pursuit** `design` — `acts.cb.eye_vel` is saccade + pursuit + T-VOR, so a
  target blanked during pursuit drifts backwards in memory. *Not yet written anywhere.*
- **SC-10 · Focal SC lesion experiment** `feature` — inactivate part of the map; endpoints should shift
  away from the lesion (Lee, Rohrer & Sparks 1988). The first test only a map can do. *Not yet written.*
- **SC-11 · Later SC roles** `idea` — memory as persistent activity, moved by eye velocity (Droulez &
  Berthoz 1991 advection; parked until memory moves into the SC); express saccades via a direct
  retina → SC path; `strength` as trigger input; gap effect / rostral pole; target slots + task inputs
  (#18: memory-guided, go/no-go, antisaccade, gap/overlap); map optimisations (anisotropy, rostral
  redundancy); trajectory memory cost (save readout only). Sources: PLAN_SC Why / step 6 / Cost; VISION #18.

## 3. Visual pathway and optics

- **VIS-1 · MT / MST / NOT / V1 reorganisation** `design` — named areas with push-pull pairs, bit-identical
  first; NOT → inferior olive teacher; per-eye crossed NOT for nasotemporal asymmetry. Source: VISION #2, §6.6.
- **VIS-2 · Eye → head frames** `design` — option A (convert slip and linear flow with ec_pos) vs B (defer,
  fix docstrings, add the Tian slope as a metric). Source: VISION #5, §6.4.
- **VIS-3 · Contrast / luminance inputs** `feature` — the position half of the flash work is done
  (f67ce10); target_present/scene_present as contrast is still open. Source: VISION #9.
- **VIS-4 · Target angular velocity computed about the world origin** `bug` — retina.py:340. Source: VISION #12.
- **VIS-5 · Prism does not reach `scene_linear_vel`; `_apply_prism` docstring wrong** `bug` —
  retina.py:351, simulator.py:123–126. Source: VISION #13.
- **VIS-6 · Base-out prism sign in the LLM schema and prompt** `bug` — prompt.py:123, scenario.py:188. Source: VISION #10.
- **VIS-7 · Lens sign convention** `design` — `_apply_lens` returns acc − lens; bench_accommodation calls
  +2 D a "plus lens"; self-contradicting comment in bench_clinical.py:380–384. Source: VISION #11.
- **VIS-8 · Fusion decision reads lesion knobs** `design` — replace the motor-integrity gate with an
  innervation cost vs fusion value. Source: VISION #16.
- **VIS-9 · Duplicated velocity ceilings** `refactor` — Sensory `v_max_scene_vel` / `v_max_target_vel` and
  Brain `v_max_okr` / `v_max_pursuit` (80 / 40) kept in sync by hand. Source: post_saccadic l.77.

## 4. Cerebellum, forward models and learning

- **CB-1 · One module, one rule per region** `refactor` — prediction generator + gain + destination per
  region (web/cerebellum.md §5.1); `cerebellum.step` is monolithic. Its docstring omits the accom / verg /
  position ECs and still says "no internal eye-position forward model".
- **CB-2 · Saccadic cerebellar region (vermis / caudal fastigial)** `feature` — pulse-step matching,
  dysmetria lesions. Source: cerebellum.md §4.2, §5.2.
- **CB-3 · Plant forward model for the velocity ECs** `design` — exact or nonlinear; reopens with SAC-1.
  Sources: CLAUDE "exact plant forward model"; post_saccadic Lead B; memory note nonlinear-forward-model.
- **CB-4 · Tonic levels as constant cerebellar predictions** `idea` — tonic_verg, tonic_acc, b_vs,
  listing_primary. Source: cerebellum.md §4.5. Related: NR-1.
- **CB-5 · Learning** `research` — adaptive filter / two-site / statistical learning (VISION #8,
  §6.1–6.2); climbing-fibre plasticity (cerebellum.md §5.4); one rule on the `1 + g·activation` terms
  (saccade_triggers §5). Note: the f67ce10 position EC is another hand-built delay copy, against the
  draft principle "prediction is learned".
- **CB-6 · Kalman/EKF reafference estimator** `research` — suppression returning inside a probabilistic EC.
  Source: post_saccadic l.254–257, 385–387.

## 5. Saccade generator internals

- **SG-1 · `e_held` (and z_fac, z_dep) filed under `sg.Weights`** `refactor` — fast states, not weights.
  Part of ARCH-1. *Not yet written anywhere.*
- **SG-2 · Rotation vectors added as if they commuted** `design` — `x_ni + target` in target selection
  (saccade_generator.py:316–317); fine at small angles. (The "Fick vs rotation vector" item raised on
  2026-10-07 was a docstring error — the retina outputs rotation-vector components; docstrings fixed.)
- **SG-3 · Uncertainty-adaptive SPRT; derive SG time constants from noise and a loss** `research` —
  saccade_triggers §5, §7.

## 6. Vergence and accommodation

- **NR-2 · AC/A and CA/C routing** `design` — drop the AC/A output bypass (R4, `u_verg += aca_vec`,
  vergence_accommodation.py:250); CA/C now also bypasses (l.273, 276); check cross-link loop stability;
  fast cross-link integrator idea. Source: manuscripts/aca_cac_routing.md §6–8.
- **NR-3 · AC/A delivers 0.857 pd/D at AC_A = 5** `tuning` — band [0, ∞) can never fail; tighten it.
  Source: aca_cac_routing §3, §8.
- **NR-4 · `steady_state_vergence.md` needs re-deriving** `doc` — tonic set points are now back-solved
  (simulator.py:259–295).

## 7. Pursuit

- **PUR-1 · Pursuit position sensitivity (`K_pursuit_pos`)** `feature` — likely moot (its motivation was
  fixed by the EC change; pursuit passes). Decide or drop. Sources: CLAUDE; MAP §2; memory pursuit-gain-split.
- **PUR-2 · Missing figures** `feature` — OKN zoom (`_okn_zoom` exists but `run()` never calls it); VOR TC
  with vs without VS. Source: web/BENCHMARKS.md §2b/2c.

## 8. Plant, strabismus, binocular integrator

- **PLANT-1 · Strabismus** `feature` — the 2nd-order plant exists; what remains is the nonlinear NI inverse.
  Sources: CLAUDE; MAP §2; VISION #16.
- **PLANT-2 · 6-D binocular integrator + asymmetric plant** `research` — binocular_integrator manuscript §4.
- **PLANT-3 · `k_orbital` unused** `bug` — in PlantParams, never read by the 2nd-order plant. Source: grant_evidence.
- **PLANT-4 · Soft orbital wall in the NI anti-windup** `feature` — still a hard clip. Source: VISION #14.
- **PLANT-5 · Alternative brain models via `set_brain_step`** `feature` — none shipped. Source: CLAUDE.

## 9. Architecture and code hygiene

- **ARCH-1 · Activations only for real neurons; Outputs for computational blocks** `refactor` — includes
  SG-1 and pt "State == Activations". Source: VISION #7, §6.3.
- **ARCH-2 · EC naming scheme** `refactor` — `ec_*` current / `ec_*_d` delayed; collision `ec_verg`
  (brain_model absolute vs cerebellum delayed); new `eye_vel`, `eye_disp`, `target_shift` not covered.
  Source: VISION #6, §6.5.
- **ARCH-3 · Review and homogenise all nonlinearities** `design` — EKF-friendliness; soft-threshold dead
  zones from σ²/τ (push_pull manuscript); √ front end idea. Source: VISION #15.
- **ARCH-4 · `g_nerve` / `g_nucleus` are read-only properties** `bug` — `with_brain(g_nerve=…)` fails;
  `analysis.extract_fcp_cascade` and `fcp.read_outputs` depend on the property. Docs list them as knobs;
  the settable ones are `g_mn_*`, `g_cn*`, `g_mlf_*`.
- **ARCH-5 · Dead or orphaned code** `refactor` — perception_cyclopean `C_*` matrices (one `noqa` import);
  first-order plant (unused by `simulate`); `bench_accommodation` (superseded by bench_vergence);
  `fitting/` (obsolete).
- **ARCH-6 · Stale code comments** `doc` — brain_model N_STATES sum "= 163" (913); brain_model "nothing
  reads it yet" (SC); cerebellum State docstring; FCP docstrings (×2, "floor removed"); `verg_copy`
  "vestigial"; saccade_generator tau_bn; gravity axis comments (otolith.py:90, perception_self_motion.py:160:
  code is y-up); SC docstring "suppression attenuates" (gate off); bench_saccades cascade caption "no moving
  hill"; perception_target docstrings ("trust", "delayed position"); `mlf_lead` schema text; cli.py
  `scripts/simulate.py`; simulator SimState docstring (`ec_scene`/`ec_target`, plant = 6).
- **ARCH-7 · Consolidation milestone** `design` — bring today's model in line with the principles where
  cheap. Source: VISION §3.

## 10. Validation, fitting and inference (grant aims)

- **VAL-1 · Fitting is obsolete** `feature` — no likelihood, posterior, identifiability or SBI;
  `sim/synthetic.py` deleted. Sources: CLAUDE § Fitting; grant_evidence; Aims 1–2.
- **VAL-2 · `tests/` is empty; clinical report has no scalar gate** `feature` — grant_evidence.
- **VAL-3 · CSV export incomplete** `feature` — no latent states, parameters, target position,
  visibility / cover / prism metadata. Source: grant_evidence.
- **VAL-4 · LLM prompt has stale vergence values** `bug` — prompt.py:261–263 (K_phasic_verg 1.0 vs 12, …).
- **VAL-5 · Reproducible LLM evaluation protocol** `feature` — manuscript.md § LLM interface.
- **VAL-6 · Aims 1–2 infrastructure** `feature` — cross-system benchmarks (visual suppression of vestibular
  responses, saccade–vergence coordination, T-VOR distance scaling, acc–verg dissociations), sensitivity
  analyses, provenance and seeds, cohort sampler, time-since-lesion field, lesion priors. Source: grant/.

## 11. Benchmarks and references infrastructure

- **BENCH-1 · `web/BENCHMARKS.md` is an obsolete spec** `doc` — old script paths, placeholders, criteria
  superseded by metrics_ranges.json; still linked from the gallery header.
- **BENCH-2 · Four inconsistent citation registries** `doc` — benchmark_references.md, references.json,
  metrics_ranges `cite`, bench `citation=` strings. `rashbass1961` resolves to the wrong paper; journals
  disagree for Bahill, Smit, Rolfs, Cohen 1977.
- **BENCH-3 · Bands not literature-anchored** `doc` — fcp_*, gravity_ocr_tilt_spv (provisional),
  gravity_tilt_reversal_*, sac_postsac_* / cascade_count.
- **BENCH-4 · Reference PDFs** `doc` — missing band-source PDFs; unlisted Optican & Miles 1985 and Daye 2015;
  Schor 1988 to `near response/context/`.
- **BENCH-5 · `--run` takes module names, `--section` section ids** `bug` — `--run near_response` matches
  nothing; the "~60 min" message is a Windows figure (~13 min on the Mac).

## 12. Ops and environment

- **OPS-1 · `main` not pushed** `ops` — 5+ commits ahead of origin; refreshed gallery (40 files) uncommitted.
- **OPS-2 · Mac setup undocumented** `ops` — launchd agents (sim-stable, sim-tunnel), `~/om-stable`,
  `~/om-data`; tunnel agent written but not loaded; SETUP.md is Windows-centric.
- **OPS-3 · Claude memory split across two directories** `ops` — MAP's memory mirror lists notes that do not
  exist; CLAUDE.md references four missing `project_*.md` notes.
- **OPS-4 · scratch/ cleanup** `ops` — numbered duplicates, ~40 artifacts; `perf_baseline.py` (used by SETUP)
  lives there.

## 13. Research questions (from the manuscripts — not project tasks)

Binocular integrator "stability tax" and its predictions (binocular_integrator §5–6); implicit noise models
of sensory nonlinearities (push_pull §3); "far fewer degrees of freedom" via noise-derived τ, θ, K (unified
template §5, §7); SG time constants from a loss (SG-3); synthetic-data matching and stimulus design for
lesion discrimination (manuscript.md); Aim 2 identifiability classes (INO, UVH, vergence–accommodation).

---

## Appendix A — Stale content by document (audit 2026-10-07)

Severity: **high** = misleads a reader or breaks if followed · **med** = outdated status/numbers · **low** = cosmetic.

| Document | Sev. | Main stale points |
|---|---|---|
| CLAUDE.md | high | Folder tree (missing pupil/eyelid/2nd-order plant/server/config, `fitting/` and `llm_pipeline/` drawn under `models/`, deleted `sim/synthetic.py`); state structure (brain ≈170 → 913, SimState has 6 fields, plant 12 states); Params lists (`aca_ratio` → `AC_A`, `g_nucleus`/`g_nerve` not settable, missing cerebellar gains, PlantParams); brain/plant contract (returns `(dbrain, fcp.Nerves)`, not an 8-tuple; `motor_cmd`); "Current status (2026-05-25)" block; four dead `project_*.md` references; `outputs/*.png` demos no bench produces; gravity axis (code is y-up); `_IDX_*` / `C_*` / `retina.C_slip` references; SG gates `gate_res`/`gate_dir`; venv path Windows-only; gallery is `web/benchmarks/index.html`; "~60 min"; `_vs_net3` example; analysis helper table incomplete; server paths (`server_data/`); `PlotConfig.panels` → `PanelName`. |
| MAP.md | high | Links to non-existent `docs/` (files are in `web/`); status dashboard frozen at 2026-06-12; missing PLAN_SC, VISION_DRAFT, SETUP, grant, post_saccadic, pulse_slide_step; memory mirror lists absent notes; PowerShell-only. |
| INTEGRATION.md | high | API contract no longer matches; examples would not run (brain step return, plant module/signature, `acc_plant.step` return, `ec_*` into sensory, no `sc`, no pupil/eyelid, line anchors). |
| web/BENCHMARKS.md | high | Obsolete spec (paths, placeholders, 1st-order plant, 40-stage cascade, pass table) — still linked from the gallery. |
| web/post_saccadic_motion.md | high | Declares the oscillation solved; it regressed (SAC-1). Formulas and line refs predate pulse-slide-step and the 2nd-order plant. |
| web/plant_compensation.md | high | 1st-order-plant framing; mechanisms since replaced (pulse-slide-step) or retired (cerebellar plant copy); floor is back as a fold. |
| PLAN_SC.md | med | Status line, Input row, open question 1, step 4, the two-bump section, "92/38 states", `sac_vel` — all describe the 2026-10-06 design; the current design is only in the Remapping / Saccade-handover rows. |
| VISION_DRAFT.md | med | "SC deferred" (§6.6); #3, #4, #9 status; #18 "memory drained by any eye movement"; branch `eye-head-frames` no longer exists; the 2026-10-07 items are missing. |
| SETUP.md | med | Windows-centric; merged branch name; "~400-state ODE" (≈1286); "~60 min"; `--run` vs `--section`; launchd undocumented; memory-copy advice. |
| web/cerebellum.md | med | Theory note: does not describe the module; E_held framing superseded; self-contradiction at l.68; brainstem τ_i 25 → 5 s. |
| references/benchmark_references.md | med | Metric names that no longer exist; missing sections (gaze holding, FCP, pupil, eyelid, clinical); citation disagreements. |
| EXPERIMENTS.md | low | Historical log, stopped 2026-04-10; its [ACTIVE] entry was resolved; old state names. |
| scratch/README.md, web/assets/README.md | low | Contents and placeholder text out of date. |
| manuscripts/*, grant/* (author's drafts — not to be edited by us) | — | manuscript.md describes the pre-May-2026 architecture (states, cascades, NI, SG, plant, defaults, results); unified_oculomotor_template §8 refers to deleted `unified_brain`; aca_cac_routing / steady_state_vergence / OVAR notes numbers; grant_evidence counts and scoreboard; specific_aims "94 / 95"; sfn_abstract "quantitatively matches" and "can be fit". |

Fixed during this audit: CLAUDE.md position-noise row (my 2026-10-06 wording); the "Fick angles" docstrings
in retina.py and superior_colliculus.py (the retina outputs rotation-vector components); memory note
`vision-draft-open` ("uncommitted").

## Appendix B — Consolidation proposal

Pending items live in seven places today: CLAUDE § Current status and § Not yet implemented, MAP §2,
EXPERIMENTS, VISION_DRAFT §8, PLAN_SC open questions, memory notes. Proposal — **one job per document**:

| Document | Job | Action |
|---|---|---|
| **BACKLOG.md** (this file, minus appendices) | All pending items and status | Adopt; others link here. |
| CLAUDE.md | Architecture and conventions reference | Remove the status and pending sections (→ BACKLOG); fix the stale facts above; hold the single brain/plant contract. |
| MAP.md | Index / router only | Drop the status dashboard and memory mirror; fix links; add BACKLOG, PLAN_SC, VISION_DRAFT, SETUP. |
| VISION_DRAFT.md | Principles and vision | Move the §8 parking lot here (keep `#n` aliases). |
| PLAN_SC.md | SC design record | Keep current decisions; condense superseded designs into a short history; open questions → BACKLOG §2. |
| INTEGRATION.md | External integration guide | Rewrite against the current API, or fold the contract into CLAUDE.md plus a runnable example. |
| EXPERIMENTS.md | — | Mark historical (or move to an archive folder). |
| web/BENCHMARKS.md | — | Retire; point the gallery header at the generated page / metrics_ranges.json. |
| web/post_saccadic_motion.md, web/plant_compensation.md | Investigation history | Add a "historical — superseded" banner pointing to SAC-1; start a fresh note when SAC-1 is picked up. |
| web/cerebellum.md | Cerebellar theory | Keep; add a short "what the module implements today" section. |
| references/benchmark_references.md | Citation index | Generate from metrics_ranges.json + references.json (one registry). |
| SETUP.md | Install / run | Add the Mac (launchd) setup; fix sizes, timing, branch. |
| manuscripts/*, grant/* | Author's writing | Leave; Appendix A lists what is outdated. |
| Claude memory | — | Merge the two memory directories; drop MAP's mirror. |

Suggested order: (1) fix what misleads — CLAUDE.md facts, INTEGRATION contract, MAP links; (2) adopt
BACKLOG and strip the duplicate pending lists; (3) tidy PLAN_SC and VISION_DRAFT; (4) banners on the
historical design notes; (5) one citation registry.
