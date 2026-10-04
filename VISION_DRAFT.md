# Vision & principles — DRAFT

> **Status: working draft, not committed to.** Distilled from a Q&A session on 2026-10-04.
> Nothing here is decided yet. To resume: re-read §4 (the raw answers), then work through
> §5 (open questions). Code decisions that depend on this are listed in §7 and are on hold.

---

## 1. Vision (draft)

A **scientific theory** of the primate oculomotor system, built as a network of
**positive-rate neural populations organized after real anatomy**. Its wiring is **derived**
from optimal estimation and control (*what* to compute), implemented through a small set of
circuit **motifs** (*how*), and it keeps itself calibrated with **biologically plausible
learning driven by its own errors and statistics**. A patient is the same brain with a lesion —
either acutely, or after its own compensation. The clinical simulator is the showcase.

## 2. Principles (draft)

1. **Theory first.** The model is the theory; benchmarks are its tests; the clinical tool
   (LLM pipeline + web app) is the showcase and may lag.
2. **Physics outside, neurons inside.** The periphery — sensors, latency, plant, muscles — is
   physics: measured, not tuned. Sensory latency belongs to the sensor (with a realistic shape;
   the shape matters, e.g. for brief flashes). The brain is neural populations only.
3. **The final-common-pathway standard.** Every module aims for what the FCP has: lesion knobs
   are real lesions, units are real neurons, structure is grounded in measured physics, few
   free knobs.
4. **Derived, then learned.** Optimality (given noise, latency, plant, task costs) says what to
   compute; motifs (push-pull readout, integrators, accumulator, prediction error) say how.
   Where anatomy is unknown: derived generic populations with a provisional label, eventually
   learnable within constraints.
5. **Learning belongs to the brain** — three kinds (see §6.1):
   - *error-driven* (cerebellar cortex, climbing-fiber error: gains, prediction across delays),
   - *instructed transfer* (cortex teaches the nuclei: consolidation),
   - *statistical / homeostatic* (local, from a population's own activity or input statistics:
     setpoints, time constants — the priors).
   Learning signals are the brain's own (slip, position error, effort, statistics); rules are
   plausible. Predicting the consequences of its own movements across delays is learned, not
   built in as an exact delay copy.
6. **Tuning is scaffolding.** Hand-tuning to benchmarks is fine now, but every tuned brain value
   should be one the learning could eventually reach and maintain.
7. **Patients go through the theory.** Patient = healthy brain + lesion; fit only the lesion
   (and time since onset). Two modes: acute (static lesion) and chronic (after the model's own
   compensation). Lesions are structures where anatomy is known; flagged functional lesions
   where it isn't.
8. **Swappable sensory interface.** Low-dimensional retinal features for now (target and scene
   position/velocity, maybe contrast) into coarse cortical populations. The brain sees only
   what an image-based front end could compute, so closed-loop rendering can replace it later.
9. **Neurons are checked qualitatively** against recordings: signs, tuning, burst / tonic /
   burst-tonic types (not exact rates).
10. **Incremental migration** — benchmarks pass at every step.

## 3. Scope and sequencing

- **In scope (next few years):** eye-head gaze shifts; cognitive saccade tasks (target choice,
  anti-saccades, memory-guided); strabismus; long-term adaptation.
- **Not image-driven** for now (see principle 8).
- **Next milestone:** consolidation — bring today's model in line with the principles where cheap.

## 4. Record of answers (2026-10-04)

| # | Question | Answer |
|---|---|---|
| 1 | If only one goal survives? | Scientific theory |
| 2 | Theory primarily of: optimality / motifs / anatomy→function / learning? | Wants all four; unsure how they should guide development (read as Marr levels) |
| 3 | Where should building a subsystem start: computation / anatomy / behavior? | Don't know |
| 4 | Most satisfying part of the model? | Final common pathway |
| 5 | Why? | Lesions are real; real neurons; grounded in physics; no free knobs |
| 6 | Weakest part? | "All the delayed stuff" |
| 7 | Why? | Not neurons; too many knobs; unclear how to learn them; feels different from the state-space equations of neurons |
| 8 | Delay statements signed | Latency is physics; brain = neurons only; prediction is learned. *Not* signed: "delay shape doesn't matter" |
| 9 | Where anatomy is unknown? | Derived neurons, eventually learnable with constraints |
| 10 | Learn from? | The brain's own errors |
| 11 | Learning rule? | Biologically plausible |
| 12 | Lesion compensation? | Two modes: acute and chronic |
| 13 | Patient fitting? | Via the theory: healthy brain + lesion + its own compensation |
| 14 | What counts as a lesion? | Structures; functional lesions allowed (flagged) where anatomy is unknown |
| 15 | Hand-set primitives? | Start by tuning to benchmarks; eventually it should learn by running simulations |
| 16 | Where does the brain begin for vision? | Coarse cortex; low-dim features now; swappable for a rendering front end |
| 17 | Scope | Eye-head gaze shifts, cognitive tasks, strabismus, long-term adaptation — all in |
| 18 | Wiring derived from? | Both: optimality = what, motifs = how |
| 19 | Role of the clinical tool? | Showcase |
| 20 | Migration? | Incremental |
| 21 | Next milestone? | Consolidate |
| 22 | Neurons vs recordings? | Qualitatively |

## 5. Open questions and tensions

- **Where development starts** (Q3) — unresolved. Candidate: "computation sets the target,
  anatomy sets the constraints, behavior is the test", but not agreed.
- **"Modern learning algorithms" vs plausible rules.** Working reading: gradient methods are
  allowed as tuning scaffolding (principle 6); the brain's own learning must be plausible. Confirm.
- **Kalman/LQG framing vs positive-only populations.** Is the generic push-pull encoding
  (r± = rectify(b ± x/2)) the bridge, replacing today's hand-built bilateral populations?
- **Estimators vs controllers vs switching parts.** The saccade decision, target memory and
  fusion gates do not fit a Kalman filter; how are they treated?
- **Where each kind of learning lives** — statistical adaptation sites are debated (brainstem vs
  nodulus/uvula for time-constant habituation).
- **Too many threads at once** — need an order for the discussions below.

## 6. Parked threads (background)

### 6.1 Three kinds of learning
| Kind | Signal | Site | Timescale | Key refs |
|---|---|---|---|---|
| Error-driven | climbing-fiber error (slip, position error) | cerebellar cortex | minutes | Marr 1969; Albus 1971; Ito; Fujita 1982 |
| Instructed transfer | Purkinje output teaches the nucleus | vestibular / deep nuclei | hours–days | Miles & Lisberger 1981; Lisberger, Pavelko & Broussard 1994; Raymond & Lisberger 1998; Raymond, Lisberger & Mauk 1996; Kassardjian et al. 2005 |
| Statistical / homeostatic | own mean activity / input statistics | brainstem (largely), periphery | seconds–days | Leigh, Robinson & Zee 1981 (PAN); Nelson et al. 2003 (firing-rate potentiation); Cohen, Cohen, Raphan & Waespe 1992 (habituation) |

Possible division of labor: statistical learning learns the world (priors, setpoints);
error-driven learning calibrates the internal model to the body (gains, delay prediction).

### 6.2 Cerebellum as adaptive filter (candidate replacement for the EC delay chains)
Granule layer = generic temporal basis on efference copies; Purkinje weights learned by
climbing-fiber error (retinal slip) with an eligibility window matched to the sensory latency.
Refs: Fujita 1982; Porrill, Dean & Stone 2004 (recurrent architecture, sensory error suffices);
Dean, Porrill, Ekerot & Jörntell 2010 (review); Kennedy et al. 2014 (granule temporal basis,
unipolar brush cells); Suvrathan, Payne & Raymond 2016 (~120 ms plasticity timing rule in
flocculus matches the visual feedback delay).

### 6.3 Modeling levels
Level 1 = positive-rate neurons; level 2 = signed lumped variables with anatomical labels;
level 3 = computational blocks. Idea: abolish level 2 (decoded nets become the defined 1→3
crossing; signed states are promoted to level 1 where anatomy is known, else labeled level 3).
Related idea: `Activations` / `read_activations` only for real neural populations; computational
modules expose `Outputs` / `read_outputs` (as the sensory side already does).

### 6.4 Eye→head frames (VS and heading)
Phase-0 probe (OKN 30°/s yaw, gaze held at ±20° by fixation): the eye-frame slip reaching VS
carries 5–8°/s of roll, but VS stores ~1°/s (short torsional storage + nodulus dumping); OKAN
slow-phase axis tilt slope ≈ 0.02–0.05, vs Tian, Zee & Walker 2007 rotational OKN 0.11.
Options: (A) convert slip and linear flow to head frame with `ec_pos` (0 new states);
(B) defer, fix the docstrings, add the Tian slope as a cited metric.

### 6.5 Efference-copy naming
Option 2: `ec_*` = current copies of commands (`ec_vel`, `ec_pos`, `ec_verg`, `ec_acc`);
`ec_*_d` = delayed, sensor-matched versions from the cerebellum. Fixes the `ec_verg` name
collision (absolute current vs relative delayed) and the `accom_cmd` / `verg_cmd` misnomers.
Note also `verg_tonic` (state) vs `tonic_verg` (parameter).

### 6.6 Visual areas for the feature channels (SC deferred)
Each channel moves from one lumped vector to a named coarse area, with bilateral push-pull pairs:

| Area | Signal | Push-pull pair means | Frame | Lesion it enables |
|---|---|---|---|---|
| V1 | binocular fusion, disparity | lumped for now | eye | — |
| MT | target retinal velocity | hemisphere = opposite hemifield (needs a hemifield gate) — or direction pairs only (undecided) | eye | hemifield target-motion deficit (Newsome et al. 1985) |
| MSTl | MT + delayed EC (+ head velocity) → pursuit memory | hemisphere, ipsiversive bias | head/world velocity | ipsiversive pursuit deficit (Dürsteler & Wurtz 1988) |
| MSTd | scene linear flow → heading | hemisphere | head (frame conversion lives here) | heading deficit |
| MST→DLPN→VPF | scene angular slip, fast OKR (today's `g_vis`) | hemisphere | eye→head | early-OKN loss |
| NOT/AOS | scene angular slip → velocity storage (today's `K_vis`) | left NOT = leftward | eye | ipsiversive OKN/OKAN loss |
| MST | disparity → vergence | lumped | eye | — |

Moves: `perception_cyclopean` fusion → V1, its `tau_vis_smooth_*` become area dynamics; the
cerebellar pursuit sum (`vpf_drive`) → MSTl; `g_vis` out of the VS output equation → MST path;
pursuit pops relabeled FEFsem/VPF velocity memory. Steps: (1) bit-identical reorganization,
(2) push-pull pairs, (3) rewiring. **SC map is a separate, later design** (log-polar map, input
tuning, selection, population readout, rostral fixation zone, memory input).

**NOT vs MST is not redundancy** — it is the Raphan–Cohen direct/indirect OKN split the model
already has:
- *Direct, cortical* (MT/MST → DLPN → VPF/flocculus → VN): fast OKN rise, ocular following = `g_vis`.
- *Indirect, subcortical* (NOT/AOS → VN): slow buildup + OKAN via velocity storage = `K_vis`.
- Lesion dissociation: occipital lesions lose the fast rise but keep slow buildup + OKAN
  (Zee et al. 1987); NOT lesions reduce ipsiversive OKN/OKAN (Kato et al. 1986); NOT stimulation
  gives slow-rise nystagmus with afternystagmus (Schiff, Cohen & Raphan 1988).
- Shared input handled by weights: NOT input = w_ret·(direct retinal slip) + w_ctx·(MST), with
  w_ret + w_ctx = 1 and both reading today's slip → bit-identical. Later: make the retinal part
  per-eye, crossed, temporal→nasal preferring → monocular nasotemporal OKN asymmetry (infantile
  strabismus, in scope).
- Future link: NOT → inferior olive carries retinal slip as the climbing-fiber error for
  floccular learning (the teacher for §6.2).

## 7. Code decisions on hold pending this vision

- Frames: option A vs B (§6.4).
- EC renames (§6.5).
- Registry split (§6.3).

Branch `eye-head-frames` @ `562fe91` holds the earlier work (stimulus renames, prism velocity
shift, stereo-override removal).

## 8. Parking lot — every open thread, one line each

Pick ONE at a time. Nothing here is in progress.

| # | Thread | Kind | Status / next action |
|---|---|---|---|
| 1 | Vision & principles (this file) | design | draft; resume at §5 |
| 2 | Visual areas MT/MST/NOT/V1 (§6.6) | refactor | planned; step 1 is bit-identical |
| 3 | SC map | new model | deferred; own design session |
| 4 | Saccadic suppression | diagnose | unclear what it does now; measure its effect before changing it |
| 5 | Eye→head frames (§6.4) | decision | option A vs B |
| 6 | EC naming (§6.5) | refactor | Option 2 proposed |
| 7 | Activations only for real neurons (§6.3) | refactor | proposed |
| 8 | Cerebellar adaptive filter / two-site / statistical learning (§6.1–6.2) | research | background gathered |
| 9 | Flash position decode (divide position by visibility) + contrast/luminance inputs | model fix | diagnosed: 10 ms flash at 10° → saccade lands at 0.8° |
| 10 | Base-out prism sign in LLM schema + prompt | **live bug** | schema/prompt say BO = +yaw; simulation shows BO on R eye = −yaw |
| 11 | Lens sign convention (lens power vs demand offset) | decision | benches label +2 "plus lens" but it acts as a minus lens |
| 12 | Target angular velocity computed about the world origin, not per eye | model fix | affects near targets / translation |
| 13 | Prism does not reach `scene_linear_vel` (+ `_apply_prism` docstring wrong) | model fix | goes with #5 |
