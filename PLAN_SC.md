# PLAN — Superior colliculus map (SC v1)

> Status: **steps 1 + 4 done (2026-10-06)** — module in shadow mode, input remapped by the saccadic
> corollary discharge, two-bump saccade handover (no moving hill), thresholded readout.
> Decisions from the 2026-10-05/06 discussions.
> Parking-lot thread #3 in [VISION_DRAFT.md](VISION_DRAFT.md); prerequisite for the task /
> target-slot thread #18.

## Why

Today the target position reaching the saccade generator is a single low-passed vector
(`perception_cyclopean` → `perception_target` memory → SG). Two problems follow from that:

- **Slow switching.** Low-passing a rate-coded position makes a target step glide through every
  intermediate position. In a place code, filtering changes only the amplitude of activity:
  the bump at A fades while the bump at B rises, with no intermediate place ever active.
- **Flash bug (#9).** Position is carried as position × visibility, so a brief flash is read as a
  target at a fraction of its eccentricity. In a place code, location codes position and
  amplitude codes visibility — a flash is a small bump in the right place.

The SC is also where later work lands: target selection and competition (global effect,
antisaccade errors), the gap effect via the rostral pole, memory as persistent activity, and
the task / BG-disinhibition inputs of #18.

## Decisions

| Topic | Decision |
|---|---|
| Scope of v1 | **Target position only.** Trigger stays in the brainstem (OPN/EBN/IBN). The accumulator may later emerge from SC dynamics; not in v1. Memory stays in `perception_target` for now. |
| Rollout | **Shadow mode first:** the SC runs in parallel and its decoded position is logged and compared with today's target position; the SG keeps using the current path until we switch over. |
| Dimensionality | **2-D** from the start (1-D is not useful enough). |
| Geometry | Ottes, Van Gisbergen & Eggermont (1986) complex-log map: `z = A·(e^w − 1)`, `w = u/B_u + i·v/B_v`, A = 3°, B_u ≈ 1.4 mm, B_v ≈ 1.8 mm/rad. One map per colliculus (contralateral hemifield), extending to **~100° eccentricity** so bumps are not truncated near the edge (edge truncation, not unit count, was the main decoding error: ~11–15% at 50° if the map stops at 50°, < 2% if it extends to 100°). |
| Size | **646 units**: 19 (u, rostro-caudal) × 17 (v, medio-lateral) per colliculus, ~0.35 mm spacing, ~1.4 units per bump σ. More than the planned ~510 because the map extends 1.4 mm into the rostral overlap (Re w ≥ −1): with less, 0.1–1° errors read ~13% too small (truncated bump in the other colliculus); with it, 1–2%. Optimisations deferred. |
| Input | **Cyclopean vision** (perception_cyclopean output), rendered as a bump: **location = target position** decoded independently of visibility by perception_cyclopean (`cyc.target_pos` is a position since 2026-10-06 — open question 6), **amplitude = visibility**. Bump σ ≈ 0.5 mm. (Step 1 read the per-eye retina and re-did the binocular fusion itself — a duplicate, removed 2026-10-06.) A direct retina → SC path (express saccades) is for later. |
| Dynamics | Each unit low-passes its input (τ ≈ 10–20 ms); positive rates (rectified) feed **local excitation + broader inhibition** (soft winner-take-all), including inhibition between the two colliculi. This nonlinearity is what makes the decoded position switch fast instead of averaging old and new bumps. |
| Readout | **Thresholded population average**: only units firing above θ = 0.1 (a full bump peaks at ≈ 1) enter, weighted by r − θ, in map coordinates, then mapped back to visual space (`z = A·(e^w − 1)`); the two colliculi combined with weights **m⁴** (m = thresholded mass), which favours the colliculus holding the complete bump near the vertical meridian (static error < 1% from 1° to 40°, < 0.015° below 1°). **No unit above threshold** → `valid` → 0 and `pos` → 0 (the empty mean is guarded to the map origin, which is the fovea): downstream treats it as "no target". Plus **total activity** as a strength signal (future trigger / accumulator input). |
| Remapping | **Cerebellar position forward model (2026-10-07).** The SC (and, since 2026-10-07, the SG's target path in the default model) receives the predicted current target position = perception's delayed cyclopean position + the cerebellum's predicted shift during the delay (`acts.cb.target_shift`, summed in brain_model) = delayed position − eye displacement still in flight (commanded eye velocity — saccade + pursuit + T-VOR — after the MN lag, through a cascade delay-matched to `cyc.target_pos`; in flight = Σ τ_k·stage_k) + delay × target-velocity estimate (`pred_err` = retinal slip + EC, × visibility). The eye and target terms cancel during pursuit; only the eye term acts for a stationary target. Error vs the true current error: 0.03° after saccades, 0.16° during 20°/s pursuit (delayed signal: 0.24° / 0.44°). Earlier versions (SC-internal CD cascade; saccade-only cerebellar EC + `e_held` + OPN + stale clock) were replaced by this. |
| Saccade handover | **Open.** With the prediction as the only input, the bump sweeps with the eye during a saccade (moving hill; readout intermediate 18–22 ms per saccade, ~1.4° behind at saccade end, settled ~30 ms later). The two-bump handover (old site fades in place, landing-site bump driven by `e_held`, 2026-10-06) avoided it but needed `e_held`, the OPN and a stale clock; dropped for simplicity. The model's cerebellar saccadic-suppression gate is applied to the SC input amplitude but is DISABLED by default (`saccadic_suppression_steepness = 0`). Options: leave the sweep; suppress the input from current eye-velocity EC speed (no moving hill, rostral bump grows at saccade end); re-enable the cerebellar gate. |
| Cost | Measured: +~33% wall time per simulated second (0.07 → ~0.093 s). The concern is memory of saved trajectories (+646 SC + 38 cerebellar states every ms); mitigate by saving only the decoded readout / a subsampled map when needed. |
| Level | SC units are level-1 neurons: real `Activations` (rectified rates). |

## Open questions

1. ~~**Remapping during and after saccades.**~~ — (b) implemented, then the two-bump handover (see
   Remapping / Saccade handover above and the step-4 results below).
2. Kernel shape and gains for the winner-take-all (fast switch without killing the global effect
   for nearby targets).
3. ~~How the two colliculi combine near the vertical meridian~~ — settled: rostral overlap to
   Re(w) = −1 plus m⁴-weighted combination.
4. 🔶 **Partly addressed (2026-10-07): target position now rides a sharper retinal cascade.**
   The retina's target-position path is 24 stages (same 50 ms mean) instead of 6 — position ×
   visibility plus a matching 24-stage visibility as its denominator, decoded to a position in
   `retina.read_outputs`; the 6-stage `target_visible` stays as the AMPLITUDE (so flash dynamics
   are unchanged). Cyclopean target position 10/50/90% = 50/64/81 ms (spread 31 ms, was 40/62/93,
   spread 53); SC readout 62/76/100 ms (spread 38, was ~65). The cerebellar saccadic EC was
   lengthened to match (92 states). Cost +156 sensory states. The glide is halved, not gone — a
   true place-coded retina (per-location delays) would remove it. Original finding:
   **The glide comes from upstream (found in step 1).** Target steps are decoded with the same
   ~70 ms glide as today's path, because the retina delays the target *position value* through
   its gamma cascade (a rate code) — the bump location rides along with it. A place code only
   switches fast if the delay acts on activity at fixed locations, as in a real retina. Options:
   per-unit delay cascades (too many states), a near-dispersionless position delay with the
   temporal smoothing moved onto bump amplitude, or a place-coded retina. Decide before step 3.
   **Confirmed after the thresholded readout:** the threshold (0.1–0.3) does not change the rise
   at all (SC 10/50/90% at 39/60/~98 ms for a 15° step, same as today's path; `e_held` is slower,
   52/81/124 ms), because the bump *location* glides — there is no old/new pair of bumps for the
   threshold to separate. The 6-stage gamma cascade alone spreads a step over 26–77 ms (10–90%);
   N = 24 would give 37–63 ms, N = 48 41–59 ms (+18 states per extra 6 stages per channel per eye).
   Whether the glide can trigger small saccades is testable only after switch-over (today the
   ~270 ms accumulator keeps the SG from firing during the glide).
5. **Attractor regime (tested 2026-10-05, not adopted).** With strong local excitation
   (g_exc 3.5, g_inh 1.0) the decoded position *jumps* instead of gliding, but ~90 ms late
   (hysteresis: the old bump holds until the new input overwhelms it), with ~5% inward bias, slow
   drift and memory-like persistence. Idea to try: **feedforward (input-driven) inhibition**, so the
   new input itself suppresses the old bump instead of waiting for the new bump to grow.
6. ~~**Cyclopean position/visibility normalisation (found 2026-10-06).**~~ **Resolved (option B,
   extended — "position should not depend on visibility"):** perception_cyclopean decodes each eye's
   position (cascaded position×visibility ÷ cascaded visibility), fuses positions, carries them as
   position × the fused visibility, and OUTPUTS `cyc.target_pos` as a position (tail ratio);
   fixational position noise now enters scaled by visibility (it is noise on position); the
   flat `C_pos` readouts in analysis / llm_pipeline / bench_gaze_holding switched to
   `pc.read_activations`. Consequence for today's path: the target memory now holds the true
   flash position, which exposed its drain flaw (staircase past the target), so the memory remap
   by the cerebellar saccadic EC became the single memory rule in both modes (old drain removed).
   Original finding: perception_cyclopean fuses
   target position as a weighted AVERAGE of the eyes' position × visibility, but target visibility
   as a binocular SUM, `clip(w_L + w_R, 0, 1)`. At full visibility both are 1 and
   `cyc.target_pos / cyc.target_visible` is exact; during a 20 ms flash (per-eye visibility ~0.38,
   fused ~0.68) it reads 6.2° for a 12° target, and the SC-driven flash saccade drops from 11.5° to
   5.9°. Options: (A) carry the matching denominator (fusion-weighted visibility) through the same
   cascade, +6 states, default untouched; (B) define the fused product as fused position × fused
   visibility, no new states, but changes today's path whenever per-eye visibility is partial.
   Also seen once (noise on, SC driving): a premature 1° saccade 78 ms after a target step, toward
   the gliding readout, before the main saccade — the small-saccade risk from the glide (#4).

## Steps

1. ✅ **Module skeleton** — `brain_models/superior_colliculus.py`: map geometry, `State` (map units),
   `step` (render input bump, low-pass, lateral kernel, rectification), readout (decoded vector +
   strength), registries. Joins `BrainState` as `sc`, shadow-only (nothing reads it).
   `analysis.sc_decoded(states)` for the readout; `states.html` regenerated.
   Verified: every non-SC state bit-identical; fixation readout matches today's within 0.02°;
   a 20 ms flash at 12° decodes at 11.97° (today's path: 4.4°); +33% wall time (0.07 → 0.093 s
   per simulated second).
2. **Shadow bench** — decoded SC position vs today's target position for: target steps, double
   steps, flashes (#9), catch-up saccades during pursuit / head motion, microsaccade-scale drift,
   and the saccade itself.
3. **Tune** σ, τ and the kernel for fast switching, correct flash amplitude, and rostral
   (microsaccade-scale) resolution.
4. ✅ (b) **Remapping** with the saccadic CD (+ thresholded readout with a `valid` signal).
   Shadow comparison (noise off; true error = target − gaze), 15° step, after saccade end:

   | after saccade end | true | `e_held` | today's path (cyclopean) | delayed retinal error (= SC input before remap) | SC readout now |
   |---|---|---|---|---|---|
   | 0 ms   | 0.13 | 0.16 | 14.43 | 12.56 | 1.97 |
   | 30 ms  | 0.06 | 0.00 | 8.86 | 5.07 | 0.12 |
   | 60 ms  | 0.04 | **2.84** (re-tracks the stale error) | 2.49 | 0.97 | −0.12 |
   | 100 ms | −0.15 | 0.95 | 0.22 | 0.07 | −0.02 |

   The remapped *input* matches the true error to ≤ 0.4° through the saccade; the ~2° at saccade
   end is the map's own lag (τ = 15 ms) as the bump sweeps back (1.6° with θ = 0.3, which cuts
   more of the trail). Without the MN low-pass on the CD the
   input carried a phantom undershoot of ~8% of the amplitude that took ~60 ms to clear.
   Flashes in the dark: 20 ms and 10 ms flashes at 12° decode at 11.8–12.1° (valid for 76 / 49 ms);
   today's path peaks at 4.4° / 2.2° and the SG makes 1–2° saccades (#9). With θ = 0.3 the 10 ms
   flash is not detected at all — hence θ = 0.1. `valid` falls to 0 between flashes (no memory in v1).
   Pursuit (20°/s step-ramp): SC ≈ `e_held` between catch-up saccades; both lag the true error
   because only the eye's own movement is remapped, not the target's.

   **Two-bump handover (2026-10-06).** Space-time plots of the horizontal meridian
   (`analysis.sc_meridian`; `outputs/sc_kymo_sweep.png` before, `outputs/sc_kymo_twobump.png`
   after) for 15° right, 25° left and 3° saccades:

   | | single remapped bump | two bumps |
   |---|---|---|
   | map activity during the saccade | slides from the target site to the rostral pole | old site fades in place, landing site grows in place |
   | readout at saccade end (true ≈ 0.05–0.1°) | 1.2–1.6° | −0.03–0.08° |
   | readout at intermediate values (> 25% from both ends) | 19–22 ms per saccade | 2–7 ms |
   | readout switch | sweeps with the bump | one jump, ~15 ms after onset (3° saccade: ~7 ms) |
   | `valid` minimum during the saccade | ≥ 0.95 | 0.66–0.89 at the handover |

   Flashes, steps and pursuit unchanged. Total activity dips to ~0.15 bump at the handover (the
   surround inhibition kills the old bump before the new one has grown) and the CD bump is
   narrower than a visual bump (~0.55–0.75 bump of total activity) because it carries its own
   surround — irrelevant to the position readout, relevant once `strength` feeds a trigger.
5. **Switch-over** — 🔶 behind a switch, tested (2026-10-06), default still shadow.
   `BrainParams.sc_drives_sg` (0 = shadow, default, bit-identical; 1 = SC drives). With 1, the SC
   readout (position [yaw, pitch] + `valid`; torsion still cyclopean) replaces the cyclopean target
   signal going into the target working memory (`perception_target`), which still feeds the SG.
   It goes through the memory, not straight to the SG, because the SC has no memory: a flash is
   valid for only 50–76 ms and the trigger accumulator needs ~270 ms.
   **The memory needed the same remap.** It was drained toward 0 by the delayed target EC, which
   the cerebellar saccadic-suppression gate mutes during saccades, so each saccade consumed only
   ~35% of it: after a correctly remembered 12° flash the eye stepped 11.5 → 7 → 4 → 2° (25° total).
   Today's path has the same flaw at small scale (2.1 → 1.1 → 0.4°). With the switch on, the memory
   is instead shifted by the saccadic EC (`x_mem −= ∫sac_vel`, `acts.cb.sac_vel`) → one saccade,
   then hold. Since 2026-10-06 this is the memory rule in both modes (see open question 6).
   Quick A/B (`outputs/sc_ab.png`): flash at 12° → 11.5° saccade (20 ms flash; today 2.1°) and
   10.2° (10 ms; today 0.5°); pursuit and fixation similar; step saccades after a previous saccade
   ~40 ms later (≈ 290 vs 250 ms) — the stale post-saccadic error no longer pre-charges the trigger
   accumulator (it rests at −0.44 instead of −0.19); the first saccade from rest is unchanged.
   Full metric gate, both modes on the same code (96 metrics, nothing written to `web/`): no metric
   newly fails; `pursuit_bode_gain_0p5hz` 0.686 FAIL → 0.709 pass; 35 moved. Largest moves:
   `sac_refractory_isi_ms` 258 → 330 ms (band ≤ 350 — the accumulator effect),
   `gaze_hold_net_drift_dark` 0.74 → 0.08°, `gaze_hold_spv_ecc_light` 0.53 → 0.60 °/s,
   `pursuit_bode_gain_1hz` 0.68 → 0.74, `sac_postsac_peak_long_noiseless` 4.85 → 4.46 °/s,
   `fix_microsaccade_rate` 0.525 → 0.500 /s (exactly at the band's lower edge).
   **After the 2026-10-06 refactor** (cyclopean input, cerebellar saccadic EC, visibility-independent
   cyclopean position, memory remap in both modes), gate re-run in both modes vs the run above:
   default model — no band status changes; `gaze_hold_net_drift_dark` 0.74 → 0.03°,
   `fix_microsaccade_rate` 0.525 → 0.575 /s, `gaze_hold_spv_ecc_light` 0.53 → 0.58 °/s,
   `listing_ocr_torsion_err` 0.056 → 0.038°, the rest < 3%. SC driving — no band status changes;
   `sac_postsac_peak_long_noiseless` 4.46 → 6.49 °/s (already failing in every mode; worse than
   the default's 4.85 — to look at with the post-saccadic oscillation), `…_short_` 6.38 → 5.91.
   **Triggering — deferred to a review "what does triggering need with the new SC" (2026-10-07).**
   Since the SG's target path gets the predicted position (no stale post-saccadic error to
   pre-charge the accumulator): default `fix_microsaccade_rate` 0.40 /s (FAIL, band ≥ 0.5; 0.60
   with the SC driving), `sac_refractory_isi_ms` 335 ms (band ≤ 350), saccade latency ~230 ms
   (first) / ~300 ms (later). Candidates: retune the brainstem accumulator, or move build-up /
   triggering into SC dynamics (step 6).
6. **Later** — design fork: **does the SC hold the latched saccade vector?** Today the brainstem's
   resettable integrator `e_held` latches the error at onset and is driven to 0 by the burst; the SC
   only supplies the goal. Alternative: the motor-site activity persists through the saccade and its
   amplitude falls with the remaining motor error (Waitzman et al. 1991; Lefèvre, Quaia & Optican 1998;
   Goossens & Van Opstal 2006), so the SC is the latch and `e_held` goes. That would change the
   two-bump handover (the old site must decay with motor error, not be suppressed at onset), and
   quick phases (no SC target) would still need a brainstem loop. Accumulator / build-up as SC dynamics; memory as persistent activity; target slots
   and task inputs (#18); map optimisations (anisotropy, rostral redundancy).
