# Benchmark Reference Papers

Systematic index of every paper cited by the benchmark suite — both the figure
captions (`citation`) and the **acceptance-band sources** (`cite` in
`web/benchmarks/metrics_ranges.json`). The model's policy is that *every band is
anchored to the literature*, so every band source below should resolve to a PDF
in this folder.

**Folder convention** — one subfolder per benchmark-page section; the vestibular
sections (VOR/OKR, Gravity, T-VOR) share `self motion and vestibular/`:

| Page section | Folder |
|---|---|
| 1. Saccades | `saccades/` |
| 2. VOR / OKR | `self motion and vestibular/` |
| 3. Gravity Estimator | `self motion and vestibular/` |
| 4. Smooth Pursuit | `pursuit/` |
| 5. Near response (vergence + accommodation) | `near response/` (+ `context/`) |
| 6. Fixation | `fixation/` |
| 7. Listing's Law | `listing/` |
| 8. Translational VOR (T-VOR) | `self motion and vestibular/` |
| (Plant subsystem) | `plant/` |

**Legend** — ✓ PDF present · ✗ **MISSING (to download)** · ★ band source (an
acceptance band depends on it) · ○ context / supporting (not a band source).

---

## 1. Saccades — `saccades/`

| Paper | Role | PDF |
|---|---|---|
| ★ Bahill, Clark & Stark (1975) *Math Biosci* 24:191 — main sequence | `sac_peak_vel_20deg`, `sac_mainseq_resid_max` | ✗ MISSING |
| ★ Robinson (1975), in *Basic Mechanisms of Ocular Motility* — local-feedback burst, pulse-step | `sac_primary_gain`, `sac_endpoint_err_*`, `vor_okr_vor_direct_gain` | ✗ MISSING |
| ★ Becker & Jürgens (1979) *Vision Res* 19:967 — double-step refractoriness | `sac_refractory_isi_ms`, `sac_primary_err_*` | ✗ MISSING |
| ★ Smit, Van Gisbergen & Cools (1987/1990) *Vision Res* — oblique trajectory straightness | `sac_oblique_straightness`, `sac_oblique_sync_ms` | ✗ MISSING |
| ★ Van Gisbergen, Van Opstal & Roebroek (1985) — oblique component synchrony | `sac_oblique_straightness` | ✗ MISSING |
| ★ Kapoula, Robinson & Hain (1986) *Exp Brain Res* 61:386 — post-saccadic drift | `sac_postsac_drift_noiseless`, `sac_postsac_drift_noisy` | ✗ MISSING |
| ○ Scudder, Kaneko & Fuchs (2002) *Exp Brain Res* 142:439 — brainstem burst generator review | cascade figure | ✗ MISSING |
| ○ Otero-Millan et al. (2018) — common saccade/microsaccade trigger | SG design | ✓ `Otero-Millan_2018_SaccadeTrigger.pdf` |

## 2. VOR / OKR — `self motion and vestibular/`

| Paper | Role | PDF |
|---|---|---|
| ★ Raphan, Matsuo & Cohen (1979) *Exp Brain Res* 35:229 — velocity storage | `vor_okr_okan_tc`, `vor_okr_postrot_tc`, `vor_okr_okn_ss_gain`, `vor_okr_okr_ss_gain`, `vor_okr_vvor_gain` | ✓ `Raphan_1979_VelocityStorage.pdf` |
| ★ Cohen, Matsuo & Raphan (1977) *J Physiol* 270:321 — VS time constant | `vor_bode_gain_*`, `vor_bode_phase_*`, `okr_bode_*` | ✗ MISSING |
| ★ Robinson (1975) — VOR direct gain (see §1) | `vor_okr_vor_direct_gain` | ✗ MISSING |
| ○ Wilson & Melvill Jones (1979) *Mammalian Vestibular Physiology* — canal/otolith sensors | sensory model | ✓ `Wilson_MelvillJones_1979_VestibularPhysiology.pdf` |

## 3. Gravity Estimator — `self motion and vestibular/`

| Paper | Role | PDF |
|---|---|---|
| ★ Laurens & Angelaki (2011) *Exp Brain Res* 210:407 — gravity estimator (OCR, OVAR, tilt) | `gravity_ocr_gain`, `gravity_ocr_sin_rmse`, `gravity_ovar_mod_*`, `gravity_tilt_tc_*` | ✗ MISSING |
| ○ Boff, Kaufman & Thomas (1986) *Handbook of Perception & Human Performance* — OCR magnitude | OCR figure | ✗ MISSING |
| ○ Denise, Darlot, Droulez, Cohen & Berthoz (1988) *Exp Brain Res* 67:629 — perceived translation in OVAR | OVAR figure | ✗ MISSING |
| ○ Wood (2002) *J Vestib Res* 12:223 — tilt–translation discrimination | OVAR figure | ✗ MISSING |
| ○ Mayne (1974) — somatogravic / vestibular systems concept | somatogravic figure | ✗ MISSING |
| ○ Laurens & Angelaki (2013) *Nat Neurosci* — internal model of gravity (Kalman) | gravity estimator design | ✓ `Laurens_Angelaki_2013_InternalModelVestibular.pdf` |
| ○ Laurens & Angelaki (2017) — 3D VOR / Kalman | gravity + 3D VOR design | ✓ `Laurens_Angelaki_2017_3DVOR.pdf` (+ `Laurens Matlab code/`) |

## 4. Smooth Pursuit — `pursuit/`

| Paper | Role | PDF |
|---|---|---|
| ★ Lisberger & Westbrook (1985) *J Neurosci* 5:1662 — pursuit steady-state gain | `pursuit_ss_gain_5degs`, `pursuit_ss_gain_10degs` | ✗ MISSING |
| ★ Lisberger, Evinger, Johanson & Fuchs (1981) *J Neurophysiol* 46:229 — pursuit dynamics/Bode | `pursuit_bode_gain_low`, `pursuit_bode_bw_hz`, `pursuit_bode_phase_0p5hz` | ✗ MISSING |
| ★ Rashbass (1961) *J Physiol* 159:326 — step-ramp; pursuit latency | `pursuit_latency_ms` | ✗ MISSING |
| ★ Carl & Gellman (1987) *J Neurophysiol* 57:1446 — pursuit initiation latency | `pursuit_latency_ms` | ✗ MISSING |
| ○ Krauzlis & Lisberger (1994) — pursuit model | pursuit design | ✓ `Krauzlis_Lisberger_1994_PursuitModel.pdf` |
| ○ Luebke & Robinson (1988) — pursuit↔fixation transition | pursuit design | ✓ `Luebke_Robinson_1988_PursuitFixationTransition.pdf` |
| ○ Orban de Xivry et al. (2013) — Kalman pursuit | pursuit design | ✓ `OrbanDeXivry_2013_KalmanPursuit.pdf` |
| ○ Robinson, Gordon & Gordon (1986) — smooth-pursuit model | pursuit design | ✓ `Robinson_1986_SmoothPursuitModel.pdf` |

## 5. Near response (vergence + accommodation) — `near response/`

| Paper | Role | PDF |
|---|---|---|
| ★ Rashbass & Westheimer (1961) *J Physiol* 159:339 — vergence dynamics | `verg_tc_s`, `verg_latency_ms`, `verg_bode_*` | ✓ `context/Rashbass_Westheimer_1961_DisjunctiveEyeMovements.pdf` |
| ★ Mays (1984) *J Neurophysiol* 51:1091 — vergence neurons | `verg_sym_conv_err`, `verg_sym_version_leak` | ✗ MISSING |
| ★ Zee, FitzGibbon & Optican (1992) *J Neurophysiol* 68:1624 — saccade–vergence interactions | `verg_asym_facilitation`, `verg_conv_div_ratio` | ✓ `Zee_1992_SaccadeVergence.pdf` |
| ★ Collewijn, Erkelens & Steinman (1988) *J Physiol* 404:157 — conv/div asymmetry | `verg_conv_div_ratio` | ✗ MISSING |
| ★ Schor (1979) *Vision Res* — AC/A, CA/C cross-links | `aca_ratio_pd_per_d`, `aca_vergence_delta` | ✗ MISSING |
| ★ Morgan (1944) *Am J Optom* — clinical AC/A | `aca_vergence_delta` | ✗ MISSING |
| ★ Stark, Takahashi & Zames (1965) *IEEE Trans* — accommodation dynamics | `acc_bode_gain_low`, `acc_bode_bw_hz` | ✗ MISSING |
| ★ Hung, Semmlow & Ciuffreda (1986) *IEEE TBME* 33:1021 — dual-mode vergence | `verg_overshoot` | ✗ MISSING |
| ★ Read & Schor (2022) — predictive accommodation | accommodation figures | ✓ `Read_2022_PredictiveAccommodation.pdf` |
| ○ Hung et al. (1992) adaptation model | near-response design | ✓ `Hung_1992_AdaptationModel.pdf` |
| ○ Hung et al. (1997) convergence/divergence | near-response design | ✓ `Hung_1997_ConvergenceDivergence.pdf` |
| ○ Schor (1988) imbalanced adaptation | near-response design | ✓ `context/Schor_1988_ImbalancedAdaptation.pdf`* |
| ○ Schor (1999) accommodation lag | near-response design | ✓ `context/Schor_1999_AccommodationLag.pdf` |
| ○ Schor & Kotulak (1986) velocity-sensitive | near-response design | ✓ `context/Schor_Kotulak_1986_VelocitySensitive.pdf` |
| ○ Horwood & Riddell — gradient AC/A vs CA/C | near-response design | ✓ `context/Horwood_Riddell_GradientACA_vs_CAC.pdf` |
| ○ Del Aguila-Carrasco et al. (2017) — accommodation dynamics | near-response design | ✓ `DelAguila_2017_AccommodationDynamics.pdf` |

\* `Schor_1988_ImbalancedAdaptation.pdf` currently sits at the `near response/`
root, not in `context/` — move it to `context/` if you want it grouped with the
other supporting Schor papers.

## 6. Fixation — `fixation/`

| Paper | Role | PDF |
|---|---|---|
| ★ Rolfs (2009) — microsaccade statistics, OU drift model | `fix_drift_rms`, `fix_microsaccade_rate` | ✗ MISSING |

> `fix_noiseless_drift_std` is a **numerical sanity gate** (eye std with all noise
> disabled → ≈0); it has no physiological band and therefore no literature source.

## 7. Listing's Law — `listing/`

| Paper | Role | PDF |
|---|---|---|
| ○ Tweed, Haslwanter & Fetter (1998) *IOVS* 39:1500 — Listing's plane during saccades | listing figures | ✗ MISSING |
| ○ van Rijn & van den Berg (1993) *Exp Brain Res* — Listing during vergence | listing figures | ✗ MISSING |
| ○ Tweed (1998) *Science* 281:1363 — optimizing 3D gaze control | listing design | ✓ `Tweed_1998_GazeControl3D.pdf` |
| ○ Listing (1854) — historical (no PDF) | — | — |

> Listing's-law metrics are currently visual checks (no quantitative bands), so
> these are context references rather than band sources.

## 8. Translational VOR (T-VOR) — `self motion and vestibular/`

| Paper | Role | PDF |
|---|---|---|
| ★ Paige & Tomko (1991) *J Neurophysiol* 65:1170 — tVOR gain vs geometry | `tvor_bode_gain_high_*` | ✗ MISSING |
| ★ Angelaki & Hess (2001) — tVOR / 3D | `tvor_bode_gain_high_*` | ✗ MISSING |
| ★ Laurens & Angelaki (2011) — (shared with §3) | `tvor_bode_gain_high_*` | ✗ MISSING |

## Plant subsystem — `plant/`

| Paper | Role | PDF |
|---|---|---|
| ★ Robinson (1964) *J Physiol* 174:245 — first-order plant (τ_p ≈ 0.15 s) | plant time constant | ✗ MISSING |
| ○ Filip et al. (2018) — OpenSim oculomotor plant | plant design | ✓ `Filip_2018_OpenSimOculomotorPlant.pdf` |

## Canals / Sensory — `self motion and vestibular/`

| Paper | Role | PDF |
|---|---|---|
| ○ Goldberg & Fernández (1971) *J Neurophysiol* 34:635 — afferent resting discharge | canal floor | ✗ MISSING |
| ○ Fernández & Goldberg (1976) — otolith LP adaptation | otolith dynamics | ✗ MISSING |

---

## Downloads still needed (band sources first)

**Band sources (★) — required so every acceptance band is literature-anchored:**

- Saccades: Bahill, Clark & Stark (1975); Robinson (1975); Becker & Jürgens (1979); Smit, Van Gisbergen & Cools (1987/1990); Van Gisbergen et al. (1985); Kapoula, Robinson & Hain (1986)
- VOR/OKR: Cohen, Matsuo & Raphan (1977); Robinson (1975)
- Gravity: Laurens & Angelaki (2011)
- Pursuit: Lisberger & Westbrook (1985); Lisberger et al. (1981); Rashbass (1961); Carl & Gellman (1987)
- Near response: Mays (1984); Collewijn et al. (1988); Schor (1979); Morgan (1944); Stark, Takahashi & Zames (1965); Hung, Semmlow & Ciuffreda (1986)
- Fixation: Rolfs (2009)
- T-VOR: Paige & Tomko (1991); Angelaki & Hess (2001); Laurens & Angelaki (2011)
- Plant: Robinson (1964)

**Context (○) — nice to have:** Scudder et al. (2002); Boff, Kaufman & Thomas
(1986); Denise et al. (1988); Wood (2002); Mayne (1974); Tweed, Haslwanter &
Fetter (1998) IOVS; van Rijn & van den Berg (1993); Goldberg & Fernández (1971);
Fernández & Goldberg (1976).
