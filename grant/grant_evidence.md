# Grant Evidence Ledger

Verified against repository HEAD `4ccbd33` on 2026-09-25. Stored quantitative benchmark data were generated at commit `2ae5369`, three commits behind HEAD (`web/benchmarks/benchmarks_data.json`).

## Architecture and dimensions

- The simulator is a JAX/Diffrax ODE solved with fixed-step Heun; defaults are `dt = 0.001 s` and `warmup = 3.0 s` (`src/oculomotor/sim/simulator.py`).
- Inputs are 6-DOF head and scene kinematics, a 3-D Cartesian target, per-eye scene/target motion and visibility, prisms, and lenses (`src/oculomotor/sim/simulator.py`).
- Default output is bilateral 3-D eye rotation `(T, 6)`; `return_states=True` returns the full ODE state PyTree (`src/oculomotor/sim/simulator.py`).
- ODE state dimension is `200 + 201 + 12 + 1 + 2 + 2 = 418`: sensory, brain, bilateral eye plant, accommodation, pupils, and eyelids (`src/oculomotor/models/sensory_models/sensory_model.py`; `src/oculomotor/models/brain_models/brain_model.py`; `src/oculomotor/models/plant_models/plant_model_second_order.py`; `src/oculomotor/models/plant_models/accommodation_plant.py`; `src/oculomotor/models/plant_models/pupil_plant.py`; `src/oculomotor/models/plant_models/eyelid_plant.py`).
- The structured parameter tree contains 185 named fields: 156 brain, 24 sensory, and 5 plant parameters (`src/oculomotor/models/brain_models/brain_model.py`; `src/oculomotor/models/sensory_models/sensory_model.py`; `src/oculomotor/models/plant_models/plant_model_first_order.py`; `src/oculomotor/models/plant_models/plant_model_second_order.py`).
- The active loop is sensory transduction -> cyclopean/self-motion processing -> saccade, pursuit, vergence/accommodation, and translational VOR drives -> neural integration -> final common pathway -> bilateral plants -> retinal feedback (`src/oculomotor/sim/simulator.py`; `src/oculomotor/models/brain_models/brain_model.py`).
- Version is shared centrally; vergence is split `+u_verg/2` to the left eye and `-u_verg/2` to the right eye before muscle routing (`src/oculomotor/models/brain_models/final_common_pathway.py`).

## Sensory and controller equations

- Semicircular-canal dynamics implement
  `H(s) = tau_c*s / [(1 + tau_c*s)(1 + tau_s*s)]`, with defaults `tau_c = 5 s`, `tau_s = 0.005 s`, spontaneous floor `80 spikes/s`, and ceiling `400 spikes/s` (`src/oculomotor/models/sensory_models/canal.py`; `src/oculomotor/models/sensory_models/sensory_model.py`).
- Otoliths are bilateral first-order low-pass filters of head-frame gravito-inertial acceleration with `tau_otolith = 0.02 s` (`src/oculomotor/models/sensory_models/otolith.py`; `src/oculomotor/models/sensory_models/sensory_model.py`).
- Default interpupillary distance is `0.064 m`; sharp visual delay is `0.05 s` (`src/oculomotor/models/sensory_models/sensory_model.py`).
- Pursuit uses `e_pred = (target_slip_ec - x_net)/(1 + K_phasic_pursuit)` and `dx/dt = rectified_drive - x/tau_pursuit` (`src/oculomotor/models/brain_models/pursuit.py`).
- Saccadic burst magnitude follows `g_burst * (1 - exp(-relu(e)/e_sat_sac))`; the generator includes held error, omnipause, accumulator, trigger, facilitation, depression, and bilateral EBN/IBN states (`src/oculomotor/models/brain_models/saccade_generator.py`).
- Neural-integrator adaptation obeys `dx_null/dt = (x_net - x_null_eff)/tau_ni_adapt`; default `tau_ni_adapt = 20 s` (`src/oculomotor/models/brain_models/neural_integrator.py`; `src/oculomotor/models/brain_models/brain_model.py`).
- Velocity-storage adaptation obeys `dx_null/dt = (omega_est_canal - x_null)/tau_vs_adapt`; defaults are `tau_vs = 20 s` and `tau_vs_adapt = 600 s` (`src/oculomotor/models/brain_models/perception_self_motion.py`; `src/oculomotor/models/brain_models/brain_model.py`).
- Translational VOR uses `omega_eye = -gaze_hat x v_head / distance` and `vergence_rate = IPD*(gaze_hat . v_head)/distance^2` (`src/oculomotor/models/brain_models/tvor.py`).
- Listing correction uses `T_LL = -(H-H0)*(V-V0)*pi/360`; the optional L2 term is `-listing_l2_frac*Vdot*radians(vergence)` and defaults to zero (`src/oculomotor/models/brain_models/listing.py`; `src/oculomotor/models/brain_models/brain_model.py`).
- Slow tonic vergence and accommodation each have default `20 s` time constants (`src/oculomotor/models/brain_models/brain_model.py`).
- Target memory holds for `2 s` and fades with `0.2 s` time constant (`src/oculomotor/models/brain_models/perception_target.py`; `src/oculomotor/models/brain_models/brain_model.py`).
- Cerebellar state dimension is 70; its states are fixed delay/efference-copy and nodulus dynamics, and its gains are parameters rather than updated weights (`src/oculomotor/models/brain_models/cerebellum.py`).

## Ocular plant

- The active eye plant is `plant_model_second_order` (`src/oculomotor/sim/simulator.py`).
- Its transfer function is `q/u = (1 + tau_see*s)/[(1 + tau_p*s)(1 + tau_muscle*s)]` (`src/oculomotor/models/plant_models/plant_model_second_order.py`).
- State equations are `dx_muscle/dt = (u - x_muscle)/tau_muscle` and `dq/dt = (x_muscle - q)/tau_p + (tau_see/tau_p)*dx_muscle/dt` (`src/oculomotor/models/plant_models/plant_model_second_order.py`).
- Defaults are `tau_p = 0.15 s`, `tau_muscle = 0.013 s`, `tau_see = 0.008 s`, and orbital limit `50 deg` (`src/oculomotor/models/plant_models/plant_model_second_order.py`).
- Each eye has six activation channels ordered LR, MR, SR, IR, SO, IO. A fixed `6 x 3` action matrix and its pseudoinverse map activation to horizontal, vertical, and torsional drive (`src/oculomotor/models/plant_models/muscle_geometry.py`).
- The active plant has no fields for muscle insertion/origin, muscle length, force-length or force-velocity curves, passive tissue stiffness, pulleys, tendon paths, globe mass/inertia, or gaze-dependent moment arms (`src/oculomotor/models/plant_models/plant_model_second_order.py`; `src/oculomotor/models/plant_models/muscle_geometry.py`).
- `k_orbital` is present in the shared `PlantParams` but is not read by the active second-order derivative (`src/oculomotor/models/plant_models/plant_model_first_order.py`; `src/oculomotor/models/plant_models/plant_model_second_order.py`).

## Adaptation and pathology controls

- Implemented history-dependent mechanisms are velocity-storage null adaptation, neural-integrator null adaptation, slow tonic vergence, slow accommodation, saccadic facilitation/depression, target sample-and-hold, and target working memory (`src/oculomotor/models/brain_models/perception_self_motion.py`; `src/oculomotor/models/brain_models/neural_integrator.py`; `src/oculomotor/models/brain_models/vergence_accommodation.py`; `src/oculomotor/models/brain_models/saccade_generator.py`; `src/oculomotor/models/brain_models/perception_target.py`).
- No simulator state or update equation changes a parameter value across trials (`src/oculomotor/sim/simulator.py`; `src/oculomotor/models/brain_models/brain_model.py`).
- Repository helpers configure unilateral vestibular hypofunction, vestibular-nucleus lesion, CN III palsy, CN III nuclear palsy, facial palsy, and Horner syndrome (`src/oculomotor/sim/simulator.py`).
- Final-common-pathway parameters provide separate left/right gains for CN III, IV, VI, and VII trunks; oculomotor, trochlear, abducens, central-caudal, and Edinger-Westphal nuclei; both MLFs; and 12 tonic muscle baselines (`src/oculomotor/models/brain_models/final_common_pathway.py`).
- The natural-language interface exposes 58 disorder-tagged patient parameters with YAML-defined numeric ranges (`src/oculomotor/schema/parameters_schema.yaml`; `src/oculomotor/llm_pipeline/patient_builder.py`).
- Lesion severity is represented by continuous gains/biases and laterality-specific fields; elapsed time since lesion is not a schema field (`src/oculomotor/schema/parameters_schema.yaml`; `src/oculomotor/llm_pipeline/patient_builder.py`).

## Quantitative validation inventory

- The canonical normal suite contains 11 sections and 37 figures: 32 behavior figures and 5 cascade figures (`src/oculomotor/benchmarks/bench_metrics.py`; `web/benchmarks/benchmarks_data.json`).
- Stored output contains 94 unique scalar metrics. `metrics_ranges.json` contains 95 named ranges, and `golden_metrics.json` contains 76 stored reference values (`web/benchmarks/benchmarks_data.json`; `web/benchmarks/metrics_ranges.json`; `src/oculomotor/benchmarks/golden_metrics.json`).
- Applying the repository evaluator to the stored data yields 54 `PASS`, 15 `FAIL`, 8 `DRIFT`, and 17 `NEW` metrics (`src/oculomotor/benchmarks/bench_metrics.py`; `web/benchmarks/benchmarks_data.json`).
- Section counts `(figures, metrics)` are: saccades `(4,10)`, VOR/OKR `(4,13)`, gravity `(7,25)`, pursuit `(3,8)`, near response `(8,21)`, fixation `(1,3)`, gaze holding `(2,6)`, Listing `(3,4)`, final common pathway `(1,4)`, pupil `(2,0)`, and eyelid `(2,0)` (`web/benchmarks/benchmarks_data.json`).
- Stored failing values include pursuit gain `0.797` at `5 deg/s` and `0.780` at `10 deg/s` against lower bound `0.8`; dark translational-VOR gain `0.0525` against range `0.3-1.5`; post-rotatory VOR time constant `35.825 s` against `10-30 s`; vergence-adaptation shift `8.313 deg` against `0.3-3 deg`; and vergence-adaptation time constant `200 s` against `5-40 s` (`web/benchmarks/benchmarks_data.json`; `web/benchmarks/metrics_ranges.json`).
- The stored clinical report has 16 qualitative figures in six sections and no scalar acceptance gate (`web/clinical_benchmarks/index.html`; `src/oculomotor/reports/run_clinical_benchmarks.py`).
- The repository test package contains only `tests/__init__.py`; quantitative regression logic resides in the benchmark harness (`tests/__init__.py`; `src/oculomotor/benchmarks/bench_metrics.py`).
- Golden metrics are snapshots of prior model output; physiological ranges are separate entries with citations or rationale (`src/oculomotor/benchmarks/golden_metrics.json`; `web/benchmarks/metrics_ranges.json`).
- No repository file defines a train/validation/test split or an independent patient cohort (`src/oculomotor/benchmarks`; `tests`).

## Synthetic data and inverse modeling

- Scenario duration is constrained to `0.5-120 s`; segments support constant, sinusoidal, and impulse profiles (`src/oculomotor/llm_pipeline/scenario.py`).
- Scenario execution uses `dt = 0.001 s` (`src/oculomotor/llm_pipeline/run.py`).
- A single run records time, bilateral 3-D eye position, conjugate eye velocity, head position/velocity, scene/target velocity, target Cartesian position, per-eye visibility/cover/prism values, pupil diameter, luminance, and eyelid state (`src/oculomotor/llm_pipeline/run.py`).
- Exported CSV omits latent neural/sensory states, the full parameter tree, head linear position, target Cartesian position, and visibility/cover/prism metadata (`src/oculomotor/server/app.py`).
- `simulate(return_states=True)` exposes all latent trajectories, but the simulator return value does not bundle the parameter tree (`src/oculomotor/sim/simulator.py`).
- Comparison execution supports 2-4 conditions and runs them sequentially (`src/oculomotor/llm_pipeline/scenario.py`; `src/oculomotor/llm_pipeline/run.py`).
- Default stochastic key is `jax.random.PRNGKey(0)`; callers may supply another key (`src/oculomotor/sim/simulator.py`).
- The fitting package is marked obsolete and retains an old flat-dictionary API for four VOR parameters: `tau_i`, `tau_p`, `tau_vs`, and `K_vs` (`src/oculomotor/fitting/__init__.py`; `src/oculomotor/fitting/optimize.py`).
- No current module implements posterior inference, likelihood estimation, Fisher information, profile likelihood, Sobol sensitivity, or simulation-based inference (`src/oculomotor/fitting`; `src/oculomotor`).

## LLM boundary

- The LLM is used to produce a structured scenario or comparison tool call; Pydantic models validate the structured object before simulator execution (`src/oculomotor/llm_pipeline/interpret.py`; `src/oculomotor/llm_pipeline/scenario.py`; `src/oculomotor/llm_pipeline/run.py`).
- Numeric patient parameters are limited to schema-defined fields and ranges before conversion to `Params` (`src/oculomotor/llm_pipeline/patient_builder.py`; `src/oculomotor/schema/parameters_schema.yaml`).
- Rerun reconstructs a stored scenario and executes it without another LLM interpretation (`src/oculomotor/server/app.py`).
- The browser server waits for simulation completion before returning a result page; it does not stream simulator state for real-time closed-loop control (`src/oculomotor/server/app.py`).
- The prompt lists `K_phasic_verg = 1.0`, `K_verg = 1.25`, and `tau_verg = 5.0 s`; active defaults are `12.0`, `2.5`, and `3.0 s`, respectively (`src/oculomotor/llm_pipeline/prompt.py`; `src/oculomotor/models/brain_models/brain_model.py`).

## Evidence-bounded claims

- Supported: a parameterized, bilateral 3-D ocular-motor ODE with explicit sensory, premotor, final-common-pathway, eye-plant, accommodation, pupil, and eyelid states (`src/oculomotor/sim/simulator.py`).
- Supported: graded and lateralized neural/vestibular perturbations and reproducible synthetic trajectories under structured stimuli (`src/oculomotor/models/brain_models/final_common_pathway.py`; `src/oculomotor/llm_pipeline/run.py`).
- Supported: 94 stored scalar benchmark outputs plus qualitative clinical simulations (`web/benchmarks/benchmarks_data.json`; `web/clinical_benchmarks/index.html`).
- Not implemented: anatomically parameterized extraocular-muscle/orbital biomechanics, trial-to-trial parameter learning, a current parameter-inference pipeline, a population/cohort sampler, held-out patient validation, or prospective clinical-test optimization (`src/oculomotor/models/plant_models/plant_model_second_order.py`; `src/oculomotor/fitting`; `src/oculomotor/llm_pipeline`).
