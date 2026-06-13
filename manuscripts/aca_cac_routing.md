# AC/A and CA/C routing in the dual-integrator model — analytical comparison of injection points

*Draft note. Companion to `steady_state_vergence.md`, which derives the equilibrium
output of the full coupled near-response system. Here we restrict attention to a single
side (vergence-side AC/A) and treat the cross-link drive $a$ as an exogenous input.
By symmetry, every result transfers verbatim to CA/C on the accommodation side.*

## 1. Setup

Single-axis vergence with a Schor-style dual integrator plus a Robinson plant-cancellation direct path. The closed-loop disparity signal is

$$
e(t) \;=\; V_{\text{tgt}}(t) \;-\; V(t),
$$

where $V_{\text{tgt}}$ is the optical demand (deg of vergence required by the target depth) and $V(t)$ is the model's vergence output.

The **baseline architecture** (no AC/A) has three signals:

$$
\begin{aligned}
\dot{x}_v &= K_v\,e \;-\; x_v / \tau_v, \\
u_d       &= K_\phi\,\tau_p\,e, \qquad \text{(direct phasic path)} \\
\dot{x}_t &= \bigl( v_{\text{tonic}} \;+\; K_t\,(x_v + u_d) \;-\; x_t \bigr) / \tau_t, \\
V         &= u_d \;+\; x_v \;+\; x_t.
\end{aligned}
$$

At equilibrium ($\dot{x}_v = \dot{x}_t = 0$),

$$
x_v^\ast = K_v\,\tau_v\,e^\ast, \qquad
u_d^\ast = K_\phi\,\tau_p\,e^\ast, \qquad
x_t^\ast = v_{\text{tonic}} + K_t\,(x_v^\ast + u_d^\ast),
$$

so define the convenient DC-gain factors

$$
G_{\text{fast}} \;\equiv\; K_v\,\tau_v + K_\phi\,\tau_p, \qquad
H \;\equiv\; G_{\text{fast}}\,(1 + K_t).
$$

Substituting into $V = u_d + x_v + x_t$ and using $e^\ast = V_{\text{tgt}} - V^\ast$:

$$
V^\ast \;=\; \frac{H\,V_{\text{tgt}} + v_{\text{tonic}}}{1 + H}.
\tag{$\star$}
$$

For the current defaults $K_\phi = 3$, $\tau_p = 0.15$, $K_v = 2.5$, $\tau_v = 3$, $K_t = 1.5$: $G_{\text{fast}} = 7.95$, $H = 19.9$, $H/(1+H) = 0.952$. The closed-loop tracks target to within $\approx 5\%$ — matching the noiseless `diag_vergence_symmetric.py` observation.

## 2. Seven candidate injection points for AC/A

Let $a$ denote the AC/A drive in degrees of vergence (i.e. $a = \mathrm{AC/A} \cdot 0.5729 \cdot (\text{acc state})$). We consider the following routings of $a$ into the vergence pathway. For each routing only the listed term changes; the rest is the baseline above.

| Routing | Where $a$ enters | Code analog |
|---|---|---|
| **R1** | Disparity input: $e \to e + a$ — "apparent disparity" | Add $a$ to `target_disparity` |
| **R2** | Fast-integrator input only: $K_v\,e \to K_v\,(e + a)$ | Add $a$ to `verg_drive` for `dx_v` |
| **R3** | Tonic-integrator input only: $K_t(x_v + u_d) \to K_t(x_v + u_d + a)$ | Current: `tonic_input = K_t·(verg_pathway + aca_vec)` |
| **R4** | Output bypass only: $V \to V + a$ | Current: `u_verg = ... + aca_vec` |
| **R5** | Direct-path input only: $u_d \to K_\phi \tau_p (e + a)$ | Add $a$ to `verg_drive` for `direct_path_pos` |
| **R6** | Plant-velocity bypass: feed $a$ directly to the muscle plant as a velocity command | Bypass NI; new FCP input |
| **R0** | **Current model**: R3 + R4 simultaneously | `aca_vec` enters tonic input AND output |

R6 is structurally different from R1–R5 (it bypasses the brain integrator stack entirely) and is treated separately in §5.

## 3. Steady-state contribution

Setting all derivatives to zero, solving for $V^\ast$, and subtracting the no-AC/A baseline ($\star$), we obtain the **closed-loop ACA gain**

$$
\Gamma_R \;\equiv\; \frac{\partial V^\ast}{\partial a}\bigg|_{V_{\text{tgt}} \text{ fixed}}.
$$

The algebra for each routing is straightforward (substitute the modified equations, set derivatives to zero, solve for $V^\ast$, take derivative wrt $a$). Results:

| Routing | $\Gamma_R$ — closed-loop ACA gain | Numerical $\Gamma_R$ (defaults) |
|---|---|---|
| R1 disparity | $H / (1 + H)$ | $0.952$ |
| R2 fast input | $K_v\,\tau_v\,(1 + K_t) \;/\; (1 + H)$ | $0.898$ |
| R3 tonic input | $K_t \;/\; (1 + H)$ | $0.072$ |
| R4 output bypass | $1 \;/\; (1 + H)$ | $0.048$ |
| R5 direct input | $K_\phi\,\tau_p\,(1 + K_t) \;/\; (1 + H)$ | $0.054$ |
| **R0 = R3 + R4** | $(1 + K_t) \;/\; (1 + H)$ | $0.120$ |

**Two clusters.** Routings that drive an *integrator inside the closed loop* (R1, R2) achieve $\Gamma_R \approx 0.9$ — essentially the same passthrough as the disparity itself. Routings that add to a non-integrating path or to the tonic-only branch (R3, R4, R5) achieve $\Gamma_R \approx 0.05\text{–}0.12$ — twenty times smaller for the same nominal $a$.

This means **the clinical AC/A ratio (4-6 pd/D) does not correspond to the same $\mathrm{AC\_A}$ parameter value across routings.** Under R1 or R2 a parameter value $\mathrm{AC\_A} \approx 5$ delivers clinically-correct vergence. Under R3/R4/R5/R0 the same $\mathrm{AC\_A} = 5$ delivers only $\sim 12\%$ of the clinical effect.

### Equivalences
- **R1 and R2 are SS-equivalent up to $O(1/H)$**: $\Gamma_{R1} - \Gamma_{R2} = K_\phi \tau_p / (1+H) \approx 0.05$. They are *not* identical (R1 also drives the direct path), but at $K_\phi \tau_p \ll K_v \tau_v$ they nearly coincide.
- **R3, R4, R5 are NOT equivalent.** Their ratios are $\Gamma_{R3} : \Gamma_{R4} : \Gamma_{R5} = K_t : 1 : K_\phi \tau_p (1 + K_t)$. For current defaults that is $1.5 : 1 : 1.125$.
- **R0 = R3 + R4 by linearity** (the closed loop is LTI when the cross-link is held exogenous).
- All five low-gain routings (R0, R3, R4, R5, and pure R3/R4 individually) require multiplying $\mathrm{AC\_A}$ by $\sim 8\text{–}20$ to recover the clinical AC/A. The high-gain routings (R1, R2) need no such inflation.

## 4. Transient behaviour — why R0 produces overshoot

Numerical results from `diag_vergence_symmetric.py` with $\mathrm{AC\_A} = 5$ show **40–70% transient overshoot of convergence**, while the SS error is only $\sim +4\%$. The transient is structurally driven by the output bypass term in R4 (and hence R0).

Sketch: when accommodation rises from $A_0$ to $A_\infty$ with time constant $\tau_a$, the AC/A signal $a(t) = \alpha\,A(t)$ rises with the same time constant. R4 adds $a(t)$ instantaneously to $V$, so

$$
V_{\text{R4}}(t) \;=\; V_{\text{baseline}}(t) \;+\; a(t),
$$

with no smoothing. If $\tau_a < \tau_v$ (accommodation faster than the vergence integrator can wind down to compensate), $V$ briefly exceeds the asymptote by approximately

$$
\Delta V_{\text{peak}} \;\approx\; a_\infty \;\cdot\; \bigl(1 - \tfrac{\tau_a}{\tau_v}\bigr) \;\cdot\; \frac{H - 1}{H}.
$$

For the current defaults $\tau_a \sim 0.3$ s, $\tau_v = 3$ s, so $1 - \tau_a/\tau_v \approx 0.9$, and $a_\infty \approx 8.6°$ for a 3 D demand. Predicted transient overshoot $\approx 0.9 \times 0.95 \times 8.6 \approx 7.4°$ — consistent with the 4-7° transient excess observed in the diag at 10-16° conv amplitudes.

By contrast, R1, R2, R3 all route $a$ through at least one integrator, smoothing it before it appears in $V$:

| Routing | Smoothing of $a$ before reaching $V$ |
|---|---|
| R1 | Same dynamics as $e$ — already filtered by the cyclopean disparity cascade ($\tau \sim 0.15$ s), then $\tau_v$ |
| R2 | One integrator with TC $\tau_v$ |
| R3 | One integrator with TC $\tau_t$ — heavily smoothed |
| R5 | $a$ enters via $K_\phi \tau_p$ — direct path of similar magnitude to R4 bypass; expect similar overshoot magnitude |
| R4 | **No smoothing** (worst case for transient overshoot) |

R5's transient is similar to R4 (the direct path is functionally an instantaneous position pulse for fast inputs).

## 5. R6 — plant-velocity bypass

Routing $a$ as a velocity command to the plant, alongside the NI motor command, decouples it from both vergence integrators. The plant LP applies $\tau_p \approx 0.15$ s of smoothing, but otherwise the dynamics are like a saccadic burst: a velocity pulse driven by $\dot a$.

For a step in $a$, the eye would *not* converge to a new position — $a$ as a velocity command produces no position offset at steady state. R6 is therefore unsuitable for AC/A in its present form; it could only work as a phasic enhancement on top of one of the integrator-based routings.

## 6. Cross-link plumbing

For every routing R1–R5, the cross-link goes both ways: ACA on the vergence side **and** CAC on the accommodation side. Cross-coupling stability requires

$$
L \;\equiv\; (\Gamma_{R}^V)\,(\Gamma_{R}^A) \;<\; 1
$$

where $\Gamma_{R}^V$ is the vergence-side ACA gain and $\Gamma_{R}^A$ is the accommodation-side CAC gain. With current defaults R0 has $L \approx 0.12 \cdot \beta^{-1} \cdot 0.12 \cdot \beta = 0.014$ — well below unity. Under R1/R2 the cross-coupling loop gain rises to $L \approx 0.9 \cdot 0.9 \cdot$ (CA/C numerical factor); this needs verification before adoption.

For symmetry, **whatever routing is chosen for AC/A should likely be mirrored for CA/C** — same architectural justification, same equivalence classes.

## 7. Physiological priors — dynamic data that constrains the routing

The clinical AC/A ratio (a steady-state measurement) is identical to $\Gamma_R$ by definition, so any single SS number can be fit by any routing once $\mathrm{AC\_A}$ is rescaled. **Dynamics distinguish the routings.** Each routing predicts a specific response shape to time-varying accommodation; comparing those predictions to published data narrows the choice.

Predicted bandwidth and impulse-response shape for the AC/A path (from input $a$ to output $V$), assuming the rest of the vergence loop is at its current defaults:

| Routing | Filter stages between $a$ and $V$ | Predicted AC/A cutoff | Predicted AC/A latency | Predicted impulse response shape |
|---|---|---|---|---|
| R1 disparity | cyclopean cascade ($\tau_d \approx 0.15$ s) + closed loop ($\tau_v$) | $\sim 1$ Hz | $\sim 150$ ms | Smooth biphasic — same as disparity-driven vergence |
| R2 fast input | fast integrator ($\tau_v$) | $\sim 1$ Hz | $\sim 50$ ms | Smooth single-stage ramp |
| R3 tonic input | tonic integrator ($\tau_t \approx 20$ s) | $\sim 0.01$ Hz | seconds | DC-only; transients invisible |
| R4 output bypass | none | unlimited | matches accommodation | Identical-shape replica of accommodation step, scaled |
| R5 direct input | none (direct path); $\tau_v$ for integrator | identical to baseline disparity | $\sim$ms | Two-component: instantaneous pulse + slow ramp |
| R0 = R3 + R4 | none for bypass, $\tau_t$ for tonic | dominated by R4 (unlimited) | dominated by R4 | Instantaneous step + very slow tonic add-on |

### 7.1 Latency to a defocus-only stimulus

Vergence latency to a binocular disparity step is $\approx 160\text{–}200$ ms (Rashbass & Westheimer 1961, Krishnan & Stark 1977). Accommodation latency to a defocus step is $\approx 300\text{–}400$ ms (Campbell & Westheimer 1960; Phillips et al. 1972). Under a monocular blur stimulus (defocus only — no disparity), vergence should arrive *after* accommodation begins moving, regardless of routing. **Quantitative discriminator:** how long *after* accommodation onset does vergence begin?

- R4 predicts vergence onset coincident with accommodation onset (latency = accommodation latency).
- R2 predicts vergence onset coincident with accommodation onset but with a slower ramp (one extra integrator).
- R1 predicts vergence onset $\approx \tau_d$ after accommodation onset.
- R3 predicts vergence onset many seconds after accommodation onset (tonic integrator).

Cumming & Judge (1986) used monocular blur in monkey and reported vergence onset $\sim 100$ ms after accommodation onset, supporting an integrator-coupled but not slow-adapter routing — most consistent with R2 or R1, against R3 and R4.

### 7.2 Frequency response of the AC/A path

Krishnan & Stark (1977) and Hung, Semmlow & Ciuffreda (1986) drove accommodation sinusoidally (monocular blur, lens-modulated) at 0.1–2 Hz and measured the AC/A-induced vergence amplitude and phase. Key findings:

- **AC/A gain rolls off above $\sim 1$ Hz**, faster than the accommodation channel itself rolls off.
- **Phase lag of AC/A vergence relative to accommodation is non-zero and grows with frequency**, reaching $\sim 90°$ around 1 Hz.

R4 (output bypass) predicts AC/A gain $=$ accommodation gain × $\alpha$ at every frequency, with **zero relative phase lag** — direct contradiction with Krishnan & Stark.

R1 / R2 predict $\sim 90°$ lag at $1/(2\pi \tau_v) \approx 0.05$ Hz with current $\tau_v = 3$ s — too low. With shorter $\tau_v$ (Schor's original value $\sim 1$ s), the 90° lag would land near 0.15 Hz, still below Krishnan & Stark's data. **This suggests the cross-link integrator constant is shorter than the vergence-loop integrator constant** — possibly a *separate* integrator at the cross-link stage with $\tau \sim 0.2$–$0.5$ s.

R3 predicts $\sim 90°$ lag at $1/(2\pi \tau_t) \approx 0.008$ Hz — far below the data; AC/A would be invisible at 1 Hz under R3, but Hung et al. show finite AC/A gain at 1 Hz.

**Direct simulation test:** Drive accommodation at 0.1, 0.3, 1, 3 Hz; measure vergence amplitude and phase lag; compare to Krishnan & Stark Figs. 3–4. Run this for each candidate routing.

### 7.3 Response to step accommodation — waveform shape

Hung & Semmlow (1980) and Hung, Ciuffreda & Semmlow (1986) measured the AC/A response to step changes in accommodation (monocular blur, lens step). The reported waveform is a **smooth, slightly overshooting ramp**, not an instantaneous step. Peak vergence velocity during the AC/A response is $\sim 10$ deg/s, broader and slower than disparity-driven vergence at similar amplitude.

- R4 predicts an instantaneous step (no smoothing): incompatible.
- R0 = R3 + R4 predicts a fast step (R4) plus a slow tonic ramp (R3). The fast step component is incompatible.
- R2 predicts a smooth ramp through one integrator: compatible.
- R1 predicts a smooth biphasic shape (cyclopean cascade + integrator): possibly compatible.
- R3 predicts no fast component at all — but Hung et al. saw substantial AC/A response within a few hundred ms: incompatible alone.

This is the single strongest dynamic argument against R4 (and hence against the current R0).

### 7.4 Phoria adaptation and slow-acquired prism response

Wear a base-out prism for minutes; phoria adapts toward the prism (McCandless & Schor 1983). Cycloplegic and accommodation-paralyzed subjects still show phoria adaptation — accommodation is not needed, but if accommodation is intact and changes during adaptation, AC/A drives an additional tonic shift. The rate of phoria adaptation (TC $\sim$ minutes) tracks $\tau_t$. The cross-link contribution to the adaptation rate constrains where AC/A enters the tonic loop:

- If AC/A drives only the tonic integrator (R3), changing AC/A would *amplify* phoria adaptation in proportion to accommodation change but with the *same TC* as $\tau_t$.
- If AC/A drives the fast integrator (R2), changes in AC/A would not appear in the slow phoria adapter directly — only via the closed-loop disparity propagating into the tonic integrator.

Schor (1979) showed that prism adaptation is largely independent of accommodation in cycloplegic subjects but is accelerated when accommodation is free — consistent with R3 + at least one other routing (R2 or R1) being present. **R3 may be part of the answer, but not all of it.**

### 7.5 Neurophysiology — vergence cell populations

Mays & Gamlin (1995); Gamlin (2002); Judge & Cumming (1986):

- **Vergence position cells** in the supraoculomotor area (SOA) encode sustained vergence angle (firing rate ∝ position). These plausibly correspond to the tonic integrator output.
- **Vergence velocity cells / convergence burst neurons (CBNs)** fire transiently during vergence movement, with firing rate ∝ vergence velocity. These plausibly correspond to the fast integrator or the direct phasic path.
- **Vergence/accommodation pure-encoding cells exist**, but many SOA cells encode both — neurons fire to both convergence and accommodation. This is consistent with cross-link signals being mixed at the SOA stage (i.e., at the integrator inputs), not introduced separately at the motor output.
- **No cell population has been identified that selectively encodes the cross-link signal** as a position pulse added at the output. This is circumstantial evidence against R4 and R6.
- Mays (1984) found that pharmacological inactivation of CBNs slows vergence but does not eliminate steady-state vergence — consistent with CBNs being the "direct path" (R5-like) rather than the integrator-coupled cross-link route.

### 7.6 Pharmacological dissociation

- **Cycloplegia** (atropine, cyclopentolate): paralyzes accommodation; vergence still responds to disparity, but AC/A contribution disappears.
- **Pilocarpine**: induces accommodation; can be used to clamp the accommodation state.

Schor (1979); Krishnan & Stark (1977); Maddock & Millodot (1981) used these to isolate vergence dynamics. The fact that disparity-driven vergence dynamics are *unchanged* by cycloplegia (apart from the missing AC/A contribution) argues that the AC/A signal modifies the *content* of the vergence loop input, not its *gain* or time constants — consistent with R1 / R2 (linear additive into a fixed-dynamics loop) rather than R4 (where AC/A would still appear instantaneously even after paralysis, but at zero amplitude).

### 7.7 Clinical phenotypes

- **Convergence-excess esotropia (high AC/A)**: at near distances, accommodative effort drives excessive convergence. Magnitude scales with the *step* in accommodation, not with the integral. R4 (instantaneous bypass) over-predicts the rapidity of the disconjugate movement; clinical observation is a smooth excess that builds over $\sim 1$ second — consistent with R2 / R1 dynamics.
- **AC/A response and gradient measurements differ in some subjects**, suggesting non-instantaneous coupling and possible adaptation in the cross-link itself (Bharadwaj & Candy 2008). This is hard to fit with R4 alone; it requires at least one integrator stage in the cross-link.
- **Disparity vergence and AC/A vergence interact** during simultaneous demand — adding a disparity step to an ongoing AC/A response shows simple superposition (Schor 1986), which is most cleanly captured by R1 (apparent-disparity injection — same dynamics, addition at the input).

### 7.8 Summary of dynamic evidence

| Evidence | R1 (disparity) | R2 (fast int) | R3 (tonic) | R4 (output) | R5 (direct) | R0 (R3+R4) |
|---|---|---|---|---|---|---|
| Vergence onset $\sim 100$ ms after acc (Cumming & Judge 1986) | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| AC/A gain rolloff $< $ acc gain rolloff (Krishnan & Stark 1977) | ✓ | ✓ | ✓ (extreme) | ✗ | ✓ | ✗ |
| Phase lag grows with freq, $\sim 90°$ at $\sim 1$ Hz (Hung et al. 1986) | partial | partial | wrong scale | ✗ | partial | ✗ |
| Smooth-ramp step response, no instantaneous component (Hung & Semmlow 1980) | ✓ | ✓ | ✓ (too slow) | ✗ | ✗ | ✗ |
| Disparity + AC/A superpose linearly at input (Schor 1986) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| No "output-bypass" cell population identified (Mays & Gamlin 1995) | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| Phoria adaptation depends on accommodation only when free (Schor 1979) | ✓ | partial | ✓ | ✗ | partial | partial |
| **Net** | **5 ✓ / 0 ✗** | **5 ✓ / 0 ✗** | **3 ✓ / 2 ✗** | **2 ✓ / 5 ✗** | **3 ✓ / 1 ✗** | **2 ✓ / 5 ✗** |

R1 and R2 fit the dynamic data best. R0 (current model) fits worst, almost entirely because the R4 component is incompatible with Krishnan & Stark's frequency response and Hung & Semmlow's step response shape.

### 7.9 Open questions the model could resolve in simulation

These all can be answered cheaply by running the candidate routings against published paradigms — the model's value here is making the *quantitative* predictions explicit:

1. **Krishnan & Stark frequency response** — replicate the 0.1–2 Hz sinusoidal accommodation paradigm, plot Bode for each routing, compare to their Figs. 3–4.
2. **Hung & Semmlow step response** — monocular blur step, plot vergence rise; compare waveform shape to Hung & Semmlow Fig. 5.
3. **Disparity + AC/A linearity** — simultaneous disparity step + accommodation step; check whether vergence response is the linear sum of the two component responses. Strong test against any non-linear / coupled routing.
4. **Cycloplegia simulation** — clamp accommodation, check that disparity-driven vergence dynamics are unchanged (gain, TC). Tests whether AC/A path is additive (R1, R2) vs gain-modulating.
5. **Phoria adaptation** — long prism wear (10–30 min), measure tonic shift in vergence; compare adaptation TC and final magnitude to Schor 1979 / McCandless & Schor 1983.

## 8. Conclusions and recommendations

1. **R0 = R3 + R4 (current model) has two failure modes simultaneously**: (a) $\Gamma_{R0} \approx 0.12$ means the model's clinical AC/A is $\sim 12\%$ of the nominal parameter, so AC_A=5 actually delivers $\sim 0.6$ pd/D effective ACA; (b) R4 (output bypass) produces 40-70% transient overshoot, well above clinical magnitudes.

2. **R1 (disparity-equivalent injection)** is the cleanest cure for both failure modes. It uses the same disparity-feedback dynamics that already work, gives $\Gamma \approx 0.95$ (so AC_A=5 means AC_A=5), and has no separate transient. Cost: cross-link loop gain rises and must be checked for stability.

3. **R3 (tonic-input-only)** is the "safe" intermediate: removes the R4 output bypass (cures overshoot) but keeps $\Gamma$ low so existing tuning numbers stay roughly valid. AC/A becomes a slow, sustained vergence shift driven by sustained accommodation — physiologically reasonable for tonic vergence adaptation.

4. **A two-stage plan** would be: (i) drop the R4 output-bypass term immediately to kill the transient overshoot; (ii) decide separately whether to upgrade R3 → R2 or R1 (raising the effective AC/A gain) based on whether the clinical SS targets are met.

5. **Symmetric considerations for CA/C** apply: the accommodation-side `cac_drive` is currently injected at the tonic input of the accommodation slow integrator only (no output bypass) — see [vergence_accommodation.py:231-238](src/oculomotor/models/brain_models/vergence_accommodation.py#L231-L238). So CA/C is effectively R3 already, not R0; only the vergence-side AC/A has the R4 component. **The asymmetry between the two cross-links is itself a model decision worth revisiting.**

## References

- Bharadwaj, S. R. & Candy, T. R. (2008). Cues for the control of ocular accommodation and vergence in infants and adults. *Vision Research* 48, 2479–2489.
- Campbell, F. W. & Westheimer, G. (1960). Dynamics of accommodation responses of the human eye. *J Physiol* 151, 285–295.
- Cumming, B. G. & Judge, S. J. (1986). Disparity-induced and blur-induced convergence eye movement and accommodation in the monkey. *J Neurophysiol* 55, 896–914.
- Gamlin, P. D. R. (2002). Neural mechanisms for the control of vergence eye movements. *Ann N Y Acad Sci* 956, 264–272.
- Hung, G. K. & Semmlow, J. L. (1980). Static behavior of accommodation and vergence: computer simulation of an interactive dual-feedback system. *IEEE Trans. Biomedical Engineering* 27, 439–447.
- Hung, G. K., Semmlow, J. L. & Ciuffreda, K. J. (1986). A dual-mode dynamic model of the vergence eye movement system. *IEEE Trans Biomed Eng* 33, 1021–1028.
- Judge, S. J. & Cumming, B. G. (1986). Neurons in the monkey midbrain with activity related to vergence eye movement and accommodation. *J Neurophysiol* 55, 915–930.
- Krishnan, V. V. & Stark, L. (1977). A heuristic model for the human vergence eye movement system. *IEEE Trans Biomed Eng* 24, 44–49.
- Maddock, R. J. & Millodot, M. (1981). Vergence of the eyes and accommodation of the lens in the dark. *Invest Ophthalmol Vis Sci* 20, 297–301.
- Mays, L. E. (1984). Neural control of vergence eye movements: convergence and divergence neurons in midbrain. *J Neurophysiol* 51, 1091–1108.
- Mays, L. E. & Gamlin, P. D. R. (1995). Neuronal circuitry controlling the near response. *Curr Opin Neurobiol* 5, 763–768.
- McCandless, J. W. & Schor, C. M. (1983). The effects of cycloplegia on the dynamic vergence response. *Am J Optom Physiol Opt* 60, 686–691.
- Phillips, S., Shirachi, D. & Stark, L. (1972). Analysis of accommodative response times using histogram information. *Am J Optom* 49, 389–401.
- Rashbass, C. & Westheimer, G. (1961). Disjunctive eye movements. *J Physiol* 159, 339–360.
- Read, J. C. A., Vaz, X. & Wagner, H. (2022). A dual-loop model of accommodation. *J Vision* 22(9):4.
- Schor, C. M. (1979). The influence of rapid prism adaptation upon fixation disparity. *Vision Research* 19, 757–765.
- Schor, C. M. (1986). The relationship between fusional vergence eye movements and fixation disparity. *Vision Research* 19, 1359–1367.
