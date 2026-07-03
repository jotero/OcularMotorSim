# Pulse-slide-step saccades with a series-elastic plant zero

Implementation note. This is the standard **pulse-slide-step** model of saccadic
innervation (Optican & Miles 1985) driving a **Robinson pole-zero plant**, plus one
addition — a lead term in the burst generator's local feedback. Nothing here is novel;
this note records how the pieces map onto our model and how they are tuned.

## 1. The plant has a series-elastic zero

The 2nd-order plant (`plant_model_second_order.py`) is not a bare two-pole low-pass. The
extraocular muscle's **series-elastic element (SEE)** transmits force *changes* to eye
position with a lead, giving a numerator zero (Robinson 1964/1981; Optican & Miles 1985
used a 4th-order plant with a zero for exactly this):

```
q_eye / motor_cmd  =  (1 + Tz·s) / [ (1 + τ₁·s)(1 + τ₂·s) ]
```

- `τ₁` = orbital slow pole (`tau_p` ≈ 150 ms), `τ₂` = muscle force-development pole
  (`tau_muscle` ≈ 13 ms), `Tz` = SEE zero (`tau_see` ≈ 8 ms).
- Realised with **no new state**: the position velocity gets the series-elastic lead term
  `w = (x_musc − x_pos)/τ₁ + (Tz/τ₁)·dx_musc`, since `x_pos/x_musc = (1+Tz s)/(1+τ₁ s)`.

The zero is *why* a bounded neural command can drive a fast eye: the plant's own lead
supplies the initial velocity, so the innervation does not need an aggressive
(differentiating, ringing) pulse. Without it, a fixed slide low-pass smooths the whole
saccade uniformly and flattens the main sequence — the small end worst.

## 2. The neural command is a pulse-slide-step

The NI (`neural_integrator.py`) issues Optican & Miles' three-component command — a **step**
(tonic position), a **pulse** (phasic velocity feedthrough), and a **slide** (a low-pass of
the pulse). The slide realises the plant-inverse lead compensation as a *smooth* branch, so
there is no burst-offset acceleration spike. Numerically-stable form (bounded coefficients;
`slide = LP(u_vel, Ts)`, `slide' = (u_vel − slide)/Ts`):

```
u_p = x_net + (τ_p + τ_f − Ts)·slide + (τ_p·τ_f)·slide'
```

`τ_f = τ_mn + τ_muscle` (lumped fast pole; MN membrane + muscle), `Ts` = slide TC
(`tau_slide`). The slide pole `1/Ts` **cancels the plant's SEE zero `1/Tz`**:

- `Ts = Tz` → exact cancellation → `eye` tracks `NI_net` with full peak velocity, but a
  small residual overshoot glissade (from lumping the two fast poles into one).
- `Ts > Tz` → the extra low-pass damps that overshoot for a clean landing, at a small
  peak-velocity cost (worst on short saccades).
- `Ts < Tz` → undershoot glissade.

This `Ts`↔`Tz` mismatch **is** the clinical pulse/slide-mismatch that produces glissades
(Optican & Miles' central point). Default `Ts = 10 ms` vs `Tz = 8 ms` — the balance point.

## 3. A lead term damps the burst generator's local-feedback loop

Robinson's local-feedback burst generator (`saccade_generator.py`) is a 2nd-order loop:
the resettable integrator `e_held` (which decrements `−u_burst` to zero) plus the
burst-neuron membrane (`τ_bn`). It is **underdamped**: when `e_held` reaches 0 the BN
membrane lag keeps the burst going, `e_held` overshoots negative, and the eye rings — a
post-saccadic glissade that is worst on small saccades (where `τ_bn` is a large fraction of
the burst). `τ_bn` sets *both* the damping and the burst strength, so it cannot simply be
lowered (that guts peak velocity).

Fix — a **lead** in the burst drive (pole-zero cancellation in the neural loop, mirroring
the SEE zero in the plant):

```
drive = relu( e_held − k_bn_lead·τ_bn·u_burst )      (was relu(e_held))
```

Since `de_held/dt = −u_burst` during the burst, `−τ_bn·u_burst` is a derivative/lead that
cancels the membrane lag, so `e_held` lands on 0 instead of overshooting. Mid-burst
`u_burst` is roughly constant, so the lead is a small offset there (peak velocity kept); it
only bites at the stop. `k_bn_lead = 1` is full lag cancellation; `0` recovers the old ringy
drive.

## Parameters

| param | where | default | role |
|---|---|---|---|
| `tau_see` | PlantParams | 8 ms | SEE plant zero `Tz` |
| `tau_slide` | BrainParams | 10 ms | slide TC `Ts` (glissade knob vs `Tz`) |
| `k_bn_lead` | BrainParams | 1.0 | burst-drive lead (× `τ_bn`) |

## Known-open

The small-saccade **main-sequence compression** (peak velocity below `700(1−e^{−A/7})` on the
short end) is a separate, pre-existing issue: the BN membrane charges from the −8° OPN clamp
toward a *decreasing* `e_held`, so `x_ebn` never reaches the amplitude on short saccades
(`x_ebn,peak ≈ 0.69·A` at 5°). It is a burst-*onset* dynamics problem, not the nonlinearity
(`e_sat_sac` already equals the target main-sequence constant) and not the visual delay
(sharpening it 3× changes nothing — `e_held` latches the settled error). The slide + lead add
~15% to it. Deferred.

## References

- Optican LM, Miles FA (1985). Visually induced adaptive changes in primate saccadic
  oculomotor control signals. *J Neurophysiol* 54(4):940–958. (Pulse-slide-step; 4th-order
  pole-zero plant; glissade = slide/step mismatch.)
- Robinson DA (1964, 1975, 1981). Ocular plant mechanics; local-feedback burst model.
