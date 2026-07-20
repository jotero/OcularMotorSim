# Integrating the model into your own loop

> How to drive the **brain** and **plant** models directly — feeding them retinal
> output you simulate yourself (or that you read from `sensory_model`), one timestep
> at a time, inside a loop you control.
>
> If you just want to run a scenario end-to-end, use
> [`simulate()`](src/oculomotor/sim/simulator.py) instead — it wires everything for
> you. This guide is for the case where you need your *own* integration loop: a
> custom solver, an RL environment step, a real-time driver, or a different
> sensory front-end.

---

## 1. The three swappable stages

The model is three pure-function stages chained in signal-flow order. Each is a
state-space model with the same contract: `step(state, inputs, params) -> (dstate, outputs)`.
You own the state; the model only computes derivatives.

```
   your retinal sim                 brain_model.step            plant_model.step
   (or sensory_model)               ─────────────────           ─────────────────
   RetinaOut  ─┐                    BrainState  ─┐               PlantState  ─┐
   canal (6,) ─┼─► SensoryOutput ──► sensory_out ─┼─► nerves(12,)─┼─► q_eye, w_eye
   otolith(3,)─┘                    brain_params ─┘   ec_*, u_acc └─► (per eye)
```

The integration point between brain and plant is the **motor command interface**:
the brain emits `nerves` (12 per-muscle activations, `[L6 | R6]`), the plant turns
them into eye rotation. That is the only coupling you must respect.

| Stage | Module | `step` returns |
|---|---|---|
| Sensory (retina) | [`sensory_models/retina.py`](src/oculomotor/models/sensory_models/retina.py) | `(dstate, RetinaOut)` |
| Sensory (bundle) | [`sensory_models/sensory_model.py`](src/oculomotor/models/sensory_models/sensory_model.py) | `read_outputs(...) -> SensoryOutput` |
| Brain | [`brain_models/brain_model.py`](src/oculomotor/models/brain_models/brain_model.py) | `(dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc)` |
| Plant (eye) | [`plant_models/plant_model_first_order.py`](src/oculomotor/models/plant_models/plant_model_first_order.py) | `(dx_p, q_eye, w_eye)` |
| Plant (lens) | [`plant_models/accommodation_plant.py`](src/oculomotor/models/plant_models/accommodation_plant.py) | `(dx_acc, x_acc)` |

---

## 2. The data structures you pass around

### `RetinaOut` — what your retinal sim produces (per eye)

This is the boundary object. Everything downstream of the retina consumes a *delayed,
gated, eye-frame* `RetinaOut`. Defined in
[`retina.py`](src/oculomotor/models/sensory_models/retina.py):

```python
class RetinaOut(NamedTuple):
    scene_angular_vel: jnp.ndarray  # (3,) [yaw, pitch, roll] deg/s — gated by scene_visible + saturated
    scene_linear_vel:  jnp.ndarray  # (3,) [x, y, z] m/s (head frame, per-eye) — gated by scene_visible
    target_pos:        jnp.ndarray  # (3,) [yaw, pitch, 0] deg — gated by target_visible
    target_vel:        jnp.ndarray  # (3,) [yaw, pitch, 0] deg/s — gated by target_motion_vis + saturated
    scene_visible:     jnp.ndarray  # scalar — delayed scene_present
    target_visible:    jnp.ndarray  # scalar — delayed target_present × target_in_vf
    defocus:           jnp.ndarray  # scalar — delayed defocus (D)
    luminance:         jnp.ndarray  # scalar — per-eye afferent retinal luminance (~[0,1]) → pupil light reflex
```

> **These signals are already post-processing.** The model assumes the visual delay
> (sharp gamma cascade, ≈ `tau_vis_sharp`), the velocity saturation
> (`v_max_scene_vel`, `v_max_target_vel`), and the visibility gating have *already
> been applied*. If you supply your own `RetinaOut`, you are responsible for those —
> see [retina.step](src/oculomotor/models/sensory_models/retina.py#L464) for exactly
> what the built-in front-end does. If you don't want to reimplement them, generate
> the `RetinaOut` with the built-in retina (Pattern B below) and only swap the parts
> you care about.

### `SensoryOutput` — what the brain consumes

The brain takes one bundle per step. Defined in
[`sensory_model.py`](src/oculomotor/models/sensory_models/sensory_model.py#L162):

```python
class SensoryOutput(NamedTuple):
    canal:    jnp.ndarray   # (6,) canal afferent rates (deg/s-equivalent)
    otolith:  jnp.ndarray   # (3,) running GIA estimate in HEAD frame (m/s²)
    retina_L: RetinaOut     # left  eye  (incl. luminance afferent)
    retina_R: RetinaOut     # right eye
```

For a purely visual experiment (no head/vestibular input), `canal` and `otolith`
still must be physically sane — use the resting values (Section 4) rather than
zeros, because `otolith` carries gravity and drives the gravity estimator.

### State pytrees

Every state is a NamedTuple pytree (Diffrax-native). You never index a flat array.

```python
brain_x : brain_model.BrainState   # nested per-subsystem (pc, sm, pt, sg, pu, va, ni, fcp, cb)
plant_x : plant_model.State        # .left (3,), .right (3,)  eye rotation vectors (deg)
acc_x   : jnp.ndarray              # (1,) lens accommodation (D)
```

Build the initial states with the provided factories:

```python
# make_x0 needs b_vs in (6,) per-population form; default_params() leaves it scalar.
params  = params._replace(brain=params.brain._replace(
    b_vs=jnp.broadcast_to(jnp.asarray(params.brain.b_vs, jnp.float32), (6,))))
brain_x = brain_model.make_x0(params.brain)         # VS at equilibrium, OPN tonic, tonic vergence...
plant_x = plant_model.rest_state()                  # both eyes at primary position
acc_x   = jnp.array([params.brain.tonic_acc])       # lens at dark focus
```

> Don't hand-roll initial states. `make_x0` seeds load-bearing setpoints
> (OPN tonic = 100 blocks the saccade burst at rest, vergence tonic, MN slow
> manifold). A zero brain state will misbehave for several hundred ms.

---

## 3. The brain → plant contract (the important part)

This is the exact wiring `simulate()` uses, distilled to the brain+plant core. It
is the heart of "plug into another loop":

```python
import jax.numpy as jnp
from oculomotor.models.brain_models import brain_model
from oculomotor.models.plant_models import plant_model_first_order as plant_model
from oculomotor.models.plant_models import accommodation_plant as acc_plant
from oculomotor.models.plant_models.muscle_geometry import M_PLANT_EYE_L, M_PLANT_EYE_R

def derivatives(brain_x, plant_x, acc_x, sensory_out, params):
    # 1. Brain: sensory bundle -> per-muscle nerve activations + efference copies
    dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc = brain_model.step(
        brain_x, sensory_out, params.brain)          # noise_acc defaults to 0.0

    # 2. Plant (per eye): nerves[:6] drive L, nerves[6:] drive R.
    #    M_PLANT_EYE_* decode the 6 muscle activations into a 3-D motor command.
    dL, q_eye_L, w_eye_L = plant_model.step(plant_x.left,  nerves[:6], params.plant, M_PLANT_EYE_L)
    dR, q_eye_R, w_eye_R = plant_model.step(plant_x.right, nerves[6:], params.plant, M_PLANT_EYE_R)

    # 3. Accommodation plant (lens): driven by the brain's accommodation command.
    dacc, _ = acc_plant.step(acc_x, u_acc, params.brain.tau_acc_plant)

    dplant = plant_model.State(left=dL, right=dR)
    return (dbrain, dplant, dacc), (q_eye_L, q_eye_R, w_eye_L, w_eye_R)
```

Key facts:

- **`nerves` is `(12,)`, ordered `[L muscles 6 | R muscles 6]`.** Slice `[:6]` / `[6:]`.
- **`M_PLANT_EYE_L/R` is the `(3, 6)` decode matrix** (muscle pulling-direction
  pseudo-inverse). Passing it as the 4th arg makes the plant accept the 6-vector;
  omit it only if you pre-decode to a 3-vector yourself.
- **`q_eye` (position, deg) and `w_eye` (velocity, deg/s) are algebraic outputs** of
  the plant step — available immediately, no extra integration. `w_eye = dx_p`.
- **`ec_vel`, `ec_pos`, `ec_verg`** are efference copies. The brain and plant don't
  need them fed back, but the *sensory* model does if you run the full closed loop
  (they go into `retina.step` / `sensory_model.step`). For an open-loop brain+plant
  driver you can ignore them.
- **`u_acc`** (scalar diopters) is the lens-plant input; feed it to `acc_plant.step`.

---

## 4. A complete minimal driver — you supply the retina

This is the scenario in the request: you simulate the retinal output yourself and
push it through the brain and plant. The cleanest way to get physically-correct
`canal`/`otolith` baselines without re-deriving the gravity frame is to read them
once from the resting sensory state and then overwrite the retina fields:

```python
import jax
import jax.numpy as jnp
from oculomotor.sim.simulator import default_params
from oculomotor.models.sensory_models import sensory_model
from oculomotor.models.sensory_models.retina import RetinaOut
from oculomotor.models.brain_models import brain_model
from oculomotor.models.plant_models import plant_model_first_order as plant_model
from oculomotor.models.plant_models import accommodation_plant as acc_plant
from oculomotor.models.plant_models.muscle_geometry import M_PLANT_EYE_L, M_PLANT_EYE_R

params = default_params()
dt     = 0.001                      # keep <= 0.001 s; the visual cascade is stiff

# ── Normalise b_vs to (6,) — simulate() does this once before solving, and
#    make_x0 / brain_model.step both require the per-population array form. ──
params = params._replace(brain=params.brain._replace(
    b_vs=jnp.broadcast_to(jnp.asarray(params.brain.b_vs, jnp.float32), (6,))))

# ── Initial states ─────────────────────────────────────────────────────────
brain_x = brain_model.make_x0(params.brain)
plant_x = plant_model.rest_state()
acc_x   = jnp.array([params.brain.tonic_acc])

# ── Resting canal + otolith baseline (upright head, no acceleration) ───────
# read_outputs fills canal afferents and the head-frame GIA (gravity) correctly.
_base = sensory_model.read_outputs(
    sensory_model.rest_state(), params.sensory,
    q_head=jnp.zeros(3), a_head=jnp.zeros(3))
canal_rest, otolith_rest = _base.canal, _base.otolith

# ── Your retinal front-end ─────────────────────────────────────────────────
# Replace this with whatever produces delayed, gated, eye-frame signals.
#
# IMPORTANT: target_pos / target_vel / scene_angular_vel are RETINAL signals —
# error/slip *relative to the current eye*, not world coordinates. A faithful
# front-end recomputes them each step from the current eye position/velocity
# (q_eye, w_eye from the previous plant step), which makes the loop intrinsically
# closed. Feeding a CONSTANT value (below) is open-loop testing only: the eye will
# not converge the way it would when the error shrinks as the eye moves.
def my_retina(t):                                   # constant 10 deg error = open-loop demo
    return RetinaOut(
        scene_angular_vel = jnp.zeros(3),
        scene_linear_vel  = jnp.zeros(3),
        target_pos        = jnp.array([10.0, 0.0, 0.0]),   # retinal error [yaw, pitch, 0] deg
        target_vel        = jnp.zeros(3),
        scene_visible     = jnp.float32(0.0),
        target_visible    = jnp.float32(1.0),
        defocus           = jnp.float32(0.0),
        luminance         = jnp.float32(0.0),
    )

# ── One integration step (Euler shown for clarity; see note on Heun) ───────
@jax.jit
def step(carry, t):
    brain_x, plant_x, acc_x = carry
    retina = my_retina(t)
    sens = sensory_model.SensoryOutput(
        canal=canal_rest, otolith=otolith_rest,
        retina_L=retina, retina_R=retina)            # same image to both eyes (monocular case)

    dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc, u_pupil, u_lid = brain_model.step(brain_x, sens, params.brain)
    dL, q_eye_L, w_eye_L = plant_model.step(plant_x.left,  nerves[:6], params.plant, M_PLANT_EYE_L)
    dR, q_eye_R, w_eye_R = plant_model.step(plant_x.right, nerves[6:], params.plant, M_PLANT_EYE_R)
    dacc, _ = acc_plant.step(acc_x, u_acc, params.brain.tau_acc_plant)

    brain_x = jax.tree_util.tree_map(lambda x, d: x + dt * d, brain_x, dbrain)
    plant_x = plant_model.State(left=plant_x.left + dt * dL, right=plant_x.right + dt * dR)
    acc_x   = acc_x + dt * dacc
    return (brain_x, plant_x, acc_x), jnp.concatenate([q_eye_L, q_eye_R])

# ── Run ─────────────────────────────────────────────────────────────────────
t_array = jnp.arange(0.0, 1.0, dt)
(_, _, _), eye_rot = jax.lax.scan(step, (brain_x, plant_x, acc_x), t_array)
# eye_rot: (T, 6) = [L yaw,pitch,roll | R yaw,pitch,roll], degrees
```

> **Solver.** The example uses forward Euler so the loop is transparent. The model
> ships with **`diffrax.Heun()` at `dt = 0.001 s`** (`SimConfig` default), and the
> visual cascade requires `dt < 2·tau_stage`. If you keep your own retina front-end
> (so the stiff cascade lives *outside* this loop), Euler at `dt ≤ 0.001 s` is fine
> for brain+plant. If you advance the built-in `retina.step` inside your loop, use
> Heun (or a smaller `dt`). To reuse the shipped solver wiring instead, see Pattern A.

> **Warmup.** `simulate()` prepends a 3 s settling window (holding the stimulus at
> its `t=0` value) so fast states reach steady state before `t=0`, then strips it.
> A hand-rolled loop has none — either start every stimulus ramped from rest, or run
> your own warmup by looping on the `t=0` inputs before recording. (See the
> "ramp from 0" convention in `CLAUDE.md`.)

---

## 5. Closing the loop with the built-in sensory model

If you want the *full* feedback loop — eye movement changes what the retina sees —
you must run `sensory_model.step` too, and feed the eye outputs and efference copies
back in. The strict evaluation order (from
[`ODE_ocular_motor`](src/oculomotor/sim/simulator.py#L303)) is:

1. `sensory_model.read_outputs(sensory_x, params.sensory)` → `SensoryOutput`
   (delayed signals from the *current* sensory state).
2. (optional) add sensory noise to the retina fields.
3. `brain_model.step(..., blink_drive)` → `nerves, ec_vel, ec_pos, ec_verg, u_acc, u_pupil, u_lid`.
4. `plant_model.step(...)` per eye → `q_eye_{L,R}, w_eye_{L,R}` (+ `acc_plant.step`).
5. compute per-eye `defocus` from the current target distance and lens state.
6. `sensory_model.step(sensory_x, q_head, w_head, …, q_eye_L_eff, w_eye_L, …, ec_vel, ec_pos, ec_verg, params.sensory)` → `dsensory`
   — **must come after the plant**, because the retina is driven by the freshly
   computed eye velocity.

At that point you're re-implementing `ODE_ocular_motor`. Unless you specifically
need a custom front-end, it's far less error-prone to call `simulate()` (Pattern A)
or swap a single stage (Pattern B) than to rebuild the closed loop by hand.

---

## 6. Three ways to integrate — pick the lightest one that works

**Pattern A — use `simulate()`, drive it with stimulus arrays.**
You don't write a loop at all; you describe head/scene/target trajectories and let
the shipped solver run. This is the default and handles warmup, noise, per-eye
stereo, prisms, and lenses. Reach for a custom loop only when `simulate()` can't
express what you need.

**Pattern B — swap one stage, keep the rest.**
- Swap the **brain**: `simulator.set_brain_step(my_brain_step)` where your function
  matches `brain_model.step`'s signature
  (`fn(brain_state, sensory_out, brain_params, noise_acc, blink_drive) -> (dbrain, nerves, ec_vel, ec_pos, ec_verg, u_acc, u_pupil, u_lid)`).
- Swap the **plant**: any module implementing
  `step(x_p, motor_cmd, plant_params, decode_matrix) -> (dx_p, q_eye, w_eye)` is a
  drop-in (wire it in `simulator.py`).
- Swap the **retina/front-end**: build your own `RetinaOut` and run the brain+plant
  yourself (Pattern C) — this is what Section 4 shows.

**Pattern C — your own loop calling the three `step()`s.**
Full control (custom solver, RL env, real-time). Section 4 is the template. You own
the states, the timestep, and (if you close the loop) the evaluation order in
Section 5.

---

## 7. Conventions that will bite you if ignored

These all live in `CLAUDE.md`; the ones that matter most at the integration boundary:

- **Units:** angles in **degrees**, angular velocity in **deg/s**, linear in
  **m / m·s⁻¹ / m·s⁻²**, accommodation/defocus in **diopters**.
- **Angular vectors are `[yaw, pitch, roll]`, not `xyz`.** Use `ypr_to_xyz` /
  `xyz_to_ypr` (in `retina.py`) before/after any rotation-matrix math. `RetinaOut`
  velocities and `q_eye`/`w_eye` are all in ypr.
- **World frame is LEFT-HANDED:** x=right, y=up, z=forward (x × y = −z).
- **Gravity / `otolith`:** the brain wants the **GIA in head frame (m/s²)**, not
  zeros. Get the resting value from `read_outputs(rest_state, …, q_head=0, a_head=0)`
  rather than guessing the axis.
- **`nerves` order is `[L6 | R6]`;** decode with `M_PLANT_EYE_L` / `M_PLANT_EYE_R`.
- **`make_x0` / `rest_state` for every state** — never start from zeros.
- **Normalise `b_vs` to `(6,)` before `make_x0` / `brain_model.step`.** `default_params()`
  leaves it a scalar; `simulate()` broadcasts it once up front and so must you, or
  `make_x0` raises `'float' object is not subscriptable`.
- **Retinal signals are errors/slip relative to the eye, not world coordinates.**
  A constant `target_pos` is an open-loop stimulus; a realistic front-end recomputes
  it from the current `q_eye` / `w_eye` each step (closing the loop).
- **Keep all pathways live** unless a test specifically isolates one (e.g. don't
  zero `scene_visible` for a saccade unless you mean "in the dark").
- **`step()` functions are pure** — they return derivatives, they don't mutate. You
  do the integration and you hold the state.

---

## 8. Where the contracts are defined (source of truth)

| Contract | File · symbol |
|---|---|
| Brain step signature | [`brain_model.step`](src/oculomotor/models/brain_models/brain_model.py#L860) |
| Brain initial state | [`brain_model.make_x0`](src/oculomotor/models/brain_models/brain_model.py#L775) |
| Plant step signature | [`plant_model_first_order.step`](src/oculomotor/models/plant_models/plant_model_first_order.py#L81) |
| Muscle decode matrices | [`muscle_geometry.M_PLANT_EYE_L/R`](src/oculomotor/models/plant_models/muscle_geometry.py) |
| Sensory bundle / readout | [`sensory_model.SensoryOutput`](src/oculomotor/models/sensory_models/sensory_model.py#L162), [`read_outputs`](src/oculomotor/models/sensory_models/sensory_model.py#L189) |
| Retina output / step | [`retina.RetinaOut`](src/oculomotor/models/sensory_models/retina.py#L448), [`retina.step`](src/oculomotor/models/sensory_models/retina.py#L464) |
| Accommodation plant | [`accommodation_plant.step`](src/oculomotor/models/plant_models/accommodation_plant.py#L33) |
| Full reference loop | [`simulator.ODE_ocular_motor`](src/oculomotor/sim/simulator.py#L303) |
| Swap a brain | [`simulator.set_brain_step`](src/oculomotor/sim/simulator.py#L87) |

When in doubt, read `ODE_ocular_motor` — it is the canonical, working example of all
three stages wired together.
