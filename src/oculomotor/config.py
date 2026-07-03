"""Central integration / solver constants — single source of truth.

This module holds solver-level numerical constants that must stay *consistent
across modules*. It is deliberately tiny and dependency-free.

What belongs here:
    - The integration step size and warmup window (solver settings).
    - Constants that more than one module must agree on and that are NOT
      learnable model parameters.

What does NOT belong here:
    - Model *parameters* (physiology, gains, time constants). Those live in the
      SensoryParams / BrainParams / PlantParams NamedTuples and are varied via
      with_sensory / with_brain / with_plant. Putting them here would bypass the
      params machinery (fitting, lesions, per-run overrides).

DT_SOLVE
    Heun fixed integration step (s). Heun is explicit, so the hard stability
    ceiling is  dt < 2 * tau_min,  where tau_min is the stiffest *integrated*
    state in the model. The NI lead-filter pole `tau_fast`
    (neural_integrator.step) is defined to TRACK this value, so that filter
    always sits at dt/tau_fast = 1 regardless of DT_SOLVE. Other fast
    saccade-generator states are fixed physiological TCs and do NOT scale:
        tau_sac  ~ 1 ms   (saccade latch)
        tau_bn   ~ 3 ms   (EBN/IBN state; needs tau_bn > 2.5*dt)
    so raising DT_SOLVE past ~0.002 s can destabilise the burst generator even
    though tau_fast is handled. Change DT_SOLVE here (not scattered literals) so
    the coupling stays in one place.

WARMUP_S
    Settling period (s) prepended before t=0 so fast states reach steady state
    before the plotted window. Stripped from all returned arrays.
"""

DT_SOLVE: float = 0.001
WARMUP_S: float = 3.0
