"""Final common pathway — version+vergence  →  motor neurons  →  output nerves.

Anatomically faithful chain:

    version_vergence (6,)
        ↓ M_NUCLEUS                  (signed nucleus drives)
        ↓ × g_nucleus                (nuclear lesion: multiplicative cell-count gain)
        ↓ MLF tract (axon)           (AIN MN → contralateral CN3_MR MN; cap = g_mlf · NERVE_MAX)
        ↓ M_NERVE_PROJ × 2           (nucleus → MN target firing rate)
        ↓ MN intrinsic f-I curve     (biophysical max NERVE_MAX, not a lesion knob)
    motor neurons (12 dynamic states, x_mn)
        ↓ tau_mn LP dynamics         (~5 ms membrane TC)
        ↓ axonal conduction clip     (cap = g_nerve · NERVE_MAX, frequency-selective)
    output nerves (12,) → extraocular muscles

The MLF is a motor-neuron-to-motor-neuron connection: abducens internuclear
neurons (AIN) fire as ordinary MNs (intrinsic NERVE_MAX), their axons enter
the contralateral MLF tract (conduction-capped), and they synapse onto CN3_MR
motoneurons in the oculomotor nucleus.  CN3_MR motoneurons sum vergence drive
(direct from supraoculomotor area) with MLF input → fire → drive MR muscle.

Lesion semantics:

    g_nucleus (12,)  Multiplicative gain on signed nucleus drive.
                     Models cell loss in a nucleus (% of cells surviving).
                     Burst AND tonic both attenuated proportionally.
                     ABN gain shared with co-located AIN (intermingled populations).

    g_mlf_L/R (2,)   Conduction cap on the MLF axon tract (axon-level clip).
                     Models demyelination / conduction block in the MLF.
                     Frequency-selective: tonic AIN drive (small) gets through;
                     saccadic burst (large) is capped → slow adducting saccades.
                     Vergence preserved (delivered via CN3_MR direct, bypasses MLF).

    g_nerve (12,)    Conduction cap on the cranial-nerve axon (axon-level clip).
                     Models nerve demyelination / fascicular lesion.
                     Frequency-selective: fixation hold preserved, burst clipped
                     → limited-motility ophthalmoplegia.  At g_nerve=0 the axon
                     transmits nothing → complete muscle paralysis.

Nerve / MN ordering (12,):  [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                              LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]

Parameters (in BrainParams):
    g_nucleus (12,)  nuclear gains [0, 1]. Default: all ones.
    g_mlf_L/R        MLF axon conduction cap fractions [0, 1]. Default: 1.
    g_nerve   (12,)  cranial-nerve axon conduction cap fractions [0, 1]. Default: all ones.
    tau_mn           MN membrane LP TC (s). Default 0.005.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

from oculomotor.models.plant_models.muscle_geometry import (
    M_NUCLEUS, M_NERVE_PROJ,
    G_NUCLEUS_DEFAULT, G_NERVE_DEFAULT,
    AIN_L, AIN_R, ABN_L, ABN_R, CN4_L, CN4_R,
    CN3_MR_L, CN3_MR_R, CN3_SR_L, CN3_SR_R, CN3_IR_L, CN3_IR_R, CN3_IO_L, CN3_IO_R,
    MR_L, MR_R,   # nerve-order rows used only to strip the AIN→MR entries from the route
)

__all__ = ['G_NUCLEUS_DEFAULT', 'G_NERVE_DEFAULT', 'N_STATES', 'step', 'rest_state',
           'Activations', 'read_activations']

# State count: 14 motor neurons in nucleus order — 12 muscle-MNs that project
# via cranial nerves to extraocular muscles, plus 2 abducens internuclear
# neurons (AIN_L, AIN_R) whose axons enter the MLF tract and synapse onto
# contralateral CN3_MR motoneurons (no cranial-nerve output).
N_STATES = 14

# Biophysical maximum firing rate (deg/s equivalent), used for the premotor
# f-I curve, MLF axon conduction cap, and cranial-nerve axon conduction cap.
# Set ABOVE the healthy peak agonist-MN drive so a healthy big saccade is not
# throttled by its own cell-body f-I ceiling.  The ×2 reciprocal-compensation
# factor in the premotor encode means a big burst (motor_cmd ≈ 130–160 deg/s
# mid-burst on a 40–60° saccade) pushes the agonist MN toward 2·motor_cmd ≈
# 270–320; a 250-cap clipped that and pinned the peak eye velocity (a 40°,
# 50° and 60° saccade all topped out at the same ~607 deg/s) and added an
# undershoot/glissade.  350 leaves headroom up to ~60–65° while staying low
# enough that MLF / cranial-nerve conduction-cap lesions in the clinically
# meaningful range (g_mlf, g_nerve in [0.3, ~0.85] → cap ≤ ~300 < the ~320
# healthy adducting-MN drive) still engage the clip → graded INO / partial-
# palsy slowing.
_NERVE_MAX = 350.0

def _smooth_clip(z, g_max):
    """Smooth one-sided ceiling clip at g_max — no rectification floor.

    output = z − softplus(z − g_max):
        - z << g_max:  ≈ z        (linear regime, passes negatives through)
        - z >> g_max:  ≈ g_max    (saturation)
        - g_max → 0:   ≈ min(z, 0)  (lesion attenuates positive drive only)
    Originally this was a two-sided clip into [0, g_max] (rectification floor
    at 0 from muscles only being able to pull, not push) — but the floor
    introduced asymmetric dynamics that no linear pulse-step compensation
    could invert, leaking into post-saccadic spurious pursuit drive.  Removing
    the floor lets the antagonist motoneurons fire negatively (i.e., the
    muscles can push as well as pull, in this model abstraction), which makes
    the effective plant a clean cascade of two LPs that linear compensation
    can invert cleanly.  See web/plant_compensation.md.
    """
    return z - jax.nn.softplus(z - g_max)


def _smooth_clip_sym(z, g_max):
    """Smooth TWO-SIDED clip into [-g_max, +g_max].

    Same top as `_smooth_clip` (pull saturates at +g_max) plus a mirror floor at
    -g_max.  Used for the *lesion-gated* cranial-nerve output (g_max = g_nerve *
    NERVE_MAX): a complete conduction block (g_max -> 0) then silences the muscle
    in BOTH directions, instead of letting it "push" (fire negative) once the
    positive pull is clipped away — a denervated muscle exerts no force.

        output = z - softplus(z - g_max) + softplus(-z - g_max)
            |z| << g_max :  ~ z       (linear; healthy push-pull preserved)
            z  >>  g_max :  ~ +g_max  (pull saturates)
            z  << -g_max :  ~ -g_max  (push saturates)
            g_max -> 0   :  ~ 0       (complete block: silent both ways)

    This keeps g_nerve a PURE clip (conduction block) distinct from g_nucleus,
    the linear gain (cell loss) applied to the drive before rectification.  For a
    healthy nerve (g_max = NERVE_MAX) the floor term softplus(-z - NM) is ~0 over
    the whole physiological drive range (max push ~-78 << -NM=-350), so it is a
    no-op vs the one-sided clip; it only bites once a lesion brings the cap down
    toward the operating range.
    """
    return z - jax.nn.softplus(z - g_max) + jax.nn.softplus(-z - g_max)


# ── Pull-only (co-contraction): the nucleus fires ≥0 ──────────────────────────
# Lift each reciprocal antagonist MN pair by a COMMON-MODE amount c = relu(-min)
# so neither motoneuron fires negative.  The pair drives muscles whose rotation
# axes are exact negatives (LR↔MR yaw, SR↔IR vertical, SO↔IO torsion), so the lift
# is a plant-null direction → it cancels in the (LR-MR)/2 decode → EXACTLY
# transparent to the eye, while the antagonist keeps physiologically pulling and
# drops out only near its off-extreme.  Lifting the pair TOGETHER (not rectifying
# each MN independently) preserves the differential — independent rectification
# loses velocity and adds a glissade.  Pairs are in NUCLEUS order; SO is driven by
# the contralateral trochlear MN, hence the crossed CN4↔CN3_IO entries.
_RECIP_PAIRS = ((ABN_L, CN3_MR_L), (CN3_SR_L, CN3_IR_L), (CN4_R, CN3_IO_L),
                (ABN_R, CN3_MR_R), (CN3_SR_R, CN3_IR_R), (CN4_L, CN3_IO_R))


def _pullonly(firing):
    """Lift each reciprocal MN pair by a common-mode amount so neither fires negative."""
    for ia, ib in _RECIP_PAIRS:
        c = jax.nn.relu(-jnp.minimum(firing[ia], firing[ib]))   # common-mode lift (plant-null)
        firing = firing.at[ia].add(c).at[ib].add(c)
    return firing


# Nucleus → nerve routing (12×14).  The AIN→MR entries are stripped: the AIN
# reaches MR through the MLF (a membrane current in step), not this direct route,
# so the stripped matrix is a pure selection — each nerve ← exactly one MN.
_ROUTE = M_NERVE_PROJ.at[MR_L, AIN_R].set(0.0).at[MR_R, AIN_L].set(0.0)


# ── State + registries ────────────────────────────────────────────────────────

class State(NamedTuple):
    """FCP state — 14 motor neuron membrane potentials.

    `mn` is a MEMBRANE POTENTIAL (signed), integrated by the LP dynamics in
    step(); it sits below threshold for the off-direction muscle.  It is NOT the
    firing rate — `read_activations` turns it into the ≥0 firing rate.

    AIN_L and AIN_R are abducens internuclear neurons whose axons enter the MLF
    and synapse on contralateral CN3_MR motoneurons (no extraocular muscle output).
    """
    mn: jnp.ndarray   # (14,) [LR_L,LR_R,CN4_L,CN4_R,MR_L,MR_R,SR_L,SR_R,IR_L,IR_R,IO_L,IO_R,AIN_L,AIN_R]


class Activations(NamedTuple):
    """Nucleus firing rate (≥0) — 14 motoneurons in nucleus order.

    Muscle MNs are ≥0 (pull-only co-contraction); the 2 AIN are internuclear (no
    muscle) so they stay signed — the MLF relays their sub-threshold drive.
    """
    mn: jnp.ndarray   # (14,) firing rates, nucleus order; muscle MNs ≥0, AIN_L/R signed


def zero_state():
    """All-zero state — useful as a NT-PyTree shape template."""
    return State(mn=jnp.zeros(14))


def read_activations(state, brain_params):
    """Nucleus firing rate (≥0) — what each motoneuron fires.

    state.mn is the nucleus MEMBRANE POTENTIAL (signed; sub-threshold for the
    off-direction muscle).  The firing rate is the cell-body output, per motoneuron:
    add the tonic baseline, cap at the f-I ceiling NERVE_MAX (the cell-body max
    rate), then pull-only on the reciprocal MN pairs → ≥0.  (The active drive is
    already in `v`, g_nucleus-scaled by step's premotor; the tonic is scaled here.)

    Lesion at THIS stage: g_nucleus — nuclear cell loss, scales the firing.  The two
    AIN are internuclear (no muscle) → their signed drive is kept; the MLF relays it
    in step.  Projection to nerve and the conduction lesion (g_nerve) act on the
    OUTPUT in step(), NOT here.

    The leak in step() reads the signed membrane directly, so this ≥0 readout never
    feeds back into the dynamics.
    """
    v       = state.mn
    g_nuc14 = jnp.concatenate([brain_params.g_nucleus, brain_params.g_nucleus[:2]])
    tonic14 = jnp.concatenate([brain_params.r_baseline, jnp.zeros(2)])
    firing  = _smooth_clip_sym(v + g_nuc14 * tonic14, _NERVE_MAX)   # signed drive, f-I capped
    return Activations(mn=_pullonly(firing))                        # muscle MNs → ≥0; AIN stays signed


def step(state, premotor_activity, brain_params):
    """Single ODE step: premotor activity → motor neurons → output nerve.

        dx_mn  =  (premotor + mlf  −  v) / tau_mn

    Three clean stages — state → activation → nerve:
      • state.mn = the nucleus MEMBRANE potential (SIGNED — below threshold for the
        off-direction muscle).  The leak relaxes it toward the synaptic input
        (premotor + MLF).  Signed so the L−R differential can reach NERVE_MAX, not
        NERVE_MAX/2 — which is why step takes the state, not the ≥0 activations:
        those couldn't drive the leak.
      • activation = read_activations(state) — the ≥0 nucleus FIRING rate (f-I
        ceiling NERVE_MAX + g_nucleus nuclear lesion, both cell-body properties).
      • nerve (here) = route @ activation, then the g_nerve conduction cap — the
        NERVE output and the AXON lesion.  Healthy the nerve == the routed firing;
        g_nerve / g_mlf < 1 are the only things that change it.

    Args:
        state:              fcp.State  MN membrane states (14,)
        premotor_activity:  (6,)       [version (3,) | vergence (3,)] — NI +
                             saccade burst + pursuit + vergence integrator output
        brain_params:       BrainParams (g_nucleus, g_nerve, g_mlf_L/R, tau_mn)

    Returns:
        dstate: fcp.State  state derivative (membrane integrator)
        nerves: (12,)      axonal firing rates to extraocular muscles
                            [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,
                             LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]
    """
    v = state.mn                             # membrane potential (signed); f-I ceiling lives in read_activations

    # Premotor synaptic input: linear projection of the upstream brain command
    # via M_NUCLEUS, scaled by g_nucleus (cell-loss gain; AIN_L/R inherit
    # ABN_L/R gain — intermingled populations).  No ×2 here: the antagonist
    # MEMBRANE goes negative (push-pull symmetric in the signed membrane), so the
    # L−R differential carries the command and `M_PLANT_EYE @ nerves` round-trips
    # to motor_cmd without a reciprocal-compensation factor (the pull-only lift
    # moves the differential onto the agonist in read_activations).  No ceiling
    # here — the synaptic drive (~motor_cmd) stays well under NERVE_MAX; the f-I
    # ceiling is applied once, in read_activations (the firing rate).
    g_nuc12  = brain_params.g_nucleus
    g_nuc14  = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    premotor = g_nuc14 * (M_NUCLEUS @ premotor_activity)                          # (14,) synaptic input

    # MLF: AIN firing feeds contralateral CN3_MR through the MLF tract; g_mlf_L/R is
    # the axon conduction cap (frequency-selective — the INO lesion).
    #
    # mlf_lead — a LOCAL phase-lead at the abducens internuclear neuron.  The
    # adducting MR rides a 2-stage path (AIN tau_mn → MLF → CN3_MR tau_mn) vs the
    # abducting LR's 1-stage, so a conjugate command lands disconjugately (post-
    # saccadic vergence transient).  Since premotor[AIN] = v[AIN] + tau_mn·d(v[AIN])/dt,
    # the blend below is a PHASIC-TONIC mix — tonic membrane v plus a phasic
    # derivative — that pre-pays one tau_mn of the AIN's own lag → the MR tracks the
    # 1-stage LR.  Biologically: the AIN fires with a phase lead vs the ABN (both get
    # the same input, the AIN is more phasic); mlf_lead ∈ [0,1] sets how phasic.  Both
    # terms are local to the abducens — which is why this lives in the FCP.
    # (Falsifiable: AINs should lead ABNs by ~mlf_lead·tau_mn during saccades.)
    #
    # FUTURE: this is plant-lag compensation done locally.  When the plant becomes
    # 2nd-order (biomechanical), the brain's plant compensation moves to a dedicated
    # forward-model / plant-inverse stage (cf. cerebellum.py EC) — and this AIN lead
    # may migrate there with it.
    a = brain_params.mlf_lead
    ain_R_mlf = (1.0 - a) * v[AIN_R] + a * premotor[AIN_R]
    ain_L_mlf = (1.0 - a) * v[AIN_L] + a * premotor[AIN_L]
    mlf = jnp.zeros(N_STATES) \
        .at[CN3_MR_L].set(_smooth_clip(ain_R_mlf, brain_params.g_mlf_L * _NERVE_MAX)) \
        .at[CN3_MR_R].set(_smooth_clip(ain_L_mlf, brain_params.g_mlf_R * _NERVE_MAX))

    # MN dynamics: the membrane relaxes toward its synaptic input (premotor + MLF)
    # with TC tau_mn.  The leak feedback is the SIGNED membrane v (not the ≥0
    # firing rate), so the firing can be read out ≥0 without perturbing dynamics.
    dx_mn = (premotor + mlf - v) / brain_params.tau_mn

    # Nerve output: route the nucleus firing to the muscles (_ROUTE is a pure
    # selection — AIN→MR is delivered via the MLF above, already in the CN3_MR
    # firing), then apply the cranial-nerve conduction lesion.  g_nerve is the AXON
    # lesion: g_nerve→0 silences the muscle (denervated, no force); g_nerve<1
    # frequency-selectively caps the burst → limited-motility ophthalmoplegia.
    activation = read_activations(state, brain_params).mn                             # ≥0 nucleus firing (14,)
    nerve      = _smooth_clip(_ROUTE @ activation, brain_params.g_nerve * _NERVE_MAX)  # project + axon lesion
    return State(mn=dx_mn), nerve


# ── Legacy flat-array adapters (deleted once brain_model migrates to BrainState) ─

def from_array(x_mn):
    """(14,) flat array → fcp.State."""
    return State(mn=x_mn)


def to_array(state):
    """fcp.State → (14,) flat array."""
    return state.mn


def rest_state(premotor_activity, brain_params):
    """Steady-state MN firing rates for a given resting premotor command.

    At rest there is no version drive → AIN MNs silent → MLF input is zero,
    so steady state equals the rectified, capped premotor input.  Used to
    initialise x_mn so the model starts on the slow manifold (skips a
    ~5·tau_mn warmup transient at t=0).
    """
    g_nuc12 = brain_params.g_nucleus
    g_nuc14 = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    return _smooth_clip(g_nuc14 * (M_NUCLEUS @ premotor_activity),
                        _NERVE_MAX)
