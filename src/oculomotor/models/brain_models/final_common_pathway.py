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
import numpy as np

# Plant biomechanics this module's motor encoding is calibrated against (the eye's
# muscle rotation axes + the per-eye/nerve index vocabulary).
from oculomotor.models.plant_models.muscle_geometry import (
    M_NERVE_L, M_NERVE_R,                                    # (6,3) per-eye muscle rotation axes
    LR, MR, SR, IR, SO, IO,                                  # per-eye muscle indices
    LR_L, MR_L, SR_L, IR_L, SO_L, IO_L,                      # combined 12-D nerve-output indices
    LR_R, MR_R, SR_R, IR_R, SO_R, IO_R,
    N_NERVES,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Motor-nucleus & nerve connectivity  (the BRAIN's motor encoding — calibrated to
# the plant's muscle geometry above, but NOT plant biomechanics, so it lives here
# with the FCP, not in muscle_geometry).  Two-stage encode:
#     Stage 1  M_NUCLEUS   (14×6): [version, vergence] → 14 nucleus activations
#     Stage 2  M_NERVE_PROJ(12×14): nuclei → 12 nerve outputs
# (This block may later split into a nucleus/MN module + a nerve module.)
# ═══════════════════════════════════════════════════════════════════════════════

_M_NERVE_L_np = np.asarray(M_NERVE_L)   # numpy views for the static encode build
_M_NERVE_R_np = np.asarray(M_NERVE_R)

# ── Motor nucleus index constants (0–13) ──────────────────────────────────────
ABN_L, ABN_R       =  0,  1   # Abducens nucleus motoneurons (CN VI) → ipsilateral LR
CN4_L, CN4_R       =  2,  3   # Trochlear nucleus (CN IV) — SO, contralateral projection
CN3_MR_L, CN3_MR_R =  4,  5   # CN III — medial rectus subnucleus (vergence drive)
CN3_SR_L, CN3_SR_R =  6,  7   # CN III — superior rectus subnucleus
CN3_IR_L, CN3_IR_R =  8,  9   # CN III — inferior rectus subnucleus
CN3_IO_L, CN3_IO_R = 10, 11   # CN III — inferior oblique subnucleus
AIN_L, AIN_R       = 12, 13   # Abducens internuclear neurons → contralateral MR via MLF

N_NUCLEI = 14

# ── Stage 1 — M_NUCLEUS (14×6) [version, vergence] → nuclei ───────────────────
# Healthy round-trip target (M_NERVE_PROJ @ M_NUCLEUS = M_FULL):
#   M_FULL (12×6) = [[M_NERVE_L | +0.5·M_NERVE_L], [M_NERVE_R | −0.5·M_NERVE_R]]
_M_FULL_np = np.vstack([
    np.hstack([_M_NERVE_L_np,  0.5 * _M_NERVE_L_np]),   # left  nerves: version | vergence
    np.hstack([_M_NERVE_R_np, -0.5 * _M_NERVE_R_np]),   # right nerves: version | vergence
]).astype(np.float32)   # (12, 6)

_M_NUCLEUS_np = np.zeros((N_NUCLEI, 6), dtype=np.float32)
# ABN motoneurons: drive ipsilateral LR (version + vergence components).
_M_NUCLEUS_np[ABN_L] = np.concatenate([_M_NERVE_L_np[LR, :],  0.5 * _M_NERVE_L_np[LR, :]])
_M_NUCLEUS_np[ABN_R] = np.concatenate([_M_NERVE_R_np[LR, :], -0.5 * _M_NERVE_R_np[LR, :]])
# AIN: pure version drive (no vergence column). Same sign as ABN version output.
_M_NUCLEUS_np[AIN_L, :3] = _M_NERVE_L_np[LR, :]   # = [-1, 0, 0]
_M_NUCLEUS_np[AIN_R, :3] = _M_NERVE_R_np[LR, :]   # = [+1, 0, 0]
# CN3_MR: vergence-only drive (version arrives at MR via MLF from contralateral AIN).
_M_NUCLEUS_np[CN3_MR_L, 3:] = +0.5 * _M_NERVE_L_np[MR, :]   # → [+½, 0, 0]
_M_NUCLEUS_np[CN3_MR_R, 3:] = -0.5 * _M_NERVE_R_np[MR, :]   # → [+½, 0, 0]
# CN4 (contralateral SO).
_M_NUCLEUS_np[CN4_R] = np.concatenate([_M_NERVE_L_np[SO, :],  0.5 * _M_NERVE_L_np[SO, :]])
_M_NUCLEUS_np[CN4_L] = np.concatenate([_M_NERVE_R_np[SO, :], -0.5 * _M_NERVE_R_np[SO, :]])
# Remaining CN3 subdivisions (SR, IR, IO): direct ipsilateral, version + vergence.
for _nuc, _mus in ((CN3_SR_L, SR), (CN3_IR_L, IR), (CN3_IO_L, IO)):
    _M_NUCLEUS_np[_nuc] = np.concatenate([_M_NERVE_L_np[_mus, :],  0.5 * _M_NERVE_L_np[_mus, :]])
for _nuc, _mus in ((CN3_SR_R, SR), (CN3_IR_R, IR), (CN3_IO_R, IO)):
    _M_NUCLEUS_np[_nuc] = np.concatenate([_M_NERVE_R_np[_mus, :], -0.5 * _M_NERVE_R_np[_mus, :]])

# ── Stage 2 — M_NERVE_PROJ (12×14) nucleus → nerve (unit MLF gain; fcp.step
#    injects g_mlf_L/R into the AIN→MR entries at runtime) ─────────────────────
_M_NERVE_PROJ_np = np.zeros((N_NERVES, N_NUCLEI), dtype=np.float32)
_M_NERVE_PROJ_np[LR_L, ABN_L] = 1.0     # ABN → ipsilateral LR (CN VI, uncrossed)
_M_NERVE_PROJ_np[LR_R, ABN_R] = 1.0
_M_NERVE_PROJ_np[MR_R, AIN_L] = 1.0     # AIN → contralateral MR via MLF (right MLF)
_M_NERVE_PROJ_np[MR_L, AIN_R] = 1.0     # left MLF
_M_NERVE_PROJ_np[SO_R, CN4_L] = 1.0     # CN4 → contralateral SO (CN IV decussates)
_M_NERVE_PROJ_np[SO_L, CN4_R] = 1.0
_M_NERVE_PROJ_np[MR_L, CN3_MR_L] = 1.0  # CN3 vergence/version → ipsilateral (uncrossed)
_M_NERVE_PROJ_np[MR_R, CN3_MR_R] = 1.0
_M_NERVE_PROJ_np[SR_L, CN3_SR_L] = 1.0
_M_NERVE_PROJ_np[SR_R, CN3_SR_R] = 1.0
_M_NERVE_PROJ_np[IR_L, CN3_IR_L] = 1.0
_M_NERVE_PROJ_np[IR_R, CN3_IR_R] = 1.0
_M_NERVE_PROJ_np[IO_L, CN3_IO_L] = 1.0
_M_NERVE_PROJ_np[IO_R, CN3_IO_R] = 1.0

# Sanity check: healthy round-trip preserves the version+½·vergence command.
_check = _M_NERVE_PROJ_np @ _M_NUCLEUS_np
assert np.allclose(_check, _M_FULL_np, atol=1e-6), \
    f"FCP healthy round-trip broken: max diff {np.abs(_check - _M_FULL_np).max():.3e}"

M_NUCLEUS    = jnp.array(_M_NUCLEUS_np)    # (14, 6) brain → nuclei
M_NERVE_PROJ = jnp.array(_M_NERVE_PROJ_np) # (12,14) nuclei → nerves (g_mlf=1 default)

# ── Lesion-gain defaults + per-nucleus tonic baseline ─────────────────────────
# g_nucleus is (12,) — one gain per anatomical nucleus.  AIN_L/AIN_R share their
# gain with ABN_L/ABN_R (intermingled abducens populations); the FCP expands
# (12,) → (14,) at runtime.
N_GAINS_NUCLEUS    = 12
G_NUCLEUS_DEFAULT  = jnp.ones(N_GAINS_NUCLEUS, dtype=jnp.float32)   # healthy: all = 1
G_NERVE_DEFAULT    = jnp.ones(N_NERVES, dtype=jnp.float32)          # healthy: all = 1
R_BASELINE_DEFAULT = jnp.full(N_GAINS_NUCLEUS, 50.0, dtype=jnp.float32)  # tonic firing

def nucleus_gains_from_trunks(g_cn3_L, g_cn3_R, g_cn4_L, g_cn4_R, g_cn6_L, g_cn6_R):
    """Expand per-motor-NUCLEUS-complex integrity gains → the (12,) per-nucleus g_nucleus.

    Mirror of nerve_gains_from_trunks, one stage upstream: a nuclear lesion hits a
    whole nucleus complex on one side, so this is a pure gather (each subnucleus
    copies its complex's gain — no sums, no products):

        CN VI → ABN      CN III → MR, SR, IR, IO subnuclei      CN IV → CN4   (per side)

    The abducens (ABN) gain is shared with the co-located AIN downstream (the FCP
    expands (12,)→(14,) with AIN inheriting ABN), so a CN VI NUCLEAR lesion silences
    BOTH the ipsilateral LR motoneurons AND the MLF outflow to the contralateral MR
    → horizontal gaze palsy (vs an isolated LR palsy from a CN VI NERVE lesion).  The
    central caudal nucleus (levator) reads the CN III complex gain directly downstream
    (eyelid_nerves), so one g_nuc_cn3 covers the whole complex → bilateral partial ptosis.
    """
    g = jnp.ones(N_GAINS_NUCLEUS, dtype=jnp.float32)
    return (g.at[ABN_L].set(g_cn6_L)
             .at[ABN_R].set(g_cn6_R)
             .at[CN4_L].set(g_cn4_L)
             .at[CN4_R].set(g_cn4_R)
             .at[CN3_MR_L].set(g_cn3_L)
             .at[CN3_SR_L].set(g_cn3_L)
             .at[CN3_IR_L].set(g_cn3_L)
             .at[CN3_IO_L].set(g_cn3_L)
             .at[CN3_MR_R].set(g_cn3_R)
             .at[CN3_SR_R].set(g_cn3_R)
             .at[CN3_IR_R].set(g_cn3_R)
             .at[CN3_IO_R].set(g_cn3_R))


def nerve_gains_from_trunks(g_cn3_L, g_cn3_R, g_cn4_L, g_cn4_R, g_cn6_L, g_cn6_R):
    """Expand per-cranial-nerve-TRUNK integrity gains → the (12,) per-muscle g_nerve.

    Every extraocular muscle is served by exactly ONE cranial nerve, so this is a
    pure gather (each muscle-nerve copies its trunk's gain — no sums, no products):

        CN VI → LR        CN III → MR, SR, IR, IO        CN IV → SO      (per side)

    The other CN III effectors that travel with the trunk — levator (lid) and the
    pupilloconstrictor parasympathetics — read the SAME CN III integrity downstream
    (from the trunk gains directly), so one g_cn3 gain covers the whole trunk.
    """
    g = jnp.ones(N_NERVES, dtype=jnp.float32)
    return (g.at[LR_L].set(g_cn6_L)
             .at[LR_R].set(g_cn6_R)
             .at[MR_L].set(g_cn3_L)
             .at[SR_L].set(g_cn3_L)
             .at[IR_L].set(g_cn3_L)
             .at[IO_L].set(g_cn3_L)
             .at[MR_R].set(g_cn3_R)
             .at[SR_R].set(g_cn3_R)
             .at[IR_R].set(g_cn3_R)
             .at[IO_R].set(g_cn3_R)
             .at[SO_L].set(g_cn4_L)
             .at[SO_R].set(g_cn4_R))

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


# Pull-only co-contraction and the f-I ceiling are PER-MUSCLE nonlinearities (no
# antagonist coupling) — they live in read_activations.  See there.


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


class Nerves(NamedTuple):
    """The brain's complete efferent output — every peripheral nerve channel, all
    lesioned in this module (the final common pathway).  One field per plant target.

    Assembled by brain_model.step from the FCP lesion helpers (read_outputs,
    ciliary_nerve, iris_nerves, eyelid_nerves); it replaces the old positional
    MotorOut bag.  Each plant reads the field(s) it consumes.
    """
    extraocular:     jnp.ndarray   # (12,) extraocular nerve activations [L6|R6]   → eye plant
    ciliary:         jnp.ndarray   # (1,)  accommodation (ciliary) command (D)      → lens plant
    iris_sphincter:  jnp.ndarray   # (2,)  CN III constrictor drive (mm) [L,R]      → iris plants
    iris_dilator:    jnp.ndarray   # (2,)  sympathetic dilator drive (mm) [L,R]     → iris plants
    lid_levator:     jnp.ndarray   # (2,)  CN III levator opener tone [L,R]         → eyelid plants
    lid_muller:      jnp.ndarray   # (2,)  sympathetic Müller opener tone [L,R]     → eyelid plants
    lid_orbicularis: jnp.ndarray   # (2,)  CN VII orbicularis closer drive [L,R]    → eyelid plants


def zero_state():
    """All-zero state — useful as a NT-PyTree shape template."""
    return State(mn=jnp.zeros(14))


# The 14-slot MN vector is [12 muscle motoneurons | AIN_L, AIN_R].  Muscle MNs drive
# nerves; the 2 abducens internuclear neurons relay through the MLF and inherit the
# abducens (ABN) gain + baseline (intermingled populations).
_N_MN    = 12
_AIN_ABN = jnp.array([ABN_L, ABN_R])


def read_activations(state, brain_params):
    """Nucleus firing rate (≥0) — the cell-body output of each nucleus cell.

    state.mn is the nucleus MEMBRANE POTENTIAL (signed; sub-threshold off-direction).
    Two cell types:
      • Motoneurons (first 12): tonic baseline + version drive, f-I capped at NERVE_MAX,
        then the per-muscle pull-only fold → ≥0.  CN3_MR is special — its version tonic
        arrives via the MLF (already in v), so it isn't minted here.
      • Interneurons (AIN_L/R): plain rectified ≥0 firing (tonic + version), no fold;
        the MLF relays this to the contralateral MR in step().

    Lesion at THIS stage: g_nucleus (nuclear cell loss).  Nerve projection + conduction
    lesion (g_nerve) act on the OUTPUT in step(), not here.  The leak in step() reads
    the signed membrane directly, so this ≥0 readout never feeds back into the dynamics.
    """
    v     = state.mn
    g_nuc = brain_params.g_nucleus      # (12,) one gain per muscle MN (nucleus order)
    tonic = brain_params.r_baseline     # (12,) per-MN tonic baseline

    # ── Motoneurons — pull-only co-contraction fold ───────────────────────────────
    # f(x) = relu(x) + relu(x − 2·tonic): floored at 0 (a muscle can't push), linear in
    # the pull range, doubled past 2·tonic (where a symmetric antagonist floors, so the
    # agonist carries the full L−R differential alone).  Per-muscle, decoupled → a one-
    # sided lesion (INO) can't leak the antagonist's relaxation into the agonist.  CN3_MR
    # mints no version tonic (its conjugate tone arrives via the MLF, already in v); only
    # its fold threshold uses the baseline.
    tonic_add = tonic.at[CN3_MR_L].set(0.0).at[CN3_MR_R].set(0.0)
    drive_mn  = _smooth_clip_sym(v[:_N_MN] + g_nuc * tonic_add, _NERVE_MAX)
    mn        = jax.nn.relu(drive_mn) + jax.nn.relu(drive_mn - 2.0 * tonic)

    # ── Interneurons (AIN_L/R) — plain rectified ≥0 firing, no fold ────────────────
    # Drive no muscle, no antagonist → just rectify (tonic + version).  This is the
    # MR's whole conjugate drive, relayed by the MLF in step; for the off-direction the
    # AIN drops to 0 and the MR's tone goes with it.  Inherit the ABN gain + baseline.
    drive_ain = _smooth_clip_sym(v[_N_MN:] + g_nuc[_AIN_ABN] * tonic[_AIN_ABN], _NERVE_MAX)
    ain       = jax.nn.relu(drive_ain)

    return Activations(mn=jnp.concatenate([mn, ain]))


def step(state, premotor_activity, brain_params):
    """Single ODE step: premotor activity → motor-neuron membrane derivative.

        dx_mn  =  (premotor + mlf  −  v) / tau_mn

    Three clean stages — state → activation → nerve — split across two functions:
      • state.mn = the nucleus MEMBRANE potential (SIGNED — below threshold for the
        off-direction muscle).  The leak relaxes it toward the synaptic input
        (premotor + MLF).  Signed so the L−R differential can reach NERVE_MAX, not
        NERVE_MAX/2 — which is why step takes the state, not the ≥0 activations:
        those couldn't drive the leak.  step() computes ONLY this membrane derivative.
      • activation = read_activations(state) — the ≥0 nucleus FIRING rate (f-I
        ceiling NERVE_MAX + g_nucleus nuclear lesion, both cell-body properties).
      • nerve = read_outputs(state) = route @ activation, then the g_nerve conduction
        cap — the NERVE output and the AXON lesion.  Pure function of the membrane
        state; healthy the nerve == the routed firing (g_nerve / g_mlf < 1 change it).

    Args:
        state:              fcp.State  MN membrane states (14,)
        premotor_activity:  (6,)       [version (3,) | vergence (3,)] — NI +
                             saccade burst + pursuit + vergence integrator output
        brain_params:       BrainParams (g_nucleus, g_nerve, g_mlf_L/R, tau_mn)

    Returns:
        dstate: fcp.State  state derivative (membrane integrator).  The nerve output
                            comes from read_outputs(state, brain_params), not here.
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

    # MLF: the AIN's ≥0 FIRING (tonic + version) crosses to the contralateral CN3_MR.
    # The tonically-firing abducens internuclear cells ARE the MR's conjugate resting
    # tone + saccade burst; the MR mints no version tonic of its own (read_activations
    # zeroes it).  So when the AIN falls silent for the off-direction (relu(T+ver)→0),
    # the MR's tone vanishes with it → the MR relaxes to 0 — no separate inhibition.
    # g_mlf caps the tract (the INO lesion): it removes tone AND burst together →
    # adduction palsy + a resting exotropia; convergence is spared (CN3_MR direct).
    #
    # mlf_lead — a LOCAL phase-lead at the AIN.  The adducting MR rides a 2-stage path
    # (AIN tau_mn → MLF → CN3_MR tau_mn) vs the abducting LR's 1-stage, so a conjugate
    # command lands disconjugately.  Since premotor[AIN] = v[AIN] + tau_mn·d(v[AIN])/dt,
    # the blend is a PHASIC-TONIC mix (tonic membrane + phasic derivative) that pre-pays
    # one tau_mn of the AIN's lag → the MR tracks the 1-stage LR.  mlf_lead ∈ [0,1] sets
    # how phasic.  (Falsifiable: AINs should lead ABNs by ~mlf_lead·tau_mn in saccades.)
    # FUTURE: with a 2nd-order plant this plant-lag compensation moves to a forward-model
    # / plant-inverse stage (cf. cerebellum.py EC) and the lead may migrate there.
    a = brain_params.mlf_lead
    ain_R = (1.0 - a) * v[AIN_R] + a * premotor[AIN_R]   # right AIN drive (phase-led), version only
    ain_L = (1.0 - a) * v[AIN_L] + a * premotor[AIN_L]   # left  AIN drive
    T_ain_L = g_nuc12[ABN_L] * brain_params.r_baseline[ABN_L]   # AIN tonic = abducens baseline (g_nuc-scaled)
    T_ain_R = g_nuc12[ABN_R] * brain_params.r_baseline[ABN_R]
    mlf = jnp.zeros(N_STATES) \
        .at[CN3_MR_L].set(_smooth_clip(jax.nn.relu(ain_R + T_ain_R), brain_params.g_mlf_L * _NERVE_MAX)) \
        .at[CN3_MR_R].set(_smooth_clip(jax.nn.relu(ain_L + T_ain_L), brain_params.g_mlf_R * _NERVE_MAX))

    # MN dynamics: the membrane relaxes toward its synaptic input (premotor + MLF)
    # with TC tau_mn.  The leak feedback is the SIGNED membrane v (not the ≥0
    # firing rate), so the firing can be read out ≥0 without perturbing dynamics.
    dx_mn = (premotor + mlf - v) / brain_params.tau_mn

    return State(mn=dx_mn)


def read_outputs(state, brain_params):
    """Nerve output (12,) — axonal firing rates to the extraocular muscles.

        [LR_L, MR_L, SR_L, IR_L, SO_L, IO_L, LR_R, MR_R, SR_R, IR_R, SO_R, IO_R]

    A pure function of the nucleus MEMBRANE state: route the ≥0 nucleus firing
    (read_activations) to the 12 muscles via _ROUTE (a pure selection — AIN→MR is
    delivered by the MLF, already integrated into the CN3_MR membrane, so its
    firing carries it), then apply the cranial-nerve conduction lesion.  Needs no
    premotor input.  g_nerve is the AXON lesion: g_nerve→0 silences the muscle
    (denervated, no force); g_nerve<1 frequency-selectively caps the burst →
    limited-motility ophthalmoplegia.
    """
    activation = read_activations(state, brain_params).mn                             # ≥0 nucleus firing (14,)
    return _smooth_clip(_ROUTE @ activation, brain_params.g_nerve * _NERVE_MAX)        # project + axon lesion


def iris_nerves(sphincter_raw, dilator_raw, brain_params):
    """Apply the peripheral iris-nerve lesions to the raw drives (pupil.command).

    The iris is an antagonist pair whose two nerves are lesioned here, exactly like
    the extraocular nerves above — each in TWO stages (nucleus × nerve):
        sphincter (constrictor) — CN III parasympathetic. NUCLEUS = Edinger-Westphal
                    (g_nuc_ew: 0 → internal ophthalmoplegia, fixed pupil, somatic
                    spared); NERVE = the CN III trunk (g_cn3: blown pupil with a nerve
                    palsy). A dorsal-midbrain (nuclear-somatic) lesion spares EW → the
                    pupil stays reactive.
        dilator                 — sympathetic → gated by g_ocular_symp (Horner miosis).
    Per eye [L, R].  Returns (sphincter, dilator) nerve activations; the iris plant
    does the push-pull decode (diam = pupil_min + dilator − sphincter).
    """
    bp = brain_params
    sphincter = sphincter_raw * jnp.array([bp.g_nuc_ew_L * bp.g_cn3_L,
                                           bp.g_nuc_ew_R * bp.g_cn3_R])
    dilator   = dilator_raw   * jnp.array([bp.g_ocular_symp_L, bp.g_ocular_symp_R])
    return sphincter, dilator


def eyelid_nerves(levator, muller, orbicularis, brain_params):
    """Apply the peripheral eyelid-nerve lesions to the raw drives (eyelid.command).

    Three antagonist muscles, each lesioned here like every other nerve:
        levator     (CN III somatic) — TWO stages: the NUCLEUS is the central caudal
                    nucleus (g_nuc_ccn), which projects to BOTH levators (ipsi +
                    contra, → bilateral partial ptosis on a nuclear lesion); the NERVE
                    follows the CN III trunk (g_cn3, → unilateral complete ptosis on a
                    nerve palsy).  Mixing fraction: eyelid_levator_contra_frac.
        muller      (sympathetic)    — gated by g_ocular_symp (Horner mild ptosis).
        orbicularis (CN VII facial)  — gated by g_cn7 (lagophthalmos / Bell's palsy).
    Per eye [L, R].  Returns (levator, muller, orbicularis) tone/drives; the eyelid
    plant decodes them into a lid closure.
    """
    bp = brain_params
    ccn_L, ccn_R = bp.g_nuc_ccn_L, bp.g_nuc_ccn_R   # central caudal nucleus integrity
    c = bp.eyelid_levator_contra_frac   # fraction of each levator's nuclear drive from CONTRA
    levator = levator * jnp.array([bp.g_cn3_L * ((1.0 - c) * ccn_L + c * ccn_R),
                                   bp.g_cn3_R * ((1.0 - c) * ccn_R + c * ccn_L)])
    muller      = muller      * jnp.array([bp.g_ocular_symp_L, bp.g_ocular_symp_R])
    orbicularis = orbicularis * jnp.array([bp.g_cn7_L, bp.g_cn7_R])
    return levator, muller, orbicularis


def ciliary_nerve(u_acc, brain_params):
    """Apply the CN III (ciliary / accommodation) lesions to the accommodation command.

    The ciliary muscle is CN III parasympathetic (like the iris sphincter), lesioned
    in TWO stages: NUCLEUS = Edinger-Westphal (g_nuc_ew — shared with the sphincter,
    so an EW lesion gives internal ophthalmoplegia = fixed pupil + cycloplegia) and
    NERVE = the CN III trunk (g_cn3).  The lens is a single BINOCULAR state, so gate
    by the BETTER eye's combined EW-nucleus × CN III-nerve integrity: bilateral loss →
    cycloplegia; unilateral → preserved (the intact eye drives the shared near
    response).  (Monocular cycloplegia would need per-eye lens states.)  Completes the
    CN III triad → ophthalmoplegia + ptosis + blown pupil + cycloplegia.
    """
    bp = brain_params
    return u_acc * jnp.maximum(bp.g_nuc_ew_L * bp.g_cn3_L, bp.g_nuc_ew_R * bp.g_cn3_R)


def rest_state(premotor_activity, brain_params):
    """Steady-state MN membrane for a given resting premotor command.

    At rest there is no version drive, so each membrane equals its premotor input
    — except CN3_MR, which also carries the MLF-delivered AIN tonic (at rest,
    version=0 → relu(ain + T) = T).  Used to initialise x_mn so the model starts on
    the slow manifold (skips a ~5·tau_mn warmup transient at t=0).
    """
    g_nuc12 = brain_params.g_nucleus
    g_nuc14 = jnp.concatenate([g_nuc12, g_nuc12[:2]])
    membrane = g_nuc14 * (M_NUCLEUS @ premotor_activity)
    T_ain_L = g_nuc12[ABN_L] * brain_params.r_baseline[ABN_L]
    T_ain_R = g_nuc12[ABN_R] * brain_params.r_baseline[ABN_R]
    membrane = membrane.at[CN3_MR_L].add(_smooth_clip(T_ain_R, brain_params.g_mlf_L * _NERVE_MAX)) \
                       .at[CN3_MR_R].add(_smooth_clip(T_ain_L, brain_params.g_mlf_R * _NERVE_MAX))
    return _smooth_clip(membrane, _NERVE_MAX)
