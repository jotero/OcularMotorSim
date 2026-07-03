"""Eyelid data generator.

Server-side machinery to generate/model eyelid motion so the eyelid becomes a
first-class *data* channel in the simulation output (payload / CSV / plots),
rather than being animated only on the client.

For now this is a POST-HOC generator (not a physiological blink/levator model):
given the eye-position trajectory it reproduces what ``web/avatar.js`` currently
does —

    * spontaneous random blinks (a short 0→1→0 pulse every few seconds,
      conjugate across the two eyes), and
    * an upper lid that follows vertical gaze (downgaze lowers the lid).

It is intentionally a drop-in mimic of the client behaviour; a proper model
(brainstem blink generator + levator/orbicularis plant, lid-saccade coupling,
reflex + spontaneous + voluntary blinks) can replace ``generate`` later behind
the same interface.

Output convention: eyelid CLOSURE in [0, 1] per eye — 0 = eye fully open,
1 = fully closed (mid-blink).  Blinks are conjugate; the gaze-follow term is
per-eye from each eye's pitch.
"""

import numpy as np

# Defaults mirror web/avatar.js (updateBlink + applyFrame eyelid block):
#   blink every ~2–5.5 s, 150 ms duration; downgaze lid-follow gain 0.4 over 70°.
_BLINK_MIN_S   = 2.0
_BLINK_MAX_S   = 5.5
_BLINK_DUR_S   = 0.15
_DOWNGAZE_GAIN = 0.4
_DOWNGAZE_DEG  = 70.0


def blink_train(t, seed=0, blink_min_s=_BLINK_MIN_S, blink_max_s=_BLINK_MAX_S,
                blink_dur_s=_BLINK_DUR_S):
    """Conjugate spontaneous-blink signal over ``t`` — (T,) in [0, 1].

    Each blink is a smooth raised-sine pulse (0→1→0) of ``blink_dur_s``; inter-
    blink intervals are uniform in [blink_min_s, blink_max_s]. Deterministic
    given ``seed`` (the client uses the wall clock, so timing differs — only the
    statistics are mimicked).
    """
    t = np.asarray(t, dtype=float)
    T = len(t)
    blink = np.zeros(T)
    if T == 0:
        return blink
    rng = np.random.default_rng(seed)
    tb = float(t[0]) + rng.uniform(blink_min_s, blink_max_s)
    while tb < t[-1]:
        mask = (t >= tb) & (t < tb + blink_dur_s)
        if mask.any():
            phase = (t[mask] - tb) / blink_dur_s
            blink[mask] = np.maximum(blink[mask], np.sin(np.pi * phase))
        tb += blink_dur_s + rng.uniform(blink_min_s, blink_max_s)
    return blink


def generate(t, pitch_L, pitch_R, seed=0,
             downgaze_gain=_DOWNGAZE_GAIN, downgaze_deg=_DOWNGAZE_DEG, **blink_kw):
    """Per-eye eyelid closure (T,) each — random blinks + downgaze lid-follow.

    Args:
        t:        (T,)  time array (s)
        pitch_L:  (T,)  left  eye pitch (deg, + = up); downgaze (< 0) lowers the lid
        pitch_R:  (T,)  right eye pitch (deg, + = up)
        seed:     int   blink-train RNG seed (deterministic)
        downgaze_gain / downgaze_deg: lid-follow slope (closure per deg of downgaze)
        **blink_kw: forwarded to blink_train (blink_min_s / blink_max_s / blink_dur_s)

    Returns:
        eyelid_L, eyelid_R:  (T,) each, closure in [0, 1] (0 = open, 1 = closed)
    """
    blink = blink_train(t, seed=seed, **blink_kw)

    def _closure(pitch):
        down = np.maximum(0.0, -np.asarray(pitch, dtype=float)) / downgaze_deg * downgaze_gain
        return np.clip(np.maximum(blink, down), 0.0, 1.0)

    return _closure(pitch_L), _closure(pitch_R)
