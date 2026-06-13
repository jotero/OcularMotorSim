"""Gradient-based parameter fitting.

⚠️ OBSOLETE (as of the 2026-06 reorg) — do not use without rewriting first.

These modules are early VOR-only scaffolding that predate two major refactors:
  * Params became a nested `Params` NamedTuple (sensory/brain/plant); `loss.py` still
    builds a flat `theta` dict (tau_i/tau_p/tau_vs/K_vs only — 4 params, VOR cascade).
  * `simulate()` gained its current multi-arg signature; `loss.py` calls the old
    `simulate(theta, t, head_vel)`.
So `loss.mse_loss` / `optimize.fit` will NOT run against the current model.

Fitting is "future work" (see CLAUDE.md). When that work begins, rewrite `loss.py` and
`optimize.py` against the current `Params` NamedTuple + `simulate()` API (and the deleted
`sim/synthetic.py` synthetic-data generator). The only consumer today is the quarantined
`scratch/run_recovery.py`. Kept for reference, not deleted, so the structure/intent survives.
"""
