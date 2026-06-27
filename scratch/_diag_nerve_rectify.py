"""Prove the lesion-aware nerve rectification is an EXACT no-op for healthy
muscles (g_nerve=1) and only attenuates the push branch for lesions (g_nerve<1).

current:  nerves = _smooth_clip(z, g_nerve*NM)                    # push passes at any g
proposed: nerves = _smooth_clip(z, g_nerve*NM) + (g_nerve-1)*(-softplus(-z))

Identity used:  _smooth_clip(z, g) = [softplus(z)-softplus(z-g)] + [-softplus(-z)]
                                      └ rectified+capped pull ┘    └ push ┘
So scaling only the push branch by g_nerve gives the proposed form, and at
g_nerve=1 the added term is 0*(...) = 0 -> bit-identical to current."""
import jax, jax.numpy as jnp
import oculomotor.models.brain_models.final_common_pathway as fcp

NM = float(fcp._NERVE_MAX)


def old_rectify(z, g):
    return fcp._smooth_clip(z, g * NM)


def new_rectify(z, g):
    return fcp._smooth_clip(z, g * NM) + (g - 1.0) * (-jax.nn.softplus(-z))


z = jnp.linspace(-3 * NM, 3 * NM, 4001)          # push (neg) .. pull (pos), past cap
print(f'NERVE_MAX = {NM:.1f}\n')

# (1) HEALTHY g_nerve=1 -> must be bit-identical
g1 = jnp.ones_like(z)
d = jnp.max(jnp.abs(new_rectify(z, g1) - old_rectify(z, g1)))
print(f'(1) HEALTHY  g_nerve=1 : max|new-old| over full range = {float(d):.3e}   '
      f'{"BIT-IDENTICAL" if float(d) == 0.0 else "DIFFERS"}')

# (2) DEAD g_nerve=0 -> pull already 0; push must go from negative to 0
g0 = jnp.zeros_like(z)
o0, n0 = old_rectify(z, g0), new_rectify(z, g0)
neg = z < 0
print(f'(2) DEAD     g_nerve=0 : push region (z<0)  '
      f'old in [{float(o0[neg].min()):+.2f}, {float(o0[neg].max()):+.2f}] (pushes)   '
      f'new in [{float(n0[neg].min()):+.3f}, {float(n0[neg].max()):+.3f}] (silent)')
print(f'                         pull region (z>0)  '
      f'old max {float(o0[z > 0].max()):+.3f}   new max {float(n0[z > 0].max()):+.3f}  (both ~0)')

# (3) PARTIAL g_nerve=0.5 -> push ~halved, pull capped at 0.5*NM
g5 = 0.5 * jnp.ones_like(z)
o5, n5 = old_rectify(z, g5), new_rectify(z, g5)
iz = int(jnp.argmin(jnp.abs(z - (-NM))))          # a strong push command
ip = int(jnp.argmin(jnp.abs(z - (2 * NM))))       # a strong pull command (past cap)
print(f'(3) PARTIAL  g_nerve=0.5: push @z=-NM  old {float(o5[iz]):+.2f} -> new {float(n5[iz]):+.2f} (~half)   '
      f'pull @z=2NM  old {float(o5[ip]):.2f}  new {float(n5[ip]):.2f} (capped 0.5NM={0.5*NM:.0f})')
