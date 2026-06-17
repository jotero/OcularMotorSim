"""Unit-check _binocular_display decision logic across the four modes (no sim)."""
import numpy as np
from oculomotor.llm_pipeline.run import _binocular_display, _fmt_dist

T = 1000
def stim(depth, present):
    pt = np.zeros((T, 3), np.float32)
    pt[:, 0] = 0.0
    pt[:, 2] = depth
    tp = np.full(T, 1.0 if present is True else 0.0, np.float32) if not isinstance(present, np.ndarray) else present
    return dict(p_target_array=pt,
                target_present_L_array=tp, target_present_R_array=tp)

sig = dict(ipd=0.064)

def report(name, stim_kw, tonic):
    cal = _binocular_display(sig, stim_kw, tonic)
    print(f"{name:30s} off_L={cal['off_L']:+.3f}  off_R={cal['off_R']:+.3f}  "
          f"label={cal['zero_label']!r}")

# 1. constant near depth 0.5 m, target present
report("near 0.5 m (constant)", stim(0.5, True), 3.66)
# 2. typical 1 m target
report("1 m (constant)", stim(1.0, True), 3.66)
# 3. varying depth 0.3 -> 1.0 m
d = np.linspace(0.3, 1.0, T).astype(np.float32)
report("varying 0.3->1.0 m", stim(d, True), 3.66)
# 4. dark / no target (tonic resting vergence 3.66)
report("dark (no target)", stim(1.0, False), 3.66)
# 5. dark with ~zero tonic
report("dark, tonic~0", stim(1.0, False), 0.0)

# expected vc at 0.5 m:
vc = 2 * np.degrees(np.arctan(0.032 / 0.5))
print(f"\nexpected near-0.5m convergence vc = {vc:.3f} deg  -> off_L should be {-vc/2:+.3f}")
print("fmt:", _fmt_dist(0.5), "|", _fmt_dist(1.0), "|", _fmt_dist(2.5))
