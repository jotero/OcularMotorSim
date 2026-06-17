"""Render the outlined wordmark paths from the generated SVG to a PNG for eyeballing.
Parses the <g transform=...><path d=...> runs and the eye art, draws with matplotlib.
"""
import re, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
import numpy as np

SVG = r"d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\web\assets\visiomlab-logo.svg"
src = open(SVG, encoding="utf-8").read()

def parse_d(d):
    toks = re.findall(r"[MLCQZ]|-?\d*\.?\d+(?:e-?\d+)?", d)
    verts, codes = [], []
    i, cur = 0, (0.0, 0.0)
    while i < len(toks):
        c = toks[i]; i += 1
        if c == "M":
            cur = (float(toks[i]), float(toks[i+1])); i += 2
            verts.append(cur); codes.append(Path.MOVETO)
        elif c == "L":
            cur = (float(toks[i]), float(toks[i+1])); i += 2
            verts.append(cur); codes.append(Path.LINETO)
        elif c == "C":
            for k in range(3):
                verts.append((float(toks[i]), float(toks[i+1]))); i += 2
                codes.append(Path.CURVE4)
            cur = verts[-1]
        elif c == "Q":
            for k in range(2):
                verts.append((float(toks[i]), float(toks[i+1]))); i += 2
                codes.append(Path.CURVE3)
            cur = verts[-1]
        elif c == "Z":
            verts.append(cur); codes.append(Path.CLOSEPOLY)
    return verts, codes

fig, ax = plt.subplots(figsize=(8, 3.2), dpi=140)
ax.set_facecolor("#f4f5f7"); fig.patch.set_facecolor("#f4f5f7")

# outlined wordmark runs: <g transform="translate(tx ty) scale(s -s)"><path d="...">
# O/M/lab live inside <g transform="translate(165 0)"> — add that offset for runs
# that appear after the group opens in the source.
group_pos = src.index("translate(165 0)")
for m in re.finditer(r'transform="translate\(([\-\d.]+) ([\-\d.]+)\) scale\(([\-\d.]+) ([\-\d.]+)\)">'
                     r'<path d="([^"]+)"', src):
    tx, ty, sx, sy = map(float, m.group(1, 2, 3, 4))
    if m.start() > group_pos:
        tx += 165.0
    verts, codes = parse_d(m.group(5))
    v = np.array(verts)
    v[:, 0] = tx + sx * v[:, 0]
    v[:, 1] = ty + sy * v[:, 1]
    ax.add_patch(PathPatch(Path(v, codes), fc="#231f20", ec="none", lw=0))

# eye art: <g transform="translate(165 0)"> wraps cls-6/cls-3/cls-2 with relative-cmd paths.
# Quick-and-dirty: skip (unchanged from original). Just mark the O center region.
ax.set_xlim(0, 581); ax.set_ylim(228.32, 0); ax.set_aspect("equal"); ax.axis("off")
out = r"d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\scratch\_logo_check.png"
fig.savefig(out, bbox_inches="tight", facecolor="#f4f5f7")
print("wrote", out)
