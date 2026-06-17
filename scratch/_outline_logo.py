"""Outline the ViSiOMlab wordmark to vector paths (font-independent).

The logo SVG used live <text> in Century Gothic, so it rendered with a fallback
font on machines lacking it. This reads GOTHIC.TTF and replaces each text run with
<path> outlines positioned exactly as the original (advance widths + letter-spacing),
preserving the eye-in-O art and all transforms. One-shot; lives in scratch/.
"""
from fontTools.ttLib import TTFont
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen

FONT = r"C:\Windows\Fonts\GOTHIC.TTF"
OUT  = r"d:\OneDrive\UC Berkeley\OMlab - JOM\Code\ClaudeOculomotorJax\web\assets\visiomlab-logo.svg"

font  = TTFont(FONT)
upem  = font["head"].unitsPerEm
cmap  = font.getBestCmap()
hmtx  = font["hmtx"]
gset  = font.getGlyphSet()

def run_path(text, font_size, ls_em):
    """Return (d, scale, advance_px) for a text run at the given size/letter-spacing."""
    s = font_size / upem
    ls_units = ls_em * upem            # letter-spacing in font units (em -> upem)
    pen_all = SVGPathPen(gset)
    x = 0.0
    for ch in text:
        gname = cmap[ord(ch)]
        tp = TransformPen(pen_all, (1, 0, 0, 1, x, 0))
        gset[gname].draw(tp)
        x += hmtx[gname][0] + ls_units
    return pen_all.getCommands(), s, x * s

def run_group(text, font_size, ls_em, tx, ty, stroke_px):
    d, s, _ = run_path(text, font_size, ls_em)
    sw = stroke_px / s                 # keep visual stroke constant under scale(s,-s)
    return (f'  <g transform="translate({tx} {ty}) scale({s:.6f} {-s:.6f})">'
            f'<path d="{d}" fill="#231f20" stroke="#231f20" '
            f'stroke-width="{sw:.3f}" stroke-miterlimit="10"/></g>')

visi = run_group("ViSi", 100, 0.0,   71.63, 141.78, 2)
o    = run_group("O",    100, 0.05,  71.63, 141.78, 2)
m    = run_group("M",    100, 0.0,  160.41, 141.78, 2)
lab  = run_group("lab",   50, -0.05, 260.42, 141.78, 1)

svg = f'''<?xml version="1.0" encoding="UTF-8"?>
<svg id="visiomlab" data-name="ViSiOMlab" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 581 228.32">
  <defs>
    <style>
      .cls-2 {{ fill: #fff; stroke: #fff; stroke-miterlimit: 10; }}
      .cls-3 {{ fill: #231f20; stroke: #231f20; stroke-miterlimit: 10; }}
      .cls-6 {{ fill: #d7df23; }}
    </style>
  </defs>
  <!-- ViSi prefix (outlined Century Gothic) -->
{visi}
  <!-- OMlab wordmark + eye-in-O, shifted right to make room for ViSi -->
  <g transform="translate(165 0)">
{o}
{m}
    <g>
      <path class="cls-6" d="M111.93,103.47c-5.87,8.94-15.1,13.25-20.6,9.63-5.51-3.62-5.21-13.8.66-22.74,5.87-8.94,15.1-13.25,20.61-9.63,5.51,3.62,5.21,13.8-.66,22.73Z"/>
      <path class="cls-3" d="M104.61,98.66c-2.35,3.57-6.04,5.3-8.24,3.85-2.2-1.45-2.09-5.52.26-9.09,2.35-3.57,6.04-5.3,8.24-3.85,2.2,1.45,2.08,5.52-.26,9.09Z"/>
      <path class="cls-2" d="M97.12,105.71c1.78-2.71,5.31-3.54,7.88-1.85,2.57,1.69,3.21,5.26,1.43,7.97-1.78,2.71-5.31,3.54-7.88,1.85-2.57-1.69-3.21-5.26-1.43-7.97Z"/>
    </g>
{lab}
  </g>
</svg>
'''

with open(OUT, "w", encoding="utf-8") as f:
    f.write(svg)

# sanity
for name, (t, fs, ls) in {"ViSi": ("ViSi",100,0.0), "O": ("O",100,0.05),
                          "M": ("M",100,0.0), "lab": ("lab",50,-0.05)}.items():
    d, s, adv = run_path(t, fs, ls)
    print(f"{name:5s} chars={len(t)} d_len={len(d):5d} advance_px={adv:7.2f} scale={s:.5f}")
print("upem =", upem, "-> wrote", OUT)
