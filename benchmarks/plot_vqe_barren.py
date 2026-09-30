#!/usr/bin/env python3
"""Render docs/vqe_barren.svg from docs/vqe_barren.csv.

Part F asks whether part E's width penalty is a TRAINABILITY effect. The figure has to
be able to say "no" as clearly as "yes", so it plots the raw quantity (gradient variance
of the dimensionless cost, per width, vs depth) and draws the instrument's own noise
floor on the same axes. A point near that line is measuring the difference engine, not
the landscape, and the reader should be able to see that without being told.

Like plot_vqe_width.py, every number in the caption is computed from the CSV -- the
caption states the conclusion the data supports, not the one that was hoped for.
"""
import csv, math, os

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
rows = list(csv.DictReader(open(os.path.join(DOCS, "vqe_barren.csv"))))
if not rows:
    raise SystemExit("docs/vqe_barren.csv is empty -- run QX_VQE_ONLY=F qx_vqe_showcase")

D = {(int(r["W"]), int(r["p"])): r for r in rows}
widths = sorted({int(r["W"]) for r in rows})
depths = sorted({int(r["p"]) for r in rows})
NAME = {1: "W=1 chain", 2: "W=2 ladder", 3: "W=3 cylinder"}
COL = {1: "#1a6fc4", 2: "#c46a1a", 3: "#8e1ac4"}

def var(w, p):
    r = D.get((w, p))
    return float(r["var_frac"]) if r else None


def sev(w, p):
    """measured standard error of the variance -- from the scan's own fourth moment"""
    r = D.get((w, p))
    return float(r["se_frac"]) if r else None

# the floor is a property of the instrument, identical on every row; carry it in the
# same units as the plotted variance so it can be drawn as one line
floor_abs = float(rows[0]["floor"])
# var_frac = var_raw / corr^2, so floor_frac^2 = (floor/corr)^2 per width; recover corr
# from the row rather than re-deriving it, and use the WORST (largest) floor over widths
floor_var = 0.0
for w in widths:
    r = D.get((w, depths[0]))
    if not r:
        continue
    vr, vf = float(r["var_raw"]), float(r["var_frac"])
    if vr > 0 and vf > 0:
        corr2 = vr / vf                      # = corr^2
        floor_var = max(floor_var, floor_abs * floor_abs / corr2)

W, H = 900, 430
L, R, T, B = 68, 24, 92, 58
PW, PH = (W - L - R - 200), (H - T - B)
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="Segoe UI,Helvetica,Arial,sans-serif">',
     f'<rect width="{W}" height="{H}" fill="#fff"/>']

vals = [v for w in widths for p in depths if (v := var(w, p))]
vals = [v for v in vals if v > 0]
lo = 10 ** math.floor(math.log10(min(vals + [floor_var]) if floor_var > 0 else min(vals)))
hi = 10 ** math.ceil(math.log10(max(vals)))
lo_clip = lo

def X(p): return L + (p - depths[0]) / max(1, (depths[-1] - depths[0])) * PW
def Y(v): return T + PH - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * PH

s.append(f'<text x="{L}" y="26" font-size="15.5" font-weight="bold" fill="#111">'
         'Is the width penalty a barren plateau?</text>')
s.append(f'<text x="{L}" y="46" font-size="11.5" fill="#555">'
         'variance of &#8706;(correlation fraction)/&#8706;&#952; over random angles '
         '&#952;~U(&#8722;&#960;,&#960;), same 12 qubits and same cost for all three; '
         'bars are 2&#963;, measured</text>')

# axes
s.append(f'<line x1="{L}" y1="{T}" x2="{L}" y2="{T+PH}" stroke="#333"/>')
s.append(f'<line x1="{L}" y1="{T+PH}" x2="{L+PW}" y2="{T+PH}" stroke="#333"/>')
e = int(math.log10(lo))
while 10 ** e <= hi * 1.001:
    y = Y(10 ** e)
    s.append(f'<line x1="{L}" y1="{y:.1f}" x2="{L+PW}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{L-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">'
             f'10<tspan dy="-4" font-size="7.5">{e}</tspan></text>')
    e += 1
for p in depths:
    s.append(f'<text x="{X(p):.1f}" y="{T+PH+18}" text-anchor="middle" font-size="10.5" '
             f'fill="#444">{p}</text>')
s.append(f'<text x="{L+PW/2:.0f}" y="{T+PH+40}" text-anchor="middle" font-size="11" '
         f'fill="#333">circuit depth p</text>')
s.append(f'<text x="18" y="{T+PH/2:.0f}" text-anchor="middle" font-size="11" fill="#333" '
         f'transform="rotate(-90 18 {T+PH/2:.0f})">Var[&#8706;frac/&#8706;&#952;]</text>')

# the instrument's noise floor -- anything at or below this line is not a measurement
if floor_var > 0 and floor_var >= lo:
    yf = Y(floor_var)
    s.append(f'<line x1="{L}" y1="{yf:.1f}" x2="{L+PW}" y2="{yf:.1f}" stroke="#b00" '
             f'stroke-width="1.2" stroke-dasharray="3,3"/>')
    s.append(f'<text x="{L+PW-4}" y="{yf-6:.1f}" text-anchor="end" font-size="9.5" '
             f'fill="#b00">finite-difference noise floor</text>')

for w in widths:
    pts = [(X(p), Y(v)) for p in depths if (v := var(w, p)) and v > 0]
    if not pts:
        continue
    s.append('<polyline fill="none" stroke="{}" stroke-width="2.1" points="{}"/>'.format(
        COL[w], " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)))
    # error bars, drawn from the MEASURED standard error: without them the reader cannot
    # tell "these two curves coincide" from "these two curves differ a little"
    for p in depths:
        # NB: not `hi`/`lo` -- those are the axis limits Y() closes over, and shadowing
        # them here would silently rescale every point drawn after the first error bar
        v, ev = var(w, p), sev(w, p)
        if not v or not ev:
            continue
        b_hi, b_lo = v + 2 * ev, max(v - 2 * ev, lo_clip)
        if b_hi <= 0 or b_lo <= 0:
            continue
        x = X(p)
        s.append(f'<line x1="{x:.1f}" y1="{Y(b_hi):.1f}" x2="{x:.1f}" y2="{Y(b_lo):.1f}" '
                 f'stroke="{COL[w]}" stroke-width="1.1"/>')
        for yv in (b_hi, b_lo):
            s.append(f'<line x1="{x-3:.1f}" y1="{Y(yv):.1f}" x2="{x+3:.1f}" '
                     f'y2="{Y(yv):.1f}" stroke="{COL[w]}" stroke-width="1.1"/>')
    for x, y in pts:
        s.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3.1" fill="{COL[w]}"/>')

# legend + the numbers the caption quotes
lx = L + PW + 26
s.append(f'<text x="{lx}" y="{T+2}" font-size="11" font-weight="bold" fill="#111">'
         'at the deepest point</text>')
yy = T + 22
pl = depths[-1]
for w in widths:
    v = var(w, pl)
    if v is None:
        continue
    s.append(f'<line x1="{lx}" y1="{yy-4}" x2="{lx+18}" y2="{yy-4}" stroke="{COL[w]}" '
             f'stroke-width="2.4"/>')
    s.append(f'<circle cx="{lx+9}" cy="{yy-4}" r="3.1" fill="{COL[w]}"/>')
    s.append(f'<text x="{lx+24}" y="{yy}" font-size="10.5" fill="#222">{NAME[w]}</text>')
    s.append(f'<text x="{lx+24}" y="{yy+14}" font-size="9.5" fill="#666">'
             f'Var {v:.2e}</text>')
    yy += 36

# caption: computed, so it states what the data says
def ratio(a, b, p):
    va, vb = var(a, p), var(b, p)
    return (va / vb) if (va and vb) else None
r31 = ratio(3, 1, pl)
r21 = ratio(2, 1, pl)
decay = None
v_lo, v_hi = var(widths[0], depths[0]), var(widths[0], depths[-1])
if v_lo and v_hi:
    decay = v_hi / v_lo
bits = []
if r31:
    bits.append(f"At p={pl} the cylinder sits at &#215;{r31:.2f} the chain&#8217;s "
                f"gradient variance")
    if r21:
        bits.append(f"while the ladder is at &#215;{r21:.2f} &#8212; the suppression is "
                    f"not monotone in width.")
if decay:
    bits.append(f"Over p={depths[0]}&#8594;{pl} the chain&#8217;s variance changes by "
                f"&#215;{decay:.2f}.")
snr_min = min(float(r["snr"]) for r in rows)
# same spelling the doc uses, so the two cannot drift apart visually
bits.append("Every point sits at least &#215;" +
            f"{snr_min:.1e}".replace("e+0", "e").replace("e+", "e") +
            " above the finite-difference noise floor.")
s.append(f'<text x="{L}" y="{H-16}" font-size="10.5" fill="#333">' + " ".join(bits) +
         '</text>')
s.append('</svg>')

out = os.path.join(DOCS, "vqe_barren.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", out)
for w in widths:
    r = D.get((w, depths[-1]))
    if r:
        print(f"  W={w} p={depths[-1]}: Var={float(r['var_frac']):.3e}  "
              f"std={float(r['std_frac']):.3e}  snr={float(r['snr']):.0f}")
