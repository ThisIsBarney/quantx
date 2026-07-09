"""Two-leg Hubbard ladder spin gap vs 1/Lx, from docs/hubbard_ladder_gap.csv
(qx_hubbard_showcase QX_LADDER sweep). Pure-stdlib SVG.

Delta_s(Lx) = E(N=2Lx, Sz=1) - E(N=2Lx, Sz=0) on the open 2xLx ladder (U=8, t=tp=1),
computed on the rung-fused d=16 fast block path with packed (N, 2Sz) sector targeting.
A linear fit in 1/Lx through the three largest sizes extrapolates to the bulk gap;
the open-ladder finite-size correction is O(1/Lx) at leading order.

Run: python benchmarks/plot_hubbard_ladder.py -> docs/hubbard_ladder_gap.svg
"""
import csv, os

HERE = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "docs", "hubbard_ladder_gap.csv"))))
pts = sorted(((int(r["Lx"]), float(r["gap"])) for r in rows))
U = float(rows[0]["U"])

# linear fit gap = a + b/Lx through the three largest sizes
big = pts[-3:]
xs = [1.0 / L for L, _ in big]; ys = [g for _, g in big]
n = len(xs); sx = sum(xs); sy = sum(ys)
sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
b = (n * sxy - sx * sy) / (n * sxx - sx * sx)
a = (sy - b * sx) / n                                  # extrapolated bulk gap

W, H = 720, 480
L0, R0, T0, B0 = 84, 40, 64, 64
PX0, PX1, PY0, PY1 = L0, W - R0, T0, H - B0
xmax = max(1.0 / L for L, _ in pts) * 1.12
ymax = max(g for _, g in pts) * 1.15
def xpx(v): return PX0 + v / xmax * (PX1 - PX0)
def ypx(v): return PY1 - v / ymax * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
g = 0.0
while g <= ymax:
    y = ypx(g)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{g:.2f}</text>')
    g += 0.05
for L, _ in pts:
    x = xpx(1.0 / L)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+19}" text-anchor="middle" font-size="10.5" fill="#444">1/{L}</text>')
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
# fit line from x=0 to x=1/4
xf0, xf1 = 0.0, max(xs) * 1.06
s.append(f'<line x1="{xpx(xf0):.1f}" y1="{ypx(a + b*xf0):.1f}" x2="{xpx(xf1):.1f}" y2="{ypx(a + b*xf1):.1f}" '
         f'stroke="#999" stroke-width="1.3" stroke-dasharray="5,4"/>')
poly = " ".join(f"{xpx(1.0/L):.1f},{ypx(gp):.1f}" for L, gp in pts)
s.append(f'<polyline points="{poly}" fill="none" stroke="#d62728" stroke-width="2.2"/>')
for L, gp in pts:
    s.append(f'<circle cx="{xpx(1.0/L):.1f}" cy="{ypx(gp):.1f}" r="4" fill="#d62728"/>')
# extrapolated intercept marker
s.append(f'<circle cx="{xpx(0):.1f}" cy="{ypx(a):.1f}" r="4.5" fill="none" stroke="#1f77b4" stroke-width="2"/>')
s.append(f'<text x="{xpx(0)+10:.1f}" y="{ypx(a)-8:.1f}" font-size="11.5" font-weight="bold" fill="#1f77b4">'
         f'&#916;<tspan baseline-shift="sub" font-size="9">s</tspan>(&#8734;) &#8776; {a:.3f} t</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-24}" text-anchor="middle" font-size="13" fill="#222">inverse ladder length  1/Lx</text>')
s.append(f'<text x="24" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" '
         f'transform="rotate(-90 24 {(PY0+PY1)/2:.0f})">spin gap  &#916;<tspan baseline-shift="sub" font-size="10">s</tspan> = E(Sz=1) &#8722; E(Sz=0)  (t=1)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">'
         f'Two-leg Hubbard ladder spin gap &#8212; U={U:g}t, half filling</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="45" text-anchor="middle" font-size="11.5" fill="#555">'
         'rung-fused d=16 fast block path, packed (N, 2Sz) sector targeting; open 2&#215;Lx, &#967;=256 (&#967;-converged to 1e-6)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-8}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'The gapped spin sector of the half-filled ladder (spin liquid of rung singlets); dashes = linear 1/Lx fit through the three largest sizes.</text>')
s.append('</svg>')
out = os.path.join(HERE, "..", "docs", "hubbard_ladder_gap.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out), f"| extrapolated gap = {a:.4f} t")
