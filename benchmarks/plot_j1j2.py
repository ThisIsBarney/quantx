"""J1-J2 frustrated-chain showcase figure: the low-chi energy gap (an entanglement
proxy) collapsing toward the Majumdar-Ghosh point, from qx_j1j2_showcase
(docs/j1j2_dimerization.csv: j2, E_per_site, gap_per_site). Pure-stdlib SVG.

The gap = E(chi=8) - E(chi=128) per site. Away from MG the frustrated ground
state is entangled and chi=8 under-converges (large gap); approaching J2/J1=0.5
the GS dimerizes to a bond-dimension-2 product and the gap collapses ~1000x. Log
y-axis. Run: python benchmarks/plot_j1j2.py -> docs/j1j2_dimerization.svg
"""
import csv, math, os

HERE = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "docs", "j1j2_dimerization.csv"))))
J = [float(r["j2"]) for r in rows]
E = [float(r["E_per_site"]) for r in rows]
G = [max(float(r["gap_per_site"]), 1e-12) for r in rows]

W, H = 760, 500
L, R, T, B = 78, 80, 64, 64
PX0, PX1, PY0, PY1 = L, W - R, T, H - B
xmin, xmax = 0.0, 0.62
ylo, yhi = -8.0, -2.5                      # log10(gap) range
def xpx(j): return PX0 + (j - xmin) / (xmax - xmin) * (PX1 - PX0)
def ypx(g): return PY1 - (math.log10(g) - ylo) / (yhi - ylo) * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# log y gridlines (decades)
for e in range(int(ylo), int(yhi) + 1):
    y = ypx(10**e)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">1e{e}</text>')
# x ticks
xt = 0.0
while xt <= 0.601:
    x = xpx(xt)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{xt:.1f}</text>')
    xt += 0.1
# MG point marker
xmg = xpx(0.5)
s.append(f'<line x1="{xmg:.1f}" y1="{PY0}" x2="{xmg:.1f}" y2="{PY1}" stroke="#d62728" stroke-width="1.3" stroke-dasharray="5,4"/>')
s.append(f'<text x="{xmg:.1f}" y="{PY0-6}" text-anchor="middle" font-size="11" fill="#d62728">Majumdar&#8211;Ghosh  J2=J1/2</text>')
# axes
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-28}" text-anchor="middle" font-size="13" fill="#222">frustration  J2 / J1</text>')
s.append(f'<text x="20" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" transform="rotate(-90 20 {(PY0+PY1)/2:.0f})">&#967;=8 energy gap / site</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">J1&#8211;J2 chain: ground-state entanglement collapse at Majumdar&#8211;Ghosh</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="45" text-anchor="middle" font-size="12" fill="#555">8-wide MPO on the fast block path (universal-MPO); N=32, ED-validated</text>')
# gap curve
pts = " ".join(f"{xpx(j):.1f},{ypx(g):.1f}" for j, g in zip(J, G))
s.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2.4"/>')
for j, g in zip(J, G):
    s.append(f'<circle cx="{xpx(j):.1f}" cy="{ypx(g):.1f}" r="3.5" fill="#1f77b4"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-10}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'gap = E(&#967;=8) &#8722; E(&#967;=128) per site; at J2=J1/2 the GS is a bond-dimension-2 dimer product (&#967;=8 gives the exact &#8722;1.5).</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "j1j2_dimerization.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out), "| gap dips from", f"{max(G):.1e}", "to", f"{min(G):.1e}")
