"""Spin-1 Haldane hidden-order figure: the den Nijs-Rommelse string-order parameter
O(r) saturating to a nonzero plateau, from qx_spin1_showcase
(docs/spin1_string_order.csv: r, O). Pure-stdlib SVG.

O(r) = -<gs| Sz_i (prod_{i<k<i+r} e^{i pi Sz_k}) Sz_{i+r} |gs> is the fingerprint of
the gapped Haldane phase: it stays NONZERO as r grows (hidden long-range order),
whereas every local two-point correlation decays to zero. The plateau (~0.37 here;
the thermodynamic Heisenberg value is 0.3743) is the order parameter of a topological
phase with no local order. Run: python benchmarks/plot_spin1_string.py
  -> docs/spin1_string_order.svg
"""
import csv, os

HERE = os.path.dirname(__file__)
O_INF = 0.374325   # thermodynamic spin-1 Heisenberg string order
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "docs", "spin1_string_order.csv"))))
R = [int(x["r"]) for x in rows]
O = [float(x["O"]) for x in rows]

W, H = 760, 500
L, Rm, T, B = 78, 70, 64, 64
PX0, PX1, PY0, PY1 = L, W - Rm, T, H - B
xmin, xmax = 0, max(R) + 2
ymin, ymax = 0.0, 0.55
def xpx(r): return PX0 + (r - xmin) / (xmax - xmin) * (PX1 - PX0)
def ypx(o): return PY1 - (o - ymin) / (ymax - ymin) * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
for k in range(0, 6):
    yv = 0.1 * k
    y = ypx(yv)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{yv:.1f}</text>')
for xt in range(0, max(R)+1, 4):
    x = xpx(xt)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{xt}</text>')
# thermodynamic plateau reference
yinf = ypx(O_INF)
s.append(f'<line x1="{PX0}" y1="{yinf:.1f}" x2="{PX1}" y2="{yinf:.1f}" stroke="#d62728" stroke-width="1.3" stroke-dasharray="5,4"/>')
s.append(f'<text x="{PX1-4}" y="{yinf-6:.1f}" text-anchor="end" font-size="11" fill="#d62728">O(&#8734;) = 0.3743  (thermodynamic)</text>')
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-28}" text-anchor="middle" font-size="13" fill="#222">string separation  r</text>')
s.append(f'<text x="22" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" transform="rotate(-90 22 {(PY0+PY1)/2:.0f})">string order  O(r)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Spin-1 Haldane chain: hidden long-range string order</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="45" text-anchor="middle" font-size="12" fill="#555">O(r) = &#8722;&#9001;Sz_i (&#928; e^{{i&#960;Sz}}) Sz_{{i+r}}&#9002; on the GS via the fast block path (N=40); ED-validated</text>')
pts = " ".join(f"{xpx(r):.1f},{ypx(o):.1f}" for r, o in zip(R, O))
s.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2.4"/>')
for r, o in zip(R, O):
    s.append(f'<circle cx="{xpx(r):.1f}" cy="{ypx(o):.1f}" r="3.5" fill="#1f77b4"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-10}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'O(r) plateaus at a nonzero value as r grows -- topological hidden order, with NO local order parameter.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "spin1_string_order.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out), "| O(r) plateaus near", f"{O[-1]:.3f}")
