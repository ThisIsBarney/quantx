"""Fermionic Hubbard showcase figure: the incremental bulk energy density from the FAST
block-sparse DMRG (qx_hubbard_showcase, docs/hubbard_lieb_wu.csv) landing on the EXACT
Lieb-Wu Bethe-ansatz energy e(U) at three interaction strengths (U=2,4,8, the Mott
regime). Pure-stdlib SVG (no matplotlib).

e_bulk(N) = (E_N - E_{N-dN})/dN removes the open-chain edge offset, so it converges to
the thermodynamic Hubbard density; the |error| vs the exact e(U) falls to ~1e-5 -- a
genuinely fermionic model (Fermi statistics carried in the Jordan-Wigner MPO) on the
same fast block path as the spin chains, checked against a solution brute-force ED
cannot reach. Run: python benchmarks/plot_hubbard.py -> docs/hubbard_lieb_wu.svg
"""
import csv, os
from collections import defaultdict

HERE = os.path.dirname(__file__)
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "docs", "hubbard_lieb_wu.csv"))))

# group by U: list of (N, e_bulk); and the exact Lieb-Wu value per U
series = defaultdict(list)
elw = {}
for r in rows:
    U = float(r["U"]); elw[U] = float(r["e_liebwu"])
    if r["e_bulk_incremental"] not in ("nan", ""):
        series[U].append((int(r["N"]), float(r["e_bulk_incremental"])))
Us = sorted(series)

W, H = 780, 500
L, Rm, T, B = 92, 150, 60, 62
PX0, PX1, PY0, PY1 = L, W - Rm, T, H - B
allN = [n for U in Us for n, _ in series[U]]
xmin, xmax = 0, max(allN) + 6
ylo, yhi = -0.92, -0.28                      # energy-density range spanning e(2)..e(8)
COL = {2.0: "#1f77b4", 4.0: "#d62728", 8.0: "#2ca02c"}
def col(U): return COL.get(U, "#888")
def xpx(n): return PX0 + (n - xmin) / (xmax - xmin) * (PX1 - PX0)
def ypx(e): return PY1 - (e - ylo) / (yhi - ylo) * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
# y grid
e = -0.9
while e <= -0.30001:
    y = ypx(e)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{e:.1f}</text>')
    e += 0.1
# x ticks
for xt in (16, 32, 48, 64):
    x = xpx(xt)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{xt}</text>')
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-26}" text-anchor="middle" font-size="13" fill="#222">chain length  N  (half-filling)</text>')
s.append(f'<text x="24" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" transform="rotate(-90 24 {(PY0+PY1)/2:.0f})">bulk energy density  e_bulk  (t=1)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Fermionic Hubbard chain: fast block path vs exact Lieb-Wu Bethe-ansatz</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="44" text-anchor="middle" font-size="12" fill="#555">d=4 Jordan-Wigner MPO via run_dmrg_blocks; dashed = exact e(U), points = (E_N&#8722;E_{{N&#8722;dN}})/dN</text>')

for U in Us:
    c = col(U); ylw = ypx(elw[U])
    # exact Lieb-Wu line (dashed) + right-margin label
    s.append(f'<line x1="{PX0}" y1="{ylw:.1f}" x2="{PX1}" y2="{ylw:.1f}" stroke="{c}" stroke-width="1.3" stroke-dasharray="6 4" opacity="0.75"/>')
    err = abs(series[U][-1][1] - elw[U])
    s.append(f'<text x="{PX1+8}" y="{ylw-3:.1f}" font-size="12" font-weight="bold" fill="{c}">U={U:.0f}</text>')
    s.append(f'<text x="{PX1+8}" y="{ylw+12:.1f}" font-size="9.5" fill="#666">e={elw[U]:.4f}</text>')
    s.append(f'<text x="{PX1+8}" y="{ylw+24:.1f}" font-size="9" fill="#999">err {err:.0e}</text>')
    pts = series[U]
    poly = " ".join(f"{xpx(n):.1f},{ypx(v):.1f}" for n, v in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="2.2"/>')
    for n, v in pts:
        s.append(f'<circle cx="{xpx(n):.1f}" cy="{ypx(v):.1f}" r="3.6" fill="{c}"/>')

s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-9}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'Mott regime (U&gt;0, gapped charge -&gt; area law, &#967;=128 converged). Brute-force ED is impossible at these N; Lieb-Wu (1968) is exact.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "hubbard_lieb_wu.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
worst = max(abs(series[U][-1][1] - elw[U]) for U in Us)
print("wrote", os.path.normpath(out), "| largest final |e_bulk - e_liebwu| =", f"{worst:.1e}")
