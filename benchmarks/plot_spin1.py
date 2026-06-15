"""Spin-1 (Haldane) showcase figure: the incremental bulk energy density converging
to the known spin-1 Heisenberg value -1.401484039 J, from qx_spin1_showcase
(docs/spin1_haldane.csv: N, E_per_site, e_bulk_incremental, wall_s). Pure-stdlib SVG.

e_bulk(N) = (E_N - E_{N-dN})/dN removes the open-chain edge/bond-counting offset, so
it converges to the bulk Haldane density far faster than the raw E/N. The |error| vs
N (log y) drops to ~1e-8: the d=3 MPO on the same fast block path as spin-1/2 nails
the known energy density. Run: python benchmarks/plot_spin1.py -> docs/spin1_haldane.svg
"""
import csv, math, os

HERE = os.path.dirname(__file__)
E_EXACT = -1.401484039
rows = list(csv.DictReader(open(os.path.join(HERE, "..", "docs", "spin1_haldane.csv"))))
# skip the first row (no increment) and any nan
pts = [(int(r["N"]), abs(float(r["e_bulk_incremental"]) - E_EXACT))
       for r in rows if r["e_bulk_incremental"] not in ("nan", "")]
N = [p[0] for p in pts]
ERR = [max(p[1], 1e-10) for p in pts]

W, H = 760, 500
L, R, T, B = 84, 70, 64, 64
PX0, PX1, PY0, PY1 = L, W - R, T, H - B
xmin, xmax = 0, max(N) + 8
ylo, yhi = -9.0, -2.0                       # log10(|error|) range
def xpx(n): return PX0 + (n - xmin) / (xmax - xmin) * (PX1 - PX0)
def ypx(e): return PY1 - (math.log10(e) - ylo) / (yhi - ylo) * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
for e in range(int(ylo), int(yhi) + 1):
    y = ypx(10**e)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">1e{e}</text>')
xt = 16
while xt <= max(N):
    x = xpx(xt)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{xt}</text>')
    xt += 16
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-28}" text-anchor="middle" font-size="13" fill="#222">chain length  N</text>')
s.append(f'<text x="22" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" transform="rotate(-90 22 {(PY0+PY1)/2:.0f})">| e_bulk(N) &#8722; e_exact |  (J)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Spin-1 Haldane chain: bulk energy density on the fast block path</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="45" text-anchor="middle" font-size="12" fill="#555">d=3 MPO via run_dmrg_blocks (universal-MPO); e_exact = &#8722;1.401484039 J, ED-validated</text>')
pts_str = " ".join(f"{xpx(n):.1f},{ypx(e):.1f}" for n, e in zip(N, ERR))
s.append(f'<polyline points="{pts_str}" fill="none" stroke="#2ca02c" stroke-width="2.4"/>')
for n, e in zip(N, ERR):
    s.append(f'<circle cx="{xpx(n):.1f}" cy="{ypx(e):.1f}" r="3.5" fill="#2ca02c"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-10}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'e_bulk(N) = (E_N &#8722; E_{N&#8722;dN})/dN removes the open-chain edge offset; gapped Haldane GS, energy saturates by &#967;~16.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "spin1_haldane.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out), "| bulk-energy error falls to", f"{min(ERR):.1e}")
