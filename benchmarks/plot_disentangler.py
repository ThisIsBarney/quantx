"""Render the F1 disentangler bond-growth figure as a dependency-free SVG (pure
stdlib -- no matplotlib/numpy, so it runs anywhere, matching plot_margin_vs_chi.py).

Reads docs/disentangler_bond_growth.csv (written by qx_thermal_spectral_bench:
columns t,A_off,A_on,B_off,B_on,max_off,max_on) and plots the bond dimension of
the unsourced state A and the sourced state B, with the Karrasch-Bardarson-Moore
ancilla-backward disentangler OFF vs ON. The A curves diverge sharply (the
disentangler cancels the real-time entanglement growth of the thermal-state
evolution); the B curves overlap (operator spreading is intrinsic), which is the
honest message: the disentangler is a compute win via the unsourced/bra state.

Run:  python benchmarks/plot_disentangler.py [csv]  ->  docs/disentangler_bond_growth.svg
"""
import csv
import os
import sys

HERE = os.path.dirname(__file__)
CSVP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "docs",
                                                           "disentangler_bond_growth.csv")
T, Aoff, Aon, Boff, Bon = [], [], [], [], []
with open(CSVP, newline="") as f:
    for row in csv.DictReader(f):
        T.append(float(row["t"]));      Aoff.append(int(row["A_off"]))
        Aon.append(int(row["A_on"]));   Boff.append(int(row["B_off"]))
        Bon.append(int(row["B_on"]))

W, H = 780, 500
L, R, TOP, BOT = 70, 230, 70, 70
PX0, PX1, PY0, PY1 = L, W - R, TOP, H - BOT
tmax = max(T) or 1.0
ymax = max(max(Aoff), max(Boff)) * 1.05 or 1.0

def xpx(t): return PX0 + t / tmax * (PX1 - PX0)
def ypx(v): return PY1 - v / ymax * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# y grid
ystep = 128 if ymax > 512 else 64
v = 0
while v <= ymax:
    y = ypx(v)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{v}</text>')
    v += ystep
# x ticks
xt = 0.0
while xt <= tmax + 1e-9:
    x = xpx(xt)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{xt:.1f}</text>')
    xt += 0.4
# axes + labels
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-30}" text-anchor="middle" font-size="13" fill="#222">time  t  (J=1)</text>')
s.append(f'<text x="20" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" '
         f'transform="rotate(-90 20 {(PY0+PY1)/2:.0f})">bond dimension</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Finite-T disentangler: purified-state bond growth</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="46" text-anchor="middle" font-size="12" fill="#555">Heisenberg chain, S(k,&#969;,T) correlator &#8212; ancilla-backward gauge OFF vs ON</text>')

def line(ys, color, dash, label, lx, ly):
    pts = " ".join(f"{xpx(t):.1f},{ypx(y):.1f}" for t, y in zip(T, ys))
    da = f' stroke-dasharray="{dash}"' if dash else ""
    s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{da}/>')
    s.append(f'<line x1="{lx}" y1="{ly}" x2="{lx+22}" y2="{ly}" stroke="{color}" stroke-width="2.4"{da}/>')
    s.append(f'<text x="{lx+28}" y="{ly+4}" font-size="10.5" fill="#333">{label}</text>')

lx, ly = PX1 + 16, PY0 + 16
s.append(f'<rect x="{lx-8}" y="{ly-14}" width="210" height="78" fill="white" stroke="#ddd" rx="4"/>')
line(Aoff, "#d62728", "",    "A unsourced, OFF", lx, ly)
line(Aon,  "#1f77b4", "",    "A unsourced, ON",  lx, ly + 18)
line(Boff, "#d62728", "5,4", "B sourced, OFF",   lx, ly + 36)
line(Bon,  "#1f77b4", "5,4", "B sourced, ON",    lx, ly + 54)

# annotation: A gap
kmid = len(T) // 2
if Aon[kmid] > 0:
    gap = Aoff[kmid] / Aon[kmid]
    ax = xpx(T[kmid])
    s.append(f'<text x="{ax:.0f}" y="{ypx((Aoff[kmid]+Aon[kmid])/2):.0f}" '
             f'font-size="11" fill="#1f77b4" font-weight="bold">A: {gap:.1f}&#215; smaller</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-12}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'The disentangler cancels the real-time entanglement growth of the unsourced state (A); the sourced ket (B) is operator-spreading-limited.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "disentangler_bond_growth.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", os.path.normpath(out))
