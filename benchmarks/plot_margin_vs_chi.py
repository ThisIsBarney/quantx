"""Render the QuantX-GPU vs TeNPy-CPU(MKL) margin-vs-chi headline figure as a
dependency-free SVG (pure stdlib -- no matplotlib/numpy, so it runs anywhere).

Data: honest, post-LAPACKE-fix, min-of-2 cool-clock measurements from
gpu_vs_tenpy_quench.py (see docs/gpu_acceleration_report.md). N=32 and N=64 were each
measured in their own session, so cross-N comparison carries the ~2x thermal-noise
caveat; the within-curve trend (advantage grows with chi) is the robust message.

Run:  python benchmarks/plot_margin_vs_chi.py   ->  docs/gpu_margin_vs_chi.svg
"""
import math
import os

CHI = [256, 512, 1024]
SERIES = [
    ("N=32  GPU (FP64)",          "#1f77b4", "none", [0.91, 1.12, 2.17]),
    ("N=32  GPU (FP32, opt-in)",  "#1f77b4", "6,4", [1.16, 1.57, 2.74]),
    ("N=64  GPU (FP64)",          "#d62728", "none", [0.79, 1.08, 2.22]),
    ("N=64  GPU (FP32, opt-in)",  "#d62728", "6,4", [1.03, 1.50, 2.66]),
]
YMIN, YMAX = 0.6, 3.0
W, H = 780, 500
L, R, T, B = 78, 250, 66, 84          # margins (R wide for the legend box)
PX0, PX1 = L, W - R                    # x plot range (px)
PY0, PY1 = T, H - B                    # y plot range (px)

def xpx(chi):  # log2 x
    lo, hi = math.log2(CHI[0]), math.log2(CHI[-1])
    return PX0 + (math.log2(chi) - lo) / (hi - lo) * (PX1 - PX0)
def ypx(v):
    return PY1 - (v - YMIN) / (YMAX - YMIN) * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')

# y gridlines + ticks
yt = v = 1.0
ticks = [0.6, 1.0, 1.5, 2.0, 2.5, 3.0]
for v in ticks:
    y = ypx(v)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#e6e6e6" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{v:.1f}×</text>')
# break-even emphasis at 1.0
y1 = ypx(1.0)
s.append(f'<line x1="{PX0}" y1="{y1:.1f}" x2="{PX1}" y2="{y1:.1f}" stroke="#999" stroke-width="1.4" stroke-dasharray="2,3"/>')
s.append(f'<text x="{PX1-4}" y="{y1-5:.1f}" text-anchor="end" font-size="10" fill="#999">break-even vs TeNPy</text>')
# x ticks
for c in CHI:
    x = xpx(c)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{c}</text>')
# axes
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
# axis labels
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-44}" text-anchor="middle" font-size="13" fill="#222">bond dimension  χ</text>')
s.append(f'<text x="22" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" '
         f'transform="rotate(-90 22 {(PY0+PY1)/2:.0f})">speedup  (QuantX-GPU / TeNPy-CPU)</text>')
# title
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Néel-quench TEBD: QuantX-GPU vs TeNPy-CPU (MKL, 1 thread)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="46" text-anchor="middle" font-size="12" fill="#555">the margin grows with χ — Heisenberg chain, 50 steps</text>')

# series
for name, color, dash, ys in SERIES:
    pts = " ".join(f"{xpx(c):.1f},{ypx(y):.1f}" for c, y in zip(CHI, ys))
    da = f' stroke-dasharray="{dash}"' if dash != "none" else ""
    s.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.4"{da}/>')
    for c, y in zip(CHI, ys):
        s.append(f'<circle cx="{xpx(c):.1f}" cy="{ypx(y):.1f}" r="4" fill="{color}"/>')
        s.append(f'<text x="{xpx(c):.1f}" y="{ypx(y)-8:.1f}" text-anchor="middle" font-size="9.5" fill="{color}">{y:.2f}</text>')

# legend (top-left inside plot)
lx, ly = PX0 + 14, PY0 + 12
s.append(f'<rect x="{lx-8}" y="{ly-12}" width="186" height="74" fill="white" stroke="#ddd" rx="4"/>')
for i, (name, color, dash, _) in enumerate(SERIES):
    yy = ly + i * 16
    da = f' stroke-dasharray="{dash}"' if dash != "none" else ""
    s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" stroke="{color}" stroke-width="2.4"{da}/>')
    s.append(f'<text x="{lx+28}" y="{yy+4}" font-size="10.5" fill="#333">{name}</text>')

# small-chi annotation + caption
s.append(f'<text x="{PX1-6}" y="{PY0+34}" text-anchor="end" font-size="10" fill="#666">small χ (128): ~4.7× — C++-overhead regime (off-scale)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'Consumer RTX 4090 + i5-13600KF, no clock pinning → absolute ratios are ±~2× thermally noisy; the upward χ-trend is robust.</text>')
s.append('</svg>')

out = os.path.join(os.path.dirname(__file__), "..", "docs", "gpu_margin_vs_chi.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", os.path.normpath(out))
