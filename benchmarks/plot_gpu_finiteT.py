"""Finite-T dynamics GPU-vs-CPU speedup vs chi, as a dependency-free SVG (pure
stdlib), matching plot_margin_vs_chi.py. Data are wall times (seconds) of the
real-time purified evolution (qx_thermal_gpu_bench) with the per-block SVD routed
to cuSOLVER (GPU) vs forced onto the CPU Eigen/LAPACK path (QX_GPU_MIN_BLK toggle).

Edit DATA below with the measured (chi, gpu_s, cpu_s) and run:
  python benchmarks/plot_gpu_finiteT.py  ->  docs/gpu_finiteT_margin.svg
"""
import math, os, sys

# (chi, gpu_seconds, cpu_seconds) -- filled from qx_thermal_gpu_bench
DATA = [
    (512,  float(sys.argv[1]) if len(sys.argv) > 3 else 0.0,
           float(sys.argv[2]) if len(sys.argv) > 3 else 0.0),
    (1024, float(sys.argv[3]) if len(sys.argv) > 4 else 0.0,
           float(sys.argv[4]) if len(sys.argv) > 4 else 0.0),
]
CHI = [d[0] for d in DATA]
SPEED = [(d[2] / d[1] if d[1] > 0 else 0.0) for d in DATA]

W, H = 640, 460
L, R, T, B = 78, 40, 70, 70
PX0, PX1, PY0, PY1 = L, W - R, T, H - B
ymax = max(2.5, max(SPEED) * 1.15)
def xpx(c): lo, hi = math.log2(CHI[0]), math.log2(CHI[-1]); return PX0 + (math.log2(c)-lo)/(hi-lo)*(PX1-PX0)
def ypx(v): return PY1 - v / ymax * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
v = 0.5
while v <= ymax:
    y = ypx(v)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{v:.1f}&#215;</text>')
    v += 0.5
y1 = ypx(1.0)
s.append(f'<line x1="{PX0}" y1="{y1:.1f}" x2="{PX1}" y2="{y1:.1f}" stroke="#999" stroke-width="1.4" stroke-dasharray="2,3"/>')
s.append(f'<text x="{PX1-4}" y="{y1-5:.1f}" text-anchor="end" font-size="10" fill="#999">break-even (GPU = CPU)</text>')
for c in CHI:
    x = xpx(c)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{c}</text>')
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-30}" text-anchor="middle" font-size="13" fill="#222">bond dimension  &#967;</text>')
s.append(f'<text x="22" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" transform="rotate(-90 22 {(PY0+PY1)/2:.0f})">speedup  (CPU time / GPU time)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Finite-T dynamics: GPU vs CPU SVD throughput</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="46" text-anchor="middle" font-size="12" fill="#555">purified real-time evolution &#8212; the advantage grows with &#967;</text>')
pts = " ".join(f"{xpx(c):.1f},{ypx(v):.1f}" for c, v in zip(CHI, SPEED))
s.append(f'<polyline points="{pts}" fill="none" stroke="#1f77b4" stroke-width="2.6"/>')
for c, v in zip(CHI, SPEED):
    s.append(f'<circle cx="{xpx(c):.1f}" cy="{ypx(v):.1f}" r="5" fill="#1f77b4"/>')
    s.append(f'<text x="{xpx(c):.1f}" y="{ypx(v)-10:.1f}" text-anchor="middle" font-size="11" fill="#1f77b4">{v:.2f}&#215;</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-12}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">Heisenberg N=20 (40 purified sites), RTX 4090 vs i5-13600KF (1 thread); QX_GPU_MIN_BLK toggle.</text>')
s.append('</svg>')
with open(os.path.join(os.path.dirname(__file__), "..", "docs", "gpu_finiteT_margin.svg"), "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote docs/gpu_finiteT_margin.svg  speedups:", [f"{c}:{v:.2f}x" for c, v in zip(CHI, SPEED)])
