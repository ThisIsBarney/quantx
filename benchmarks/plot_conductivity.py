"""Current spectral function (proportional to the optical conductivity sigma(omega,T))
from the finite-T current-current correlator C(t;T) = <J(t)J(0)>_beta emitted by
qx_thermal_transport_bench (docs/transport_correlator.csv: columns t, ReC_b<beta>,
ImC_b<beta>, ...). This is the RAW spectral function Re int e^{i omega t} C(t) dt,
not the Kubo-reweighted conductivity (which carries an extra (1-e^{-beta omega})/omega
factor) -- so the f-sum (integral) tracks <J^2>_T and the trend with T is in the total
weight, not a Drude peak.

sigma(omega) ∝ Re integral_{-inf}^{inf} e^{i omega t} C(t) dt. With C(-t) = C(t)^*
(the symmetric correlator), the integral is real and reduces to a cosine/sine
transform of the t>=0 data, windowed with a Gaussian to tame the finite-time
truncation ringing. Pure stdlib (no numpy) so it runs anywhere, matching the
other QuantX plotters.

Run:  python benchmarks/plot_conductivity.py [csv]  ->  docs/conductivity_sigma_wT.svg
"""
import cmath
import csv
import math
import os
import sys

HERE = os.path.dirname(__file__)
CSVP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "docs",
                                                          "transport_correlator.csv")

# --- read t, and Re/Im C(t) per beta ---------------------------------------
with open(CSVP, newline="") as f:
    rows = list(csv.reader(f))
header = rows[0]
betas, re_idx, im_idx = [], [], []
for i, h in enumerate(header):
    if h.startswith("ReC_b"):
        betas.append(float(h[5:])); re_idx.append(i)
    elif h.startswith("ImC_b"):
        im_idx.append(i)
T = [float(r[0]) for r in rows[1:]]
dt = T[1] - T[0]
tmax = T[-1]
C = {b: [complex(float(rows[1 + k][re_idx[bi]]), float(rows[1 + k][im_idx[bi]]))
         for k in range(len(T))]
     for bi, b in enumerate(betas)}

# --- sigma(omega) = 2 * sum_t w(t) Re[e^{i omega t} C(t)] dt (t=0 half-weight) -
ETA = 0.4                                   # Gaussian window width (frac of tmax)
def window(t): return math.exp(-0.5 * (t / (ETA * tmax)) ** 2)
WMAX, NW = 6.0, 240
omegas = [WMAX * m / NW for m in range(NW + 1)]
def sigma(series):
    out = []
    for w in omegas:
        s = 0.5 * series[0].real * window(0.0)   # t=0 counted once
        for k in range(1, len(T)):
            s += window(T[k]) * (cmath.exp(1j * w * T[k]) * series[k]).real
        out.append(2.0 * s * dt)
    return out
S = {b: sigma(C[b]) for b in betas}

# --- SVG ---------------------------------------------------------------------
W, H = 780, 500
L, R, TOP, BOT = 70, 170, 70, 70
PX0, PX1, PY0, PY1 = L, W - R, TOP, H - BOT
ymax = max(max(v for v in S[b]) for b in betas) * 1.08
ymin = min(0.0, min(min(v for v in S[b]) for b in betas))
def xpx(w): return PX0 + w / WMAX * (PX1 - PX0)
def ypx(v): return PY1 - (v - ymin) / (ymax - ymin) * (PY1 - PY0)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
yt = ymin
while yt <= ymax:
    y = ypx(yt)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee" stroke-width="1"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{yt:.1f}</text>')
    yt += max(0.5, round((ymax - ymin) / 6 * 2) / 2)
wt = 0.0
while wt <= WMAX + 1e-9:
    x = xpx(wt)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444" stroke-width="1"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+20}" text-anchor="middle" font-size="11" fill="#444">{wt:.0f}</text>')
    wt += 1.0
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{ypx(0):.1f}" x2="{PX1}" y2="{ypx(0):.1f}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-30}" text-anchor="middle" font-size="13" fill="#222">frequency  &#969;  (J=1)</text>')
s.append(f'<text x="20" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" '
         f'transform="rotate(-90 20 {(PY0+PY1)/2:.0f})">&#963;(&#969;,T)  (arb.)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="26" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">Finite-T current spectral function  &#8733; &#963;(&#969;,T)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="46" text-anchor="middle" font-size="12" fill="#555">Heisenberg chain, FFT of &#10216;J(t)J(0)&#10217;<tspan baseline-shift="sub" font-size="9">&#946;</tspan> (Gaussian window); raw, not Kubo-reweighted</text>')

colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
lx, ly = PX1 + 16, PY0 + 16
s.append(f'<rect x="{lx-8}" y="{ly-14}" width="150" height="{18*len(betas)+10}" fill="white" stroke="#ddd" rx="4"/>')
for bi, b in enumerate(sorted(betas)):
    col = colors[bi % len(colors)]
    pts = " ".join(f"{xpx(omegas[m]):.1f},{ypx(S[b][m]):.1f}" for m in range(len(omegas)))
    s.append(f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="2.4"/>')
    yy = ly + bi * 18
    s.append(f'<line x1="{lx}" y1="{yy}" x2="{lx+22}" y2="{yy}" stroke="{col}" stroke-width="2.4"/>')
    s.append(f'<text x="{lx+28}" y="{yy+4}" font-size="11" fill="#333">T = {1.0/b:.2f}  (&#946;={b:.1f})</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-12}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'Response peaks near the exchange scale &#969;&#8776;J; total weight (f-sum &#8733; &#10216;J&#178;&#10217;<tspan baseline-shift="sub" font-size="8">T</tspan>) decreases on cooling. Absolute scale arbitrary.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "conductivity_sigma_wT.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", os.path.normpath(out))
