"""Dynamical spin structure factor S(k,omega,T) from the finite-T correlator
C_{j,l0}(t;T) emitted by qx_thermal_skw_emit (docs/skw_b<beta>.csv).

S(k,omega) = 2 Re sum_{t>=0} w(t) e^{i omega t} sum_x e^{-i k x} C_{l0+x,l0}(t) dt
(t=0 half-weighted), Gaussian-windowed in time, clamped to >=0 for the colormap.
Pure stdlib (no numpy). Renders a k-omega heatmap; the spinon continuum's lower
edge omega = (pi/2)|sin k| (des Cloizeaux-Pearson) is overlaid for reference.

Run:  python benchmarks/plot_skw.py docs/skw_b0.5.csv  ->  docs/skw_b0.5.svg
"""
import cmath
import csv
import math
import os
import sys

CSVP = sys.argv[1] if len(sys.argv) > 1 else "docs/skw_b1.0.csv"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "spinhalf"   # "spinhalf" | "spin1" | "hubbard"
with open(CSVP, newline="") as f:
    rows = list(csv.reader(f))
meta = {kv.split("=")[0]: kv.split("=")[1] for kv in rows[0][1:] if "=" in kv}
l0 = int(meta["l0"]); N = int(meta["n_phys"]); beta = float(meta.get("beta", "0"))
T = [float(r[0]) for r in rows[1:]]
dt = T[1] - T[0]; tmax = T[-1]
# C[t][j] complex
C = []
for r in rows[1:]:
    re = [float(r[1 + j]) for j in range(N)]
    im = [float(r[1 + N + j]) for j in range(N)]
    C.append([complex(re[j], im[j]) for j in range(N)])

ETA = 0.4
def window(t): return math.exp(-0.5 * (t / (ETA * tmax)) ** 2)

NK, NW, WMAX = 44, 80, 6.0
ks = [math.pi * a / NK for a in range(NK + 1)]            # 0..pi
ws = [WMAX * b / NW for b in range(NW + 1)]               # 0..WMAX

# Ck(t) = sum_x e^{-ikx} C_{l0+x}(t)
def skw():
    grid = [[0.0] * (NK + 1) for _ in range(NW + 1)]
    Ck = [[0j] * (NK + 1) for _ in range(len(T))]
    for ti in range(len(T)):
        for ki, k in enumerate(ks):
            acc = 0j
            for j in range(N):
                acc += cmath.exp(-1j * k * (j - l0)) * C[ti][j]
            Ck[ti][ki] = acc
    vmax = 0.0
    for wi, w in enumerate(ws):
        for ki in range(NK + 1):
            s = 0.5 * Ck[0][ki].real * window(0.0)
            for ti in range(1, len(T)):
                s += window(T[ti]) * (cmath.exp(1j * w * T[ti]) * Ck[ti][ki]).real
            v = max(0.0, 2.0 * s * dt)
            grid[wi][ki] = v
            vmax = max(vmax, v)
    return grid, (vmax or 1.0)

grid, vmax = skw()

# perceptual-ish colormap (dark-blue -> teal -> green -> yellow)
def color(u):
    u = max(0.0, min(1.0, u))
    stops = [(0.0, (12, 12, 60)), (0.35, (24, 110, 130)),
             (0.65, (60, 180, 90)), (1.0, (250, 232, 90))]
    for i in range(len(stops) - 1):
        a, ca = stops[i]; b, cb = stops[i + 1]
        if u <= b:
            f = (u - a) / (b - a) if b > a else 0
            return tuple(int(ca[c] + f * (cb[c] - ca[c])) for c in range(3))
    return stops[-1][1]

W, H = 700, 500
L, Rm, TOP, BOT = 70, 90, 60, 64
PX0, PX1, PY0, PY1 = L, W - Rm, TOP, H - BOT
def xpx(k): return PX0 + k / math.pi * (PX1 - PX0)
def ypx(w): return PY1 - w / WMAX * (PY1 - PY0)
cw = (PX1 - PX0) / (NK + 1)
ch = (PY1 - PY0) / (NW + 1)

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">']
s.append(f'<rect width="{W}" height="{H}" fill="white"/>')
for wi in range(NW + 1):
    for ki in range(NK + 1):
        r, g, b = color(grid[wi][ki] / vmax)
        x = PX0 + ki * cw; y = PY1 - (wi + 1) * ch
        s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw+0.6:.1f}" height="{ch+0.6:.1f}" fill="rgb({r},{g},{b})"/>')
if MODEL == "hubbard":
    # Mott-insulator SPIN response: gapless spinons at the superexchange scale.
    # Strong-coupling lower edge (pi/2) J_eff |sin k|, J_eff = 4 t^2 / U (U=4t -> 1).
    Jeff = 1.0
    low = " ".join(f"{xpx(k):.1f},{ypx((math.pi/2)*Jeff*abs(math.sin(k))):.1f}" for k in ks)
    s.append(f'<polyline points="{low}" fill="none" stroke="#fff" stroke-width="1.3" stroke-dasharray="4,3" opacity="0.85"/>')
elif MODEL == "spin1":
    # Haldane single-magnon branch w(k) = sqrt(Delta^2 + B(1+cos k)): gapped minimum
    # Delta ~ 0.41 J at k=pi (the Haldane gap), rising to ~2.7 J at k=0.
    DELTA, B = 0.41, 3.56
    mag = " ".join(f"{xpx(k):.1f},{ypx(math.sqrt(DELTA**2 + B*(1+math.cos(k)))):.1f}" for k in ks)
    s.append(f'<polyline points="{mag}" fill="none" stroke="#fff" stroke-width="1.4" stroke-dasharray="4,3" opacity="0.85"/>')
    kpi = xpx(math.pi)
    s.append(f'<line x1="{kpi-18:.1f}" y1="{ypx(DELTA):.1f}" x2="{kpi:.1f}" y2="{ypx(DELTA):.1f}" stroke="#ff4444" stroke-width="1.6"/>')
    s.append(f'<text x="{kpi-22:.1f}" y="{ypx(DELTA)+4:.1f}" text-anchor="end" font-size="10" fill="#ff6666">&#916;&#8776;0.41</text>')
else:
    # des Cloizeaux-Pearson lower edge  w = (pi/2) sin k  and upper edge pi sin(k/2)
    low = " ".join(f"{xpx(k):.1f},{ypx((math.pi/2)*abs(math.sin(k))):.1f}" for k in ks)
    up = " ".join(f"{xpx(k):.1f},{ypx(math.pi*abs(math.sin(k/2))):.1f}" for k in ks)
    s.append(f'<polyline points="{low}" fill="none" stroke="#fff" stroke-width="1.3" stroke-dasharray="4,3" opacity="0.8"/>')
    s.append(f'<polyline points="{up}" fill="none" stroke="#fff" stroke-width="1.1" stroke-dasharray="2,3" opacity="0.6"/>')
# axes
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
for frac, lab in [(0, "0"), (0.5, "&#960;/2"), (1.0, "&#960;")]:
    x = PX0 + frac * (PX1 - PX0)
    s.append(f'<text x="{x:.1f}" y="{PY1+18}" text-anchor="middle" font-size="11" fill="#444">{lab}</text>')
for wv in range(0, int(WMAX) + 1, 1):
    y = ypx(wv)
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" fill="#444">{wv}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-26}" text-anchor="middle" font-size="13" fill="#222">momentum  k</text>')
s.append(f'<text x="20" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="13" fill="#222" '
         f'transform="rotate(-90 20 {(PY0+PY1)/2:.0f})">frequency  &#969;  (J=1)</text>')
T_lab = ("inf" if beta == 0 else f"{1.0/beta:.2f}")
_chain = ("Hubbard chain (U=4t, half filling) &#8212; spin response" if MODEL == "hubbard"
          else "spin-1 Haldane chain" if MODEL == "spin1" else "Heisenberg chain")
_ref = ("dashed = spinon edge (&#960;/2)J_eff|sin k|, J_eff=4t&#178;/U: the Mott spin sector is GAPLESS"
        if MODEL == "hubbard" else
        "dashed = single-magnon branch, gapped at k=&#960; (Haldane gap &#916;&#8776;0.41)"
        if MODEL == "spin1" else "dashed = des Cloizeaux&#8211;Pearson two-spinon edges")
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="24" text-anchor="middle" font-size="15" font-weight="bold" fill="#111">S(k,&#969;,T) &#8212; {_chain}, T = {T_lab} (&#946;={beta:.1f})</text>')
_path = "d&gt;2 fast path" if MODEL == "spin1" else "block path"
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="42" text-anchor="middle" font-size="11" fill="#666">purification + real-time block-TEBD ({_path}), N={N} (demo); {_ref}</text>')
# colorbar
cbx = PX1 + 16
for i in range(60):
    u = i / 59.0; r, g, b = color(u)
    y = PY1 - (i + 1) * (PY1 - PY0) / 60
    s.append(f'<rect x="{cbx}" y="{y:.1f}" width="14" height="{(PY1-PY0)/60+0.6:.1f}" fill="rgb({r},{g},{b})"/>')
s.append(f'<text x="{cbx+7}" y="{PY0-6}" text-anchor="middle" font-size="9" fill="#666">S (max)</text>')
s.append(f'<text x="{cbx+7}" y="{PY1+12}" text-anchor="middle" font-size="9" fill="#666">0</text>')
s.append('</svg>')

out = os.path.splitext(CSVP)[0] + ".svg"
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", os.path.normpath(out))
