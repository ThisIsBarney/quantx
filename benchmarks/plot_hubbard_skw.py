"""Spin-charge separation of the 1D Hubbard chain: S_spin(k,w) and S_charge(k,w)
side by side, from the T=0 correlators emitted by qx_hubbard_spectra
(docs/hubbard_spin_correlator.csv + docs/hubbard_charge_correlator.csv).

S(k,w) = 2 Re sum_{t>=0} w(t) e^{i w t} sum_x e^{-i k x} C_{l0+x,l0}(t) dt
(t=0 half-weighted), Gaussian-windowed in time, clamped >=0. Pure stdlib.

The physics in one picture: the SPIN panel is gapless at k=pi (spinon continuum,
velocity v_s), the CHARGE panel opens the Mott gap (holon continuum pushed up to
~Delta_c) -- one electron, two separated collective excitations.

Run:  python benchmarks/plot_hubbard_skw.py  ->  docs/hubbard_spin_charge_skw.svg
"""
import cmath, csv, math, os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")

def load(path):
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    meta = {kv.split("=")[0]: kv.split("=")[1] for kv in rows[0][1:] if "=" in kv}
    l0 = int(meta["l0"]); N = int(meta["n_phys"]); U = float(meta.get("U", "0"))
    T = [float(r[0]) for r in rows[1:]]
    C = []
    for r in rows[1:]:
        re = [float(r[1 + j]) for j in range(N)]
        im = [float(r[1 + N + j]) for j in range(N)]
        C.append([complex(re[j], im[j]) for j in range(N)])
    return l0, N, U, T, C

ETA = 0.4
NK, NW, WMAX = 40, 80, 6.0
ks = [math.pi * a / NK for a in range(NK + 1)]
ws = [WMAX * b / NW for b in range(NW + 1)]

def skw(l0, N, T, C):
    dt = T[1] - T[0]; tmax = T[-1]
    win = lambda t: math.exp(-0.5 * (t / (ETA * tmax)) ** 2)
    Ck = [[sum(cmath.exp(-1j * k * (j - l0)) * C[ti][j] for j in range(N))
           for k in ks] for ti in range(len(T))]
    grid = [[0.0] * (NK + 1) for _ in range(NW + 1)]
    vmax = 0.0
    for wi, w in enumerate(ws):
        for ki in range(NK + 1):
            s = 0.5 * Ck[0][ki].real * win(0.0)
            for ti in range(1, len(T)):
                s += win(T[ti]) * (cmath.exp(1j * w * T[ti]) * Ck[ti][ki]).real
            v = max(0.0, 2.0 * s * dt)
            grid[wi][ki] = v
            vmax = max(vmax, v)
    return grid, (vmax or 1.0)

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

l0, N, U, T, Cs = load(os.path.join(DOCS, "hubbard_spin_correlator.csv"))
_,  _, _, _, Cc = load(os.path.join(DOCS, "hubbard_charge_correlator.csv"))
gS, vS = skw(l0, N, T, Cs)
gC, vC = skw(l0, N, T, Cc)

W, H = 980, 520
PW, GAP, L0X, TOP, BOT = 380, 60, 74, 72, 66
PY0, PY1 = TOP, H - BOT
panels = [(L0X, gS, vS, "SPIN:  S_s(k,&#969;)  &#8212; gapless spinons"),
          (L0X + PW + GAP, gC, vC, "CHARGE:  S_c(k,&#969;)  &#8212; Mott gap")]
s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
     f'viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{W/2:.0f}" y="26" text-anchor="middle" font-size="16" font-weight="bold" fill="#111">'
         f'Spin-charge separation &#8212; Hubbard chain, U={U:g}t, half filling (T=0)</text>')
s.append(f'<text x="{W/2:.0f}" y="46" text-anchor="middle" font-size="11.5" fill="#555">'
         f'fast block path (packed two-U(1)), N={N}, real-time TEBD after (2Sz)/(n&#8722;1) source at l0={l0}; ED-validated correlators</text>')

for PX0, grid, vmax, title in panels:
    PX1 = PX0 + PW
    cw = PW / (NK + 1); ch = (PY1 - PY0) / (NW + 1)
    for wi in range(NW + 1):
        for ki in range(NK + 1):
            r, g, b = color(grid[wi][ki] / vmax)
            x = PX0 + ki * cw; y = PY1 - (wi + 1) * ch
            s.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{cw+0.6:.1f}" height="{ch+0.6:.1f}" fill="rgb({r},{g},{b})"/>')
    xpx = lambda k: PX0 + k / math.pi * PW
    ypx = lambda w: PY1 - w / WMAX * (PY1 - PY0)
    if "SPIN" in title:
        # strong-coupling spinon lower edge ~ (pi/2) J_eff |sin k|, J_eff = 4t^2/U
        Jeff = 4.0 / U
        edge = " ".join(f"{xpx(k):.1f},{ypx((math.pi/2)*Jeff*abs(math.sin(k))):.1f}" for k in ks)
        s.append(f'<polyline points="{edge}" fill="none" stroke="#fff" stroke-width="1.3" stroke-dasharray="4,3" opacity="0.85"/>')
        s.append(f'<text x="{xpx(2.0):.1f}" y="{ypx((math.pi/2)*Jeff)+16:.1f}" font-size="10" fill="#fff" opacity="0.9">(&#960;/2)J_eff|sin k|</text>')
    else:
        # optical (q=pi) Mott gap marker from the Bethe ansatz, U=4t: Delta_c ~ 1.29t
        DC = 1.286
        s.append(f'<line x1="{xpx(math.pi)-30:.1f}" y1="{ypx(DC):.1f}" x2="{xpx(math.pi):.1f}" y2="{ypx(DC):.1f}" stroke="#ff5555" stroke-width="1.8"/>')
        s.append(f'<text x="{xpx(math.pi)-34:.1f}" y="{ypx(DC)+4:.1f}" text-anchor="end" font-size="10.5" fill="#ff7777">&#916;_c&#8776;{DC:.2f}</text>')
    s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
    s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
    for frac, lab in [(0, "0"), (0.5, "&#960;/2"), (1.0, "&#960;")]:
        s.append(f'<text x="{PX0+frac*PW:.1f}" y="{PY1+17}" text-anchor="middle" font-size="11" fill="#444">{lab}</text>')
    for wv in range(0, int(WMAX) + 1, 1):
        s.append(f'<text x="{PX0-7}" y="{ypx(wv)+4:.1f}" text-anchor="end" font-size="10.5" fill="#444">{wv}</text>')
    s.append(f'<text x="{PX0+PW/2:.0f}" y="{PY1+38}" text-anchor="middle" font-size="12.5" fill="#222">momentum  k</text>')
    s.append(f'<text x="{PX0+PW/2:.0f}" y="{TOP-10}" text-anchor="middle" font-size="13" font-weight="bold" fill="#222">{title}</text>')
s.append(f'<text x="{L0X-52}" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="12.5" fill="#222" '
         f'transform="rotate(-90 {L0X-52} {(PY0+PY1)/2:.0f})">frequency  &#969;  (t=1)</text>')
s.append(f'<text x="{W/2:.0f}" y="{H-12}" text-anchor="middle" font-size="10" fill="#777" font-style="italic">'
         'Same ground state, two probes: the spin response is gapless at k=&#960; (spinons, J_eff=4t&#178;/U) '
         'while the charge response is pushed above the Mott gap (holons) &#8212; spin&#8211;charge separation.</text>')
s.append('</svg>')

out = os.path.join(DOCS, "hubbard_spin_charge_skw.svg")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(s))
print("wrote", os.path.normpath(out))
