"""t-t'-U evolution: stripes and pairing vs the diagonal hopping t'.

Reads three QX_PAIRS/QX_STRIPES logs (t' = -0.25, 0, +0.25; the t'=0 profile
comes from the stripes log since the original t'=0 pair run predates PROF
emission) and renders docs/hubbard_ttprime.svg:

  top    -- rung-averaged hole density 1-n(x) per t': how the stripe crystal
            responds to the diagonal hopping;
  bottom -- |Phi_yy(r)| (semilog) per t' with exponential fits: how the pairing
            correlation length responds.

Also writes docs/hubbard_ttprime.csv (kind = prof/pairf rows for all t').

Run: python benchmarks/plot_ttprime.py -> docs/hubbard_ttprime.svg
"""
import math, os, re, sys

HERE = os.path.dirname(__file__)
SOURCES = [   # (tp, [logs to scan -- first PROF/PAIRF hit per kind wins])
    (-0.25, ["pairfield_16x4_tp-025.log"]),
    ( 0.00, ["pairfield_16x4_chi1024.log", "stripes_16x4_chi1024.log"]),
    (+0.25, ["pairfield_16x4_tp+025.log"]),
]

HEAD_RE = re.compile(
    r"(?:PAIRS|STRIPES) Lx=(\d+) W=(\d+) chi=(\d+) U=(-?[\d.]+) holes=(\d+) "
    r"hpin=(-?[\d.]+) nsw=(\d+)(?: x0=(\d+))?(?: tp=(-?[\d.]+))? : E0=(-?[\d.]+) \(([\d.]+)s\)")

def load(tp, logs):
    out = {"tp": tp, "prof": {}, "yy": {}, "E0": None, "wall": 0.0, "head": None}
    for name in logs:
        path = os.path.join(HERE, name)
        if not os.path.exists(path):
            return None
        for line in open(path):
            m = HEAD_RE.match(line)
            if m and out["head"] is None:
                out["head"] = dict(Lx=int(m[1]), W=int(m[2]), chi=int(m[3]),
                                   U=float(m[4]), holes=int(m[5]))
                out["E0"] = float(m[10]); out["wall"] = float(m[11])
            m = re.match(r"PROF (\d+) (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)", line)
            if m and int(m[1]) not in out["prof"]:
                out["prof"][int(m[1])] = (int(m[2]), int(m[3]), float(m[4]))
            m = re.match(r"PAIRF yy (\d+) (-?[\d.eE+-]+)", line)
            if m and int(m[1]) not in out["yy"]:
                out["yy"][int(m[1])] = float(m[2])
    return out if out["prof"] and out["yy"] else None

runs = []
for tp, logs in SOURCES:
    r = load(tp, logs)
    if r is None:
        sys.exit(f"missing/incomplete logs for tp={tp}: {logs}")
    runs.append(r)
hd = runs[0]["head"]
Lx, Wd = hd["Lx"], hd["W"]

with open(os.path.join(HERE, "..", "docs", "hubbard_ttprime.csv"), "w", newline="") as f:
    f.write("tp,kind,a,b,val\n")
    for r in runs:
        for i, (x, y, nv) in sorted(r["prof"].items()):
            f.write(f"{r['tp']},prof,{x},{y},{nv:.8f}\n")
        for rr, v in sorted(r["yy"].items()):
            f.write(f"{r['tp']},pairf_yy,{rr},0,{v:.10e}\n")

def fit_xi(yy):
    fit = [(r, math.log(abs(v))) for r, v in yy.items() if r >= 2 and v != 0]
    if len(fit) < 2: return float("inf"), 0.0, 0.0
    xs = [a for a, _ in fit]; ys = [b for _, b in fit]
    nf, sx, sy = len(xs), sum(xs), sum(ys)
    sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
    sl = (nf*sxy - sx*sy) / (nf*sxx - sx*sx)
    aa = (sy - sl*sx) / nf
    return (-1.0/sl if sl < 0 else float("inf")), sl, aa

COL = {-0.25: "#1f77b4", 0.0: "#777777", 0.25: "#d62728"}
W, H = 760, 640
PX0, PX1 = 84, 726
TY0, TY1 = 96, 300          # top: hole density
BY0, BY1 = 372, 576         # bottom: |Phi_yy| semilog

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111">'
         f'Stripes and pairing vs diagonal hopping t&#8242; &#8212; {Lx}&#215;{Wd} Hubbard cylinder, '
         f'U={hd["U"]:g}t, &#948;=1/{(Lx*Wd)//hd["holes"]}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="48" text-anchor="middle" font-size="11.5" fill="#555">'
         f't-t&#8242;-U model, &#967;={hd["chi"]}, 20 mixer sweeps per point, staggered edge pinning; '
         'same protocol at every t&#8242; (only the diagonal hopping changes)</text>')

# top panel: hole density
hmax = max((1.0 - nv) for r in runs for (_, _, nv) in r["prof"].values()) * 1.15
cw = (PX1 - PX0) / Lx
def xctr(x): return PX0 + (x + 0.5) * cw
def typx(v): return TY1 - v / hmax * (TY1 - TY0)
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY0-12}" text-anchor="middle" font-size="12.5" fill="#222" font-weight="bold">'
         'rung-averaged hole density  1 &#8722; n(x)</text>')
g = 0.0
while g <= hmax:
    yv = typx(g)
    s.append(f'<line x1="{PX0}" y1="{yv:.1f}" x2="{PX1}" y2="{yv:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-7}" y="{yv+4:.1f}" text-anchor="end" font-size="10.5" fill="#444">{g:.2f}</text>')
    g += 0.05
mean_h = hd["holes"] / (Lx * Wd)
s.append(f'<line x1="{PX0}" y1="{typx(mean_h):.1f}" x2="{PX1}" y2="{typx(mean_h):.1f}" '
         f'stroke="#bbb" stroke-dasharray="5,4"/>')
for r in runs:
    hx = [0.0] * Lx
    for (x, y, nv) in r["prof"].values(): hx[x] += (1.0 - nv) / Wd
    c = COL[r["tp"]]
    poly = " ".join(f"{xctr(x):.1f},{typx(hx[x]):.1f}" for x in range(Lx))
    s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="2.2" opacity="0.9"/>')
    for x in range(Lx):
        s.append(f'<circle cx="{xctr(x):.1f}" cy="{typx(hx[x]):.1f}" r="3.2" fill="{c}"/>')
for x in range(Lx):
    s.append(f'<text x="{xctr(x):.1f}" y="{TY1+16:.1f}" text-anchor="middle" font-size="9.5" fill="#444">{x}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY1+32:.1f}" text-anchor="middle" font-size="11.5" fill="#222">column x</text>')
s.append(f'<line x1="{PX0}" y1="{TY0}" x2="{PX0}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{TY1}" x2="{PX1}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')

# bottom panel: |Phi_yy(r)| semilog
vals = [abs(v) for r in runs for v in r["yy"].values() if v != 0]
lo = 10 ** math.floor(math.log10(min(vals)))
hi = 10 ** math.ceil(math.log10(max(vals)))
rmax_d = max(rr for r in runs for rr in r["yy"])
def bxpx(rr): return PX0 + (rr - 0.5) / rmax_d * (PX1 - PX0)
def bypx(v): return BY1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (BY1 - BY0)
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-12}" text-anchor="middle" font-size="12.5" fill="#222" font-weight="bold">'
         'singlet pair-field correlation  |&#934;yy(r)|  (rung&#8211;rung, y-averaged; log scale)</text>')
v = lo
while v <= hi * 1.0001:
    yv = bypx(v)
    s.append(f'<line x1="{PX0}" y1="{yv:.1f}" x2="{PX1}" y2="{yv:.1f}" stroke="#eee"/>')
    e = round(math.log10(v))
    s.append(f'<text x="{PX0-7}" y="{yv+4:.1f}" text-anchor="end" font-size="10.5" fill="#444">10<tspan baseline-shift="super" font-size="8">{e}</tspan></text>')
    v *= 10
for rr in range(1, rmax_d + 1):
    xv = bxpx(rr)
    s.append(f'<line x1="{xv:.1f}" y1="{BY1}" x2="{xv:.1f}" y2="{BY1+5}" stroke="#444"/>')
    s.append(f'<text x="{xv:.1f}" y="{BY1+18}" text-anchor="middle" font-size="10" fill="#444">{rr}</text>')
for r in runs:
    c = COL[r["tp"]]
    xi, sl, aa = fit_xi(r["yy"])
    if math.isfinite(xi):
        r0, r1 = 1.7, rmax_d + 0.2
        s.append(f'<line x1="{bxpx(r0):.1f}" y1="{bypx(min(max(math.exp(aa+sl*r0),lo),hi)):.1f}" '
                 f'x2="{bxpx(r1):.1f}" y2="{bypx(min(max(math.exp(aa+sl*r1),lo),hi)):.1f}" '
                 f'stroke="{c}" stroke-width="1" stroke-dasharray="5,4" opacity="0.55"/>')
    pts = sorted(r["yy"].items())
    poly = " ".join(f"{bxpx(a):.1f},{bypx(abs(b)):.1f}" for a, b in pts if b != 0)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="2" opacity="0.9"/>')
    for a, b in pts:
        if b == 0: continue
        fill = c if b > 0 else "white"
        s.append(f'<circle cx="{bxpx(a):.1f}" cy="{bypx(abs(b)):.1f}" r="3.8" fill="{fill}" stroke="{c}" stroke-width="1.6"/>')
    r["xi"] = xi
s.append(f'<line x1="{PX0}" y1="{BY0}" x2="{PX0}" y2="{BY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{BY1}" x2="{PX1}" y2="{BY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY1+36:.1f}" text-anchor="middle" font-size="11.5" fill="#222">bond separation r</text>')
# legend (shared)
ly = BY0 + 10
for r in runs:
    c = COL[r["tp"]]
    xi_s = f'{r["xi"]:.1f}' if math.isfinite(r["xi"]) else "&#8734;"
    s.append(f'<line x1="{PX1-232}" y1="{ly}" x2="{PX1-206}" y2="{ly}" stroke="{c}" stroke-width="2.4"/>')
    s.append(f'<text x="{PX1-198}" y="{ly+4}" font-size="11.5" fill="#333">t&#8242; = {r["tp"]:+.2f}   '
             f'(&#958;<tspan baseline-shift="sub" font-size="8.5">pair</tspan> &#8776; {xi_s})</text>')
    ly += 20
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-14}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'open marker = &#934;yy &lt; 0; dashes = exponential fits to the r &#8805; 2 tail. '
         'Same engine, same protocol; only t&#8242; varies.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "hubbard_ttprime.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out), "|",
      " ".join(f"tp={r['tp']:+.2f}: E0={r['E0']:.4f} xi={r['xi']:.2f}" for r in runs))
