"""Singlet pair-field correlations on the doped Hubbard cylinder stripe state,
from a QX_PAIRS showcase log ("PAIRF kind r phi" lines).

Renders docs/hubbard_pairfield.svg: semilog |Phi(r)| for the three bond-orientation
channels -- yy (rung-rung, y-averaged), xx (leg-leg), yx (mixed) -- with marker fill
encoding the SIGN (filled = positive, open = negative). The d-wave fingerprint is
Phi_yx < 0 while Phi_yy > 0 (opposite sign between orthogonal bond orientations);
the dashed line is an exponential fit to the yy tail giving the pairing correlation
length xi. Also writes docs/hubbard_pairfield.csv.

Run: python benchmarks/plot_pairfield.py [logfile] -> docs/hubbard_pairfield.svg
"""
import math, os, re, sys

HERE = os.path.dirname(__file__)
LOG = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "pairfield_16x4_chi1024.log")

head, data = None, {"yy": [], "xx": [], "yx": []}
for line in open(LOG):
    m = re.match(r"PAIRS Lx=(\d+) W=(\d+) chi=(\d+) U=([\d.]+) holes=(\d+) hpin=([\d.]+) "
                 r"nsw=(\d+) x0=(\d+) : E0=(-?[\d.]+) \(([\d.]+)s\)", line)
    if m:
        head = dict(Lx=int(m[1]), W=int(m[2]), chi=int(m[3]), U=float(m[4]),
                    holes=int(m[5]), hpin=float(m[6]), nsw=int(m[7]), x0=int(m[8]),
                    E0=float(m[9]), wall=float(m[10]))
    m = re.match(r"PAIRF (\w+) (\d+) (-?[\d.eE+-]+)", line)
    if m and m[1] in data:
        data[m[1]].append((int(m[2]), float(m[3])))
assert head and data["yy"], f"no PAIRS/PAIRF lines in {LOG}"

with open(os.path.join(HERE, "..", "docs", "hubbard_pairfield.csv"), "w", newline="") as f:
    f.write("kind,r,phi\n")
    for kind in ("yy", "xx", "yx"):
        for r, v in sorted(data[kind]):
            f.write(f"{kind},{r},{v:.10e}\n")

# exponential fit ln|phi_yy| = a - r/xi over r >= 2
fit = [(r, math.log(abs(v))) for r, v in data["yy"] if r >= 2 and v != 0]
xs = [r for r, _ in fit]; ys = [l for _, l in fit]
nf = len(xs); sx = sum(xs); sy = sum(ys)
sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
slope = (nf * sxy - sx * sy) / (nf * sxx - sx * sx)
aa = (sy - slope * sx) / nf
xi = -1.0 / slope if slope < 0 else float("inf")

vals = [abs(v) for k in data for _, v in data[k] if v != 0]
lo = 10 ** math.floor(math.log10(min(vals)))
hi = 10 ** math.ceil(math.log10(max(vals)))
rmax_d = max(r for k in data for r, _ in data[k])

W, H = 720, 520
PX0, PX1, PY0, PY1 = 86, 690, 96, 428
def xpx(r): return PX0 + (r - 0.5) / (rmax_d + 0.5 - 0.5) * (PX1 - PX0)
def ypx(v): return PY1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (PY1 - PY0)

COL = {"yy": "#d62728", "xx": "#1f77b4", "yx": "#7a3fbf"}
LBL = {"yy": "rung&#8211;rung  &#934;yy(r), y-averaged",
       "xx": "leg&#8211;leg  &#934;xx(r)",
       "yx": "rung&#8211;leg  &#934;yx(r)"}

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111">'
         f'Singlet pair-field correlations on the stripe state &#8212; {head["Lx"]}&#215;{head["W"]} Hubbard cylinder, '
         f'U={head["U"]:g}t, &#948;=1/{(head["Lx"]*head["W"])//head["holes"]}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="48" text-anchor="middle" font-size="11.5" fill="#555">'
         f'&#934;(r) = &#10216;&#916;&#8224;(x&#8320;={head["x0"]}) &#916;(x&#8320;+r)&#10217;, singlet pairs on rung/leg bonds; '
         f'&#967;={head["chi"]}, {head["nsw"]} mixer sweeps + {sum(len(v) for v in data.values())} correlator MPOs; '
         f'E&#8320;={head["E0"]:.4f}t ({head["wall"]:.0f}s)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="66" text-anchor="middle" font-size="11" fill="#666">'
         'filled marker = &#934; &gt; 0,  open marker = &#934; &lt; 0.  '
         '<tspan font-weight="bold">d-wave fingerprint: &#934;yx &lt; 0 while &#934;yy &gt; 0</tspan></text>')
# grid: log decades
v = lo
while v <= hi * 1.0001:
    y = ypx(v)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
    e = round(math.log10(v))
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="10.5" fill="#444">10<tspan baseline-shift="super" font-size="8">{e}</tspan></text>')
    v *= 10
for r in range(1, rmax_d + 1):
    x = xpx(r)
    s.append(f'<line x1="{x:.1f}" y1="{PY1}" x2="{x:.1f}" y2="{PY1+5}" stroke="#444"/>')
    s.append(f'<text x="{x:.1f}" y="{PY1+19}" text-anchor="middle" font-size="10.5" fill="#444">{r}</text>')
s.append(f'<line x1="{PX0}" y1="{PY0}" x2="{PX0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{PY1}" x2="{PX1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
# exponential fit line (yy tail)
if math.isfinite(xi):
    r0, r1 = 1.6, rmax_d + 0.2
    y0v, y1v = math.exp(aa + slope * r0), math.exp(aa + slope * r1)
    y0v = min(max(y0v, lo), hi); y1v = min(max(y1v, lo), hi)
    s.append(f'<line x1="{xpx(r0):.1f}" y1="{ypx(y0v):.1f}" x2="{xpx(r1):.1f}" y2="{ypx(y1v):.1f}" '
             f'stroke="#c99" stroke-width="1.3" stroke-dasharray="6,4"/>')
    s.append(f'<text x="{xpx(rmax_d*0.62):.1f}" y="{ypx(math.exp(aa+slope*rmax_d*0.62))-10:.1f}" '
             f'font-size="11" fill="#a66">&#8764; e<tspan baseline-shift="super" font-size="8.5">&#8722;r/{xi:.1f}</tspan></text>')
for kind in ("yy", "xx", "yx"):
    pts = sorted(data[kind])
    poly = " ".join(f"{xpx(r):.1f},{ypx(abs(v)):.1f}" for r, v in pts if v != 0)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{COL[kind]}" stroke-width="1.8" opacity="0.85"/>')
    for r, v in pts:
        if v == 0: continue
        fill = COL[kind] if v > 0 else "white"
        s.append(f'<circle cx="{xpx(r):.1f}" cy="{ypx(abs(v)):.1f}" r="4.2" fill="{fill}" '
                 f'stroke="{COL[kind]}" stroke-width="1.8"/>')
# legend
ly = PY0 + 8
for kind in ("yy", "xx", "yx"):
    s.append(f'<line x1="{PX1-198}" y1="{ly}" x2="{PX1-172}" y2="{ly}" stroke="{COL[kind]}" stroke-width="2.2"/>')
    s.append(f'<circle cx="{PX1-185}" cy="{ly}" r="4" fill="{COL[kind]}" stroke="{COL[kind]}" stroke-width="1.6"/>')
    s.append(f'<text x="{PX1-164}" y="{ly+4}" font-size="11.5" fill="#333">{LBL[kind]}</text>')
    ly += 21
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-46}" text-anchor="middle" font-size="12.5" fill="#222">'
         'bond separation r along the cylinder axis</text>')
s.append(f'<text x="26" y="{(PY0+PY1)/2:.0f}" text-anchor="middle" font-size="12.5" fill="#222" '
         f'transform="rotate(-90 26 {(PY0+PY1)/2:.0f})">|&#934;(r)| = |&#10216;&#916;&#8224;&#916;&#10217;|  (log scale)</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-24}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'Each point is one 7-wide correlator MPO evaluated on the converged block-DMRG ground state '
         '(one GS run + per-observable environment rebuilds); machinery ED-validated to 1.9e-9 (qx_hubbard_test PAIR-FIELD).</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "hubbard_pairfield.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
sgn = lambda k, r: next((("+" if v > 0 else "-") for rr, v in data[k] if rr == r), "?")
print("wrote", os.path.normpath(out),
      f"| xi_pair={xi:.2f} | signs at r=2: yy{sgn('yy',2)} xx{sgn('xx',2)} yx{sgn('yx',2)}")
