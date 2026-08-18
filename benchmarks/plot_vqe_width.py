"""Width scan: chain vs ladder vs cylinder, all in ONE representation.

The earlier geometry figure (plot_vqe_geometry.py) compared a d=4 chain against a
d=16 rung-fused ladder. Those two carry their entropy across different cuts, so part
of any difference between them is representation, not physics. This figure removes
that confound: docs/vqe_width.csv holds W=1 Lx=6, W=2 Lx=3 and W=3 Lx=2 -- all
site-split (d=4), all n=6 sites and 12 qubits, all cut at the same bond, all lossless
at chi=64, all through the same driver. Only the connectivity graph changes.

  left  -- relative energy error |E_VQE - E_0| / |E_0|, each width at its own optimum;
  right -- S_mid, the entropy across the balanced 6|6-qubit cut (the comparable one).

Run: python benchmarks/plot_vqe_width.py -> docs/vqe_width.svg
"""
import csv, math, os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")

rows = list(csv.DictReader(open(os.path.join(DOCS, "vqe_width.csv"))))
D = {(int(r["W"]), int(r["p"])): r for r in rows}
widths = sorted({int(r["W"]) for r in rows})
depths = sorted({int(r["p"]) for r in rows})

NAME = {1: "chain W=1 (Lx=6)", 2: "ladder W=2 (Lx=3)", 3: "cylinder W=3 (Lx=2)"}
COL = {1: "#1f77b4", 2: "#ff7f0e", 3: "#d62728"}


def err(w, p):
    r = D.get((w, p))
    if not r:
        return None
    return abs(float(r["E_vqe"]) - float(r["E_exact"])) / abs(float(r["E_exact"]))


def smid(w, p):
    r = D.get((w, p))
    return float(r["S_mid"]) if r else None


W, H = 830, 452
PY0, PY1 = 116, 340
LX0, LX1 = 76, 396
RX0, RX1 = 478, 798

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111">'
         'Width, not qubit count, is what a fermionic VQE pays for</text>')
s.append(f'<text x="{W/2:.0f}" y="48" text-anchor="middle" font-size="11.5" fill="#555">'
         'chain / ladder / cylinder at 12 qubits &#8212; same representation (d=4), same MPS length, '
         'same balanced cut, all lossless at &#967;=64</text>')
nang = {w: int(D[(w, depths[0])]["nangle"]) // depths[0] for w in widths if (w, depths[0]) in D}
s.append(f'<text x="{W/2:.0f}" y="66" text-anchor="middle" font-size="10.5" fill="#777">'
         'each width optimised at its OWN optimum, identical search protocol; the wider ones get MORE angles per layer ('
         + ', '.join(f'W={w}: {nang[w]}' for w in widths if w in nang) + ')</text>')

# ---- left: relative energy error (log) ----
errs = [e for w in widths for p in depths if (e := err(w, p))]
lo = 10 ** math.floor(math.log10(min(errs)))
hi = 10 ** math.ceil(math.log10(max(errs)))
def lxpx(p): return LX0 + (p - depths[0]) / max(1, depths[-1] - depths[0]) * (LX1 - LX0)
def lypx(v): return PY1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (PY1 - PY0)
s.append(f'<text x="{(LX0+LX1)/2:.0f}" y="{PY0-14}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#222">'
         'relative energy error</text>')
v = lo
while v <= hi * 1.0001:
    y = lypx(v)
    s.append(f'<line x1="{LX0}" y1="{y:.1f}" x2="{LX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{LX0-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">10<tspan baseline-shift="super" font-size="8">{round(math.log10(v))}</tspan></text>')
    v *= 10
for w in widths:
    pts = [(p, err(w, p)) for p in depths if err(w, p)]
    if not pts:
        continue
    s.append('<polyline points="' + " ".join(f"{lxpx(p):.1f},{lypx(e):.1f}" for p, e in pts) +
             f'" fill="none" stroke="{COL[w]}" stroke-width="2.2"/>')
    for p, e in pts:
        s.append(f'<circle cx="{lxpx(p):.1f}" cy="{lypx(e):.1f}" r="3.3" fill="{COL[w]}"/>')

# ---- right: balanced-cut entropy (linear) ----
Ss = [x for w in widths for p in depths if (x := smid(w, p)) is not None]
smax = max(Ss) * 1.18
def rxpx(p): return RX0 + (p - depths[0]) / max(1, depths[-1] - depths[0]) * (RX1 - RX0)
def rypx(v): return PY1 - v / smax * (PY1 - PY0)
s.append(f'<text x="{(RX0+RX1)/2:.0f}" y="{PY0-14}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#222">'
         'entropy across the balanced 6|6-qubit cut</text>')
g = 0.0
while g <= smax:
    y = rypx(g)
    s.append(f'<line x1="{RX0}" y1="{y:.1f}" x2="{RX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{RX0-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">{g:.1f}</text>')
    g += 0.5
for w in widths:
    pts = [(p, smid(w, p)) for p in depths if smid(w, p) is not None]
    if not pts:
        continue
    s.append('<polyline points="' + " ".join(f"{rxpx(p):.1f},{rypx(e):.1f}" for p, e in pts) +
             f'" fill="none" stroke="{COL[w]}" stroke-width="2.2"/>')
    for p, e in pts:
        s.append(f'<circle cx="{rxpx(p):.1f}" cy="{rypx(e):.1f}" r="3.3" fill="{COL[w]}"/>')

for x0, x1, xf in ((LX0, LX1, lxpx), (RX0, RX1, rxpx)):
    for p in depths:
        s.append(f'<line x1="{xf(p):.1f}" y1="{PY1}" x2="{xf(p):.1f}" y2="{PY1+5}" stroke="#444"/>')
        s.append(f'<text x="{xf(p):.1f}" y="{PY1+19}" text-anchor="middle" font-size="10" fill="#444">{p}</text>')
    s.append(f'<line x1="{x0}" y1="{PY0}" x2="{x0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
    s.append(f'<line x1="{x0}" y1="{PY1}" x2="{x1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{PY1+38:.1f}" text-anchor="middle" font-size="11.5" fill="#222">circuit depth p</text>')

ly = PY0 + 8
for w in widths:
    s.append(f'<line x1="{LX0+16}" y1="{ly}" x2="{LX0+40}" y2="{ly}" stroke="{COL[w]}" stroke-width="2.6"/>')
    s.append(f'<circle cx="{LX0+28}" cy="{ly}" r="3.3" fill="{COL[w]}"/>')
    s.append(f'<text x="{LX0+47}" y="{ly+4}" font-size="11" fill="#333">{NAME[w]}</text>')
    ly += 18

pmax = depths[-1]
# State what the data shows, not what the figure was built to show. The energy claim
# and the entropy claim are separate, and only one of them has to come out monotone --
# the caption is generated from the numbers so it cannot drift from them.
es = [err(w, pmax) for w in widths]
ss = [smid(w, pmax) for w in widths]
if all(es) and all(x is not None for x in ss):
    mono_S = all(ss[i] < ss[i+1] for i in range(len(ss)-1))
    mono_e = all(es[i] < es[i+1] for i in range(len(es)-1))
    head = (f'At depth {pmax} the error grows monotonically with width, '
            f'&#215;{es[-1]/es[0]:.1f} from W=1 to W={widths[-1]}.'
            if mono_e else
            f'At depth {pmax} the widest lattice sits &#215;{es[-1]/es[0]:.1f} from the chain\'s error '
            f'(not monotonic in W).')
    tail = ('Entropy across the balanced cut orders the same way.' if mono_S else
            'Entropy across the balanced cut does NOT order the same way &#8212; peak at W='
            + str(widths[ss.index(max(ss))]) + '.')
    s.append(f'<text x="{W/2:.0f}" y="{H-28}" text-anchor="middle" font-size="11" fill="#0a7d28" font-weight="bold">'
             f'{head} {tail}</text>')
s.append(f'<text x="{W/2:.0f}" y="{H-10}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'Same qubits, same simulator, same optimiser, more variational freedom for the wider lattices &#8212; only the graph differs. '
         'At 12 qubits the W=2 balanced cut slices a rung; the W=3 one is a clean column boundary.</text>')
s.append('</svg>')

out = os.path.join(DOCS, "vqe_width.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out))
for w in widths:
    p = max(p for p in depths if (w, p) in D)
    r = D[(w, p)]
    print(f"  W={w} Lx={r['Lx']} ({r['nbond']} bonds, {r['ngroup']} groups) p={p}: "
          f"err={err(w,p):.3%}  frac={float(r['frac']):.4f}  S_mid={smid(w,p):.3f}  S_max={float(r['S_max']):.3f}")
