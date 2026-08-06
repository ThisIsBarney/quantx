"""Geometry at matched qubit count: does quasi-2D cost a fermionic VQE more?

Reads docs/vqe_hubbard.csv / vqe_entanglement.csv (chain, parts A/B) and
docs/vqe_ladder.csv (2-leg ladder at its own optimum, part D) and renders
docs/vqe_geometry.svg -- two panels against circuit depth, one pair of curves per
matched qubit count:

  left  -- relative energy error |E_VQE - E_0| / |E_0|: how well the ansatz does;
  right -- half-chain entropy S: how hard the prepared state is to simulate.

Both simulations are LOSSLESS at these sizes (chi = 4^(n/2) for the chain,
16^(Lx/2) for the ladder), so neither curve is a truncation artefact.

Run: python benchmarks/plot_vqe_geometry.py -> docs/vqe_geometry.svg
"""
import csv, math, os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")

A = {(int(r["n"]), int(r["p"])): r for r in csv.DictReader(open(os.path.join(DOCS, "vqe_hubbard.csv")))}
B = {(int(r["n"]), int(r["p"])): r for r in csv.DictReader(open(os.path.join(DOCS, "vqe_entanglement.csv")))}
L = {(int(r["Lx"]), int(r["p"])): r for r in csv.DictReader(open(os.path.join(DOCS, "vqe_ladder.csv")))}

# matched pairs: (qubits, chain sites, ladder rungs)
PAIRS = [(12, 6, 3), (16, 8, 4)]
depths = sorted({p for (_, p) in L})

def chain_err(n, p):
    r = A.get((n, p))
    if not r: return None
    return abs(float(r["E_vqe"]) - float(r["E_exact"])) / abs(float(r["E_exact"]))

def ladder_err(lx, p):
    r = L.get((lx, p))
    if not r: return None
    return abs(float(r["E_vqe"]) - float(r["E_exact"])) / abs(float(r["E_exact"]))

def chain_S(n, p):
    # part B covers n >= 8; part A now records S too and covers the smaller sizes
    r = B.get((n, p)) or A.get((n, p))
    return float(r["S_half"]) if r and "S_half" in r else None

def ladder_S(lx, p):
    r = L.get((lx, p))
    return float(r["S_half"]) if r else None

W, H = 820, 430
PY0, PY1 = 108, 330
LX0, LX1 = 72, 392          # left panel
RX0, RX1 = 470, 792         # right panel

COL = {12: "#1f77b4", 16: "#d62728"}

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{W/2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111">'
         'Quasi-2D costs a fermionic VQE twice &#8212; worse energy AND more entanglement</text>')
s.append(f'<text x="{W/2:.0f}" y="48" text-anchor="middle" font-size="11.5" fill="#555">'
         '1D chain vs 2-leg ladder (rung-fused, d=16) at MATCHED qubit count, each with its own optimised angles; '
         'both simulations lossless</text>')
s.append(f'<text x="{W/2:.0f}" y="66" text-anchor="middle" font-size="10.5" fill="#777">'
         'solid = chain, dashed = ladder; colour = qubit count</text>')

# ---- left panel: relative energy error (log) ----
errs = [e for q, n, lx in PAIRS for p in depths
        for e in (chain_err(n, p), ladder_err(lx, p)) if e]
lo = 10 ** math.floor(math.log10(min(errs)))
hi = 10 ** math.ceil(math.log10(max(errs)))
def lxpx(p): return LX0 + (p - depths[0]) / (depths[-1] - depths[0]) * (LX1 - LX0)
def lypx(v): return PY1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (PY1 - PY0)
s.append(f'<text x="{(LX0+LX1)/2:.0f}" y="{PY0-14}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#222">'
         'relative energy error</text>')
v = lo
while v <= hi * 1.0001:
    y = lypx(v)
    s.append(f'<line x1="{LX0}" y1="{y:.1f}" x2="{LX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{LX0-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">10<tspan baseline-shift="super" font-size="8">{round(math.log10(v))}</tspan></text>')
    v *= 10
for q, n, lx in PAIRS:
    c = COL[q]
    cp = [(p, chain_err(n, p)) for p in depths if chain_err(n, p)]
    lp = [(p, ladder_err(lx, p)) for p in depths if ladder_err(lx, p)]
    if cp:
        s.append('<polyline points="' + " ".join(f"{lxpx(p):.1f},{lypx(e):.1f}" for p, e in cp) +
                 f'" fill="none" stroke="{c}" stroke-width="2.1"/>')
        for p, e in cp: s.append(f'<circle cx="{lxpx(p):.1f}" cy="{lypx(e):.1f}" r="3.2" fill="{c}"/>')
    if lp:
        s.append('<polyline points="' + " ".join(f"{lxpx(p):.1f},{lypx(e):.1f}" for p, e in lp) +
                 f'" fill="none" stroke="{c}" stroke-width="2.1" stroke-dasharray="6,4"/>')
        for p, e in lp: s.append(f'<circle cx="{lxpx(p):.1f}" cy="{lypx(e):.1f}" r="3.2" fill="white" stroke="{c}" stroke-width="1.8"/>')

# ---- right panel: half-chain entropy (linear) ----
Ss = [x for q, n, lx in PAIRS for p in depths
      for x in (chain_S(n, p), ladder_S(lx, p)) if x]
smax = max(Ss) * 1.15
def rxpx(p): return RX0 + (p - depths[0]) / (depths[-1] - depths[0]) * (RX1 - RX0)
def rypx(v): return PY1 - v / smax * (PY1 - PY0)
s.append(f'<text x="{(RX0+RX1)/2:.0f}" y="{PY0-14}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#222">'
         'half-chain entropy S</text>')
g = 0.0
while g <= smax:
    y = rypx(g)
    s.append(f'<line x1="{RX0}" y1="{y:.1f}" x2="{RX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{RX0-7}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">{g:.1f}</text>')
    g += 0.5
for q, n, lx in PAIRS:
    c = COL[q]
    cp = [(p, chain_S(n, p)) for p in depths if chain_S(n, p)]
    lp = [(p, ladder_S(lx, p)) for p in depths if ladder_S(lx, p)]
    if cp:
        s.append('<polyline points="' + " ".join(f"{rxpx(p):.1f},{rypx(e):.1f}" for p, e in cp) +
                 f'" fill="none" stroke="{c}" stroke-width="2.1"/>')
        for p, e in cp: s.append(f'<circle cx="{rxpx(p):.1f}" cy="{rypx(e):.1f}" r="3.2" fill="{c}"/>')
    if lp:
        s.append('<polyline points="' + " ".join(f"{rxpx(p):.1f},{rypx(e):.1f}" for p, e in lp) +
                 f'" fill="none" stroke="{c}" stroke-width="2.1" stroke-dasharray="6,4"/>')
        for p, e in lp: s.append(f'<circle cx="{rxpx(p):.1f}" cy="{rypx(e):.1f}" r="3.2" fill="white" stroke="{c}" stroke-width="1.8"/>')

for x0, x1, xf in ((LX0, LX1, lxpx), (RX0, RX1, rxpx)):
    for p in depths:
        s.append(f'<line x1="{xf(p):.1f}" y1="{PY1}" x2="{xf(p):.1f}" y2="{PY1+5}" stroke="#444"/>')
        s.append(f'<text x="{xf(p):.1f}" y="{PY1+19}" text-anchor="middle" font-size="10" fill="#444">{p}</text>')
    s.append(f'<line x1="{x0}" y1="{PY0}" x2="{x0}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
    s.append(f'<line x1="{x0}" y1="{PY1}" x2="{x1}" y2="{PY1}" stroke="#444" stroke-width="1.2"/>')
    s.append(f'<text x="{(x0+x1)/2:.0f}" y="{PY1+38:.1f}" text-anchor="middle" font-size="11.5" fill="#222">circuit depth p</text>')

ly = PY0 + 6
for q, n, lx in PAIRS:
    s.append(f'<line x1="{RX1-118}" y1="{ly}" x2="{RX1-96}" y2="{ly}" stroke="{COL[q]}" stroke-width="2.4"/>')
    s.append(f'<text x="{RX1-90}" y="{ly+4}" font-size="11" fill="#333">{q} qubits</text>')
    ly += 18

pmax = depths[-1]
cs, ls_ = chain_S(8, pmax), ladder_S(4, pmax)
ce, le = chain_err(8, pmax), ladder_err(4, pmax)
if cs and ls_ and ce and le:
    s.append(f'<text x="{W/2:.0f}" y="{H-26}" text-anchor="middle" font-size="11" fill="#0a7d28" font-weight="bold">'
             f'At 16 qubits and depth {pmax}: the ladder needs {ls_/cs:.1f}&#215; the entropy and still lands '
             f'{le/ce:.1f}&#215; further from the ground state.</text>')
s.append(f'<text x="{W/2:.0f}" y="{H-9}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'The width, not the qubit count, is what makes a fermionic VQE state hard &#8212; and the same width makes the ansatz converge worse.</text>')
s.append('</svg>')

out = os.path.join(DOCS, "vqe_geometry.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out))
for q, n, lx in PAIRS:
    for p in (depths[-1],):
        print(f"  {q:>2}q p={p}: chain S={chain_S(n,p)}, err={chain_err(n,p):.3%} | "
              f"ladder S={ladder_S(lx,p):.3f}, err={ladder_err(lx,p):.3%}")
