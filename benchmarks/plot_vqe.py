"""Fermionic VQE (HVA) on the block-sparse engine: circuit depth and the
classical-simulability question.

Reads docs/vqe_hubbard.csv (part A: depth convergence, exact simulation),
docs/vqe_entanglement.csv (part B: largest-bond entropy vs size) and, if present,
docs/vqe_transfer_check.csv (part C: is the angle transfer sound?), and renders
docs/vqe_hubbard.svg:

  top    -- energy error per site vs circuit depth p, at the sizes where the MPS
            simulation is EXACT (chi = 4^(n/2)): how deep the ansatz must be, and
            how that depth grows with the system;
  bottom -- largest bond entanglement entropy S of the prepared state vs system size
            (log x), for several depths, with a logarithmic fit. S is what decides
            classical simulability (an MPS needs chi ~ e^S): S growing like log n
            rather than like n means the VQE state is NOT volume-law entangled, so
            no qubit count makes this ansatz classically hard at these depths.

Run: python benchmarks/plot_vqe.py -> docs/vqe_hubbard.svg
"""
import csv, math, os

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")

A = list(csv.DictReader(open(os.path.join(DOCS, "vqe_hubbard.csv"))))
B = list(csv.DictReader(open(os.path.join(DOCS, "vqe_entanglement.csv"))))
try:
    C = list(csv.DictReader(open(os.path.join(DOCS, "vqe_transfer_check.csv"))))
except OSError:
    C = []

sizes_a = sorted({int(r["n"]) for r in A})
sizes_b = sorted({int(r["n"]) for r in B})
depths  = sorted({int(r["p"]) for r in A})

def a_err(n, p):
    m = [r for r in A if int(r["n"]) == n and int(r["p"]) == p]
    return float(m[0]["err_site"]) if m else None

def b_S(n, p):
    m = [r for r in B if int(r["n"]) == n and int(r["p"]) == p]
    return float(m[0]["S_max"]) if m else None

def b_row(n, p):
    m = [r for r in B if int(r["n"]) == n and int(r["p"]) == p]
    return m[0] if m else None

# logarithmic fit S = a + b ln(n) on the deepest available circuit
p_top = max(int(r["p"]) for r in B)
fit_pts = [(n, b_S(n, p_top)) for n in sizes_b if b_S(n, p_top) is not None]
xs = [math.log(n) for n, _ in fit_pts]; ys = [s for _, s in fit_pts]
nf, sx, sy = len(xs), sum(xs), sum(ys)
sxx = sum(x*x for x in xs); sxy = sum(x*y for x, y in zip(xs, ys))
b_fit = (nf*sxy - sx*sy) / (nf*sxx - sx*sx)
a_fit = (sy - b_fit*sx) / nf
resid = max(abs(s - (a_fit + b_fit*math.log(n))) for n, s in fit_pts)

W, H = 780, 700
PX0, PX1 = 92, 744
TY0, TY1 = 108, 330            # top panel (depth convergence)
BY0, BY1 = 432, 636            # bottom panel (entropy vs size)

COL_A = {4: "#1f77b4", 6: "#2ca02c", 8: "#d62728"}
COL_B = {1: "#bbbbbb", 2: "#1f77b4", 4: "#2ca02c", 8: "#d62728"}

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="30" text-anchor="middle" font-size="16" font-weight="bold" fill="#111">'
         'Fermionic VQE on the block-sparse engine &#8212; how deep, and is it classically hard?</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="52" text-anchor="middle" font-size="11.5" fill="#555">'
         'Hamiltonian Variational Ansatz, 1D Hubbard chain, U=4t, half filling; every layer conserves N and 2Sz, '
         'so the U(1)&#215;U(1) block engine simulates the circuit exactly</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="70" text-anchor="middle" font-size="10.5" fill="#777">'
         'circuit vs a dense state-vector simulator: 5.3e-15 (n=4), 4.0e-11 (n=6) &#8212; qx_hubbard_test, section HVA CIRCUIT</text>')

# ---------- top panel: energy error vs depth ----------
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY0-16}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#222">'
         'energy error per site vs circuit depth  '
         '<tspan font-weight="normal" fill="#666">(exact simulation: &#967; = 4<tspan baseline-shift="super" font-size="8">n/2</tspan>)</tspan></text>')
evals = [a_err(n, p) for n in sizes_a for p in depths if a_err(n, p)]
lo = 10 ** math.floor(math.log10(min(evals)))
hi = 10 ** math.ceil(math.log10(max(evals)))
def txpx(p): return PX0 + (p - depths[0]) / (depths[-1] - depths[0]) * (PX1 - PX0)
def typx(v): return TY1 - (math.log10(v) - math.log10(lo)) / (math.log10(hi) - math.log10(lo)) * (TY1 - TY0)
v = lo
while v <= hi * 1.0001:
    y = typx(v)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">10<tspan baseline-shift="super" font-size="8">{round(math.log10(v))}</tspan></text>')
    v *= 10
for p in depths:
    x = txpx(p)
    s.append(f'<line x1="{x:.1f}" y1="{TY1}" x2="{x:.1f}" y2="{TY1+5}" stroke="#444"/>')
    s.append(f'<text x="{x:.1f}" y="{TY1+19}" text-anchor="middle" font-size="10.5" fill="#444">{p}</text>')
for n in sizes_a:
    pts = [(p, a_err(n, p)) for p in depths if a_err(n, p)]
    c = COL_A.get(n, "#888")
    poly = " ".join(f"{txpx(p):.1f},{typx(e):.1f}" for p, e in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="2.1"/>')
    for p, e in pts:
        s.append(f'<circle cx="{txpx(p):.1f}" cy="{typx(e):.1f}" r="3.4" fill="{c}"/>')
    lp, le = pts[-1]
    s.append(f'<text x="{txpx(lp)-6:.1f}" y="{typx(le)+14:.1f}" text-anchor="end" font-size="11" fill="{c}" font-weight="bold">n={n} ({2*n}q)</text>')
s.append(f'<line x1="{PX0}" y1="{TY0}" x2="{PX0}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{TY1}" x2="{PX1}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY1+38:.1f}" text-anchor="middle" font-size="12" fill="#222">circuit depth p (layers)</text>')
s.append(f'<text x="26" y="{(TY0+TY1)/2:.0f}" text-anchor="middle" font-size="12" fill="#222" '
         f'transform="rotate(-90 26 {(TY0+TY1)/2:.0f})">|E&#8321;&#8323;&#8331; &#8722; E&#8320;| / site (t)</text>')

# ---------- bottom panel: entropy vs size ----------
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-52}" text-anchor="middle" font-size="12.5" font-weight="bold" fill="#222">'
         'largest bond entanglement entropy of the prepared state vs system size</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-34}" text-anchor="middle" font-size="10.8" fill="#666">'
         'an MPS needs &#967; ~ e<tspan baseline-shift="super" font-size="8">S</tspan>: S growing like log n, not like n, '
         'means the ansatz is NOT volume-law entangled at these depths</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-16}" text-anchor="middle" font-size="10.5" fill="#777">'
         'angles transferred from the largest exactly-simulable size (validated in part C)</text>')
Svals = [b_S(n, p) for n in sizes_b for p in depths if b_S(n, p) is not None]
smin, smax = 0.0, max(Svals) * 1.15
lnmin, lnmax = math.log(sizes_b[0]) - 0.12, math.log(sizes_b[-1]) + 0.12
def bxpx(n): return PX0 + (math.log(n) - lnmin) / (lnmax - lnmin) * (PX1 - PX0)
def bypx(v): return BY1 - (v - smin) / (smax - smin) * (BY1 - BY0)
g = 0.0
while g <= smax:
    y = bypx(g)
    s.append(f'<line x1="{PX0}" y1="{y:.1f}" x2="{PX1}" y2="{y:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-8}" y="{y+4:.1f}" text-anchor="end" font-size="10" fill="#444">{g:.1f}</text>')
    g += 0.5
for n in sizes_b:
    x = bxpx(n)
    s.append(f'<line x1="{x:.1f}" y1="{BY1}" x2="{x:.1f}" y2="{BY1+5}" stroke="#444"/>')
    s.append(f'<text x="{x:.1f}" y="{BY1+18}" text-anchor="middle" font-size="10.5" fill="#444">{n}</text>')
    s.append(f'<text x="{x:.1f}" y="{BY1+31}" text-anchor="middle" font-size="9" fill="#888">{2*n}q</text>')
# the log fit
fx = [lnmin, lnmax]
s.append(f'<line x1="{bxpx(math.exp(fx[0])):.1f}" y1="{bypx(a_fit + b_fit*fx[0]):.1f}" '
         f'x2="{bxpx(math.exp(fx[1])):.1f}" y2="{bypx(a_fit + b_fit*fx[1]):.1f}" '
         f'stroke="#d62728" stroke-width="1.2" stroke-dasharray="6,4" opacity="0.6"/>')
s.append(f'<text x="{bxpx(sizes_b[1]):.1f}" y="{bypx(a_fit + b_fit*math.log(sizes_b[1]))-12:.1f}" '
         f'font-size="10.5" fill="#d62728">S &#8776; {b_fit:.2f} ln n {a_fit:+.2f}  (max dev {resid:.3f})</text>')
for p in [1, 2, 4, 8]:
    pts = [(n, b_S(n, p)) for n in sizes_b if b_S(n, p) is not None]
    if not pts: continue
    c = COL_B.get(p, "#888")
    poly = " ".join(f"{bxpx(n):.1f},{bypx(S):.1f}" for n, S in pts)
    s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="2.1"/>')
    for n, S in pts:
        s.append(f'<circle cx="{bxpx(n):.1f}" cy="{bypx(S):.1f}" r="3.4" fill="{c}"/>')
    ln, lS = pts[-1]
    s.append(f'<text x="{bxpx(ln)+7:.1f}" y="{bypx(lS)+4:.1f}" font-size="10.5" fill="{c}" font-weight="bold">p={p}</text>')
s.append(f'<line x1="{PX0}" y1="{BY0}" x2="{PX0}" y2="{BY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{BY1}" x2="{PX1}" y2="{BY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY1+50:.1f}" text-anchor="middle" font-size="12" fill="#222">'
         'chain length n (log scale) &#8212; the circuit uses 2n qubits</text>')
s.append(f'<text x="26" y="{(BY0+BY1)/2:.0f}" text-anchor="middle" font-size="12" fill="#222" '
         f'transform="rotate(-90 26 {(BY0+BY1)/2:.0f})">largest bond entropy S</text>')

# footer: the headline number at the largest size
top = b_row(sizes_b[-1], p_top)
if top:
    s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-24}" text-anchor="middle" font-size="11" fill="#0a7d28" font-weight="bold">'
             f'At {2*sizes_b[-1]} qubits and depth {p_top} the circuit captures {100*float(top["frac"]):.0f}% of the correlation '
             f'energy with S = {float(top["S_max"]):.2f} &#8212; simulated classically to a discarded weight of '
             f'{float(top["trunc_err"]):.0e} at &#967; = {top["chi_used"]}.</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-8}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         '&#967; is the total bond dimension over all charge sectors at a 1e-10 cutoff, so it far exceeds e'
         '<tspan baseline-shift="super" font-size="7">S</tspan>; S is the cutoff-independent measure.</text>')
s.append('</svg>')

out = os.path.join(DOCS, "vqe_hubbard.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out))
print(f"  S(p={p_top}) fit: S = {b_fit:.3f} ln(n) {a_fit:+.3f}   (max deviation {resid:.4f})")
for n in sizes_b:
    r = b_row(n, p_top)
    if r: print(f"  n={n:>2} ({2*n:>2}q): S={float(r['S_max']):.3f}  S/n={float(r['S_max'])/n:.4f}  "
                f"frac={float(r['frac']):.3f}  chi={r['chi_used']}  trunc={float(r['trunc_err']):.1e}")
if C:
    print("  transfer check (part C):")
    for r in C:
        print(f"    n={r['n']} p={r['p']}: S {float(r['S_transfer']):.3f} -> {float(r['S_reopt']):.3f} "
              f"(dS={float(r['dS']):+.3f}), E {float(r['E_transfer']):.4f} -> {float(r['E_reopt']):.4f}")
