"""lambda=4 (half-filled) vs lambda=8 (filled) stripe duel on the doped Hubbard cylinder.

Reads benchmarks/duel_lam{4,8}_chi*.log (QX_STRIPES runs seeded with each stripe
wavelength) and renders docs/hubbard_stripe_duel.svg:

  top    -- the two converged hole-density profiles 1-n(x) at the largest chi, with
            the measured stripe wavelength from a discrete Fourier check: this is the
            PATTERN-RETENTION test. If both seeds relax to the same profile the energy
            comparison is meaningless and the figure says so.
  bottom -- E0/site vs discarded weight for both patterns, with linear extrapolations
            to zero truncation error and the resolution bar |dE| vs the extrapolation
            spread -- the honest verdict on whether the duel is decided.

Also writes docs/hubbard_stripe_duel.csv.

Run: python benchmarks/plot_stripe_duel.py -> docs/hubbard_stripe_duel.svg
"""
import glob, math, os, re, sys

HERE = os.path.dirname(__file__)
HEAD_RE = re.compile(
    r"STRIPES Lx=(\d+) W=(\d+) chi=(\d+) U=(-?[\d.]+) holes=(\d+) hpin=(-?[\d.]+) "
    r"nsw=(\d+) tp=(-?[\d.]+) : E0=(-?[\d.]+) \(([\d.]+)s\).*?lam=(\d+) "
    r"trunc=([\d.eE+-]+)")

runs = []                                   # one dict per (lam, chi) log
for path in sorted(glob.glob(os.path.join(HERE, "duel_lam*_chi*.log"))):
    head, prof = None, {}
    for line in open(path):
        m = HEAD_RE.match(line)
        if m:
            head = dict(Lx=int(m[1]), W=int(m[2]), chi=int(m[3]), U=float(m[4]),
                        holes=int(m[5]), nsw=int(m[7]), E0=float(m[9]),
                        wall=float(m[10]), lam=int(m[11]), trunc=float(m[12]))
        m = re.match(r"PROF (\d+) (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)", line)
        if m:
            prof[int(m[1])] = (int(m[2]), int(m[3]), float(m[4]), float(m[5]))
    if head and prof:
        head["prof"] = prof
        head["E0site"] = head["E0"] / (head["Lx"] * head["W"])
        runs.append(head)
if not runs:
    sys.exit("no complete duel logs (benchmarks/duel_lam*_chi*.log)")

Lx, Wd = runs[0]["Lx"], runs[0]["W"]

def hole_profile(r):
    h = [0.0] * Lx
    for (x, y, nv, _) in r["prof"].values():
        h[x] += (1.0 - nv) / Wd
    return h

def fourier_amp(h, lam):
    """Amplitude of the wavelength-lam component of the hole density.

    The window MUST hold an integer number of periods of every wavelength being
    compared, or the non-commensurate one leaks badly: on a 12-column bulk window
    (1.5 periods of lam=8) a pure lam=4 profile reports A8 = 0.054 > A4 = 0.026 and
    looks like a lam=8 state. The full Lx=16 span is 4 periods of 4 AND 2 of 8, so
    both are clean there (A4 = 0.032, A8 = 0.011 for the same data).
    """
    if Lx % lam != 0:
        return float("nan")
    k = 2 * math.pi / lam
    c = sum(v * math.cos(k * i) for i, v in enumerate(h)) / Lx
    s = sum(v * math.sin(k * i) for i, v in enumerate(h)) / Lx
    return 2 * math.hypot(c, s)

for r in runs:
    h = hole_profile(r)
    r["hole"] = h
    r["a4"] = fourier_amp(h, 4)
    r["a8"] = fourier_amp(h, 8)

with open(os.path.join(HERE, "..", "docs", "hubbard_stripe_duel.csv"), "w", newline="") as f:
    f.write("lam,chi,E0,E0site,trunc,wall,amp_lam4,amp_lam8\n")
    for r in sorted(runs, key=lambda r: (r["lam"], r["chi"])):
        f.write(f'{r["lam"]},{r["chi"]},{r["E0"]:.10f},{r["E0site"]:.10f},'
                f'{r["trunc"]:.6e},{r["wall"]:.1f},{r["a4"]:.6f},{r["a8"]:.6f}\n')

by_lam = {}
for r in runs:
    by_lam.setdefault(r["lam"], []).append(r)
for v in by_lam.values():
    v.sort(key=lambda r: r["trunc"])

def extrapolate(pts):
    """linear E0site vs discarded weight -> intercept; None if < 2 points."""
    if len(pts) < 2:
        return None, None
    xs = [p["trunc"] for p in pts]; ys = [p["E0site"] for p in pts]
    nf, sx, sy = len(xs), sum(xs), sum(ys)
    sxx = sum(x * x for x in xs); sxy = sum(x * y for x, y in zip(xs, ys))
    den = nf * sxx - sx * sx
    if abs(den) < 1e-300:
        return None, None
    b = (nf * sxy - sx * sy) / den
    return (sy - b * sx) / nf, b                       # intercept, slope

ext = {lam: extrapolate(pts) for lam, pts in by_lam.items()}
best_chi = {lam: max(pts, key=lambda r: r["chi"]) for lam, pts in by_lam.items()}

COL = {4: "#d62728", 8: "#1f77b4"}
NAME = {4: "&#955;=4  (half-filled stripes, 2 holes each)",
        8: "&#955;=8  (filled stripes, 4 holes each)"}
W, H = 760, 640
PX0, PX1 = 88, 726
TY0, TY1 = 104, 296
BY0, BY1 = 392, 566

s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="Segoe UI, Arial, sans-serif">',
     f'<rect width="{W}" height="{H}" fill="white"/>']
hd = runs[0]
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="28" text-anchor="middle" font-size="15.5" font-weight="bold" fill="#111">'
         f'Stripe-wavelength duel &#8212; {Lx}&#215;{Wd} Hubbard cylinder, U={hd["U"]:g}t, '
         f'&#948;=1/{(Lx*Wd)//hd["holes"]}: &#955;=4 vs &#955;=8</text>')
chis = sorted({r["chi"] for r in runs})
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="48" text-anchor="middle" font-size="11.5" fill="#555">'
         f'identical protocol per point (&#967; = {", ".join(str(c) for c in chis)}, '
         f'{hd["nsw"]} mixer sweeps, same edge pinning); only the hole SEED pattern differs</text>')

# ---- top panel: converged hole profiles at the largest chi (pattern retention) ----
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY0-14}" text-anchor="middle" font-size="12.5" fill="#222" font-weight="bold">'
         'converged hole density 1 &#8722; n(x) at the largest &#967;  '
         '<tspan font-weight="normal" fill="#666">(did each seed keep its pattern?)</tspan></text>')
hmax = max(max(r["hole"]) for r in best_chi.values()) * 1.18
cw = (PX1 - PX0) / Lx
def xctr(x): return PX0 + (x + 0.5) * cw
def typx(v): return TY1 - v / hmax * (TY1 - TY0)
g = 0.0
while g <= hmax:
    yv = typx(g)
    s.append(f'<line x1="{PX0}" y1="{yv:.1f}" x2="{PX1}" y2="{yv:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-7}" y="{yv+4:.1f}" text-anchor="end" font-size="10.5" fill="#444">{g:.2f}</text>')
    g += 0.05
mean_h = hd["holes"] / (Lx * Wd)
s.append(f'<line x1="{PX0}" y1="{typx(mean_h):.1f}" x2="{PX1}" y2="{typx(mean_h):.1f}" stroke="#bbb" stroke-dasharray="5,4"/>')
for lam in sorted(best_chi):
    r = best_chi[lam]; c = COL[lam]
    poly = " ".join(f"{xctr(x):.1f},{typx(r['hole'][x]):.1f}" for x in range(Lx))
    s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="2.2" opacity="0.9"/>')
    for x in range(Lx):
        s.append(f'<circle cx="{xctr(x):.1f}" cy="{typx(r["hole"][x]):.1f}" r="3.2" fill="{c}"/>')
for x in range(Lx):
    s.append(f'<text x="{xctr(x):.1f}" y="{TY1+16:.1f}" text-anchor="middle" font-size="9.5" fill="#444">{x}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{TY1+32:.1f}" text-anchor="middle" font-size="11.5" fill="#222">column x</text>')
s.append(f'<line x1="{PX0}" y1="{TY0}" x2="{PX0}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{TY1}" x2="{PX1}" y2="{TY1}" stroke="#444" stroke-width="1.2"/>')
ly = TY0 + 8
for lam in sorted(best_chi):
    r = best_chi[lam]
    s.append(f'<line x1="{PX1-268}" y1="{ly}" x2="{PX1-242}" y2="{ly}" stroke="{COL[lam]}" stroke-width="2.4"/>')
    s.append(f'<text x="{PX1-234}" y="{ly+4}" font-size="11" fill="#333">{NAME[lam]} &#8212; '
             f'A&#8324;={r["a4"]:.4f}, A&#8328;={r["a8"]:.4f}</text>')
    ly += 19

# ---- bottom panel: E0/site vs discarded weight ----
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY0-14}" text-anchor="middle" font-size="12.5" fill="#222" font-weight="bold">'
         'ground-state energy per site vs discarded weight  '
         '<tspan font-weight="normal" fill="#666">(dashes = linear extrapolation to &#967;&#8594;&#8734;)</tspan></text>')
allE = [r["E0site"] for r in runs] + [v[0] for v in ext.values() if v[0] is not None]
emin, emax = min(allE), max(allE)
pad = (emax - emin) * 0.25 or 1e-6
emin -= pad; emax += pad
tmax = max(r["trunc"] for r in runs) * 1.15
def bxpx(v): return PX0 + v / tmax * (PX1 - PX0)
def bypx(v): return BY1 - (v - emin) / (emax - emin) * (BY1 - BY0)
for k in range(5):
    yv = BY0 + k * (BY1 - BY0) / 4
    ev = emin + (1 - k / 4) * (emax - emin)
    s.append(f'<line x1="{PX0}" y1="{yv:.1f}" x2="{PX1}" y2="{yv:.1f}" stroke="#eee"/>')
    s.append(f'<text x="{PX0-7}" y="{yv+4:.1f}" text-anchor="end" font-size="10" fill="#444">{ev:.5f}</text>')
for k in range(5):
    xv = PX0 + k * (PX1 - PX0) / 4
    tv = k * tmax / 4
    s.append(f'<line x1="{xv:.1f}" y1="{BY1}" x2="{xv:.1f}" y2="{BY1+5}" stroke="#444"/>')
    s.append(f'<text x="{xv:.1f}" y="{BY1+18}" text-anchor="middle" font-size="10" fill="#444">{tv:.1e}</text>')
for lam, pts in sorted(by_lam.items()):
    c = COL[lam]
    a, b = ext[lam]
    if a is not None:
        s.append(f'<line x1="{bxpx(0):.1f}" y1="{bypx(a):.1f}" x2="{bxpx(tmax):.1f}" '
                 f'y2="{bypx(a + b*tmax):.1f}" stroke="{c}" stroke-width="1.2" '
                 f'stroke-dasharray="6,4" opacity="0.7"/>')
        s.append(f'<circle cx="{bxpx(0):.1f}" cy="{bypx(a):.1f}" r="5" fill="none" stroke="{c}" stroke-width="2"/>')
    poly = " ".join(f"{bxpx(p['trunc']):.1f},{bypx(p['E0site']):.1f}" for p in pts)
    if len(pts) > 1:
        s.append(f'<polyline points="{poly}" fill="none" stroke="{c}" stroke-width="1.6" opacity="0.5"/>')
    for p in pts:
        s.append(f'<circle cx="{bxpx(p["trunc"]):.1f}" cy="{bypx(p["E0site"]):.1f}" r="4" fill="{c}"/>')
        s.append(f'<text x="{bxpx(p["trunc"]):.1f}" y="{bypx(p["E0site"])-9:.1f}" text-anchor="middle" '
                 f'font-size="9" fill="{c}">&#967;={p["chi"]}</text>')
s.append(f'<line x1="{PX0}" y1="{BY0}" x2="{PX0}" y2="{BY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<line x1="{PX0}" y1="{BY1}" x2="{PX1}" y2="{BY1}" stroke="#444" stroke-width="1.2"/>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{BY1+36:.1f}" text-anchor="middle" font-size="11.5" fill="#222">'
         'discarded weight (truncation error)</text>')
s.append(f'<text x="{22}" y="{(BY0+BY1)/2:.0f}" text-anchor="middle" font-size="11.5" fill="#222" '
         f'transform="rotate(-90 22 {(BY0+BY1)/2:.0f})">E&#8320;/site (t)</text>')

# ---- verdict line ----
verdict, vcol = "", "#555"
if len(by_lam) == 2:
    l1, l2 = sorted(by_lam)
    b1, b2 = best_chi[l1], best_chi[l2]
    dE = b2["E0site"] - b1["E0site"]
    e1, e2 = ext[l1][0], ext[l2][0]
    same = (max(b1["a4"], b1["a8"]) > 0 and
            (b1["a4"] > b1["a8"]) == (b2["a4"] > b2["a8"]))
    if same:
        verdict = ("Both seeds relaxed to the SAME dominant wavelength &#8212; "
                   "the two runs are not independent states, so no energy verdict.")
        vcol = "#b06000"
    elif e1 is not None and e2 is not None:
        dext = e2 - e1
        spread = abs((b2["E0site"] - e2) - (b1["E0site"] - e1))
        decided = abs(dext) > spread
        verdict = (f'&#916;E/site (&#955;=8 &#8722; &#955;=4) = {dE:+.2e} at the largest &#967;, '
                   f'{dext:+.2e} extrapolated; extrapolation spread {spread:.1e} '
                   f'&#8594; {"RESOLVED" if decided else "NOT resolved"}')
        vcol = "#0a7d28" if decided else "#b06000"
    else:
        verdict = (f'&#916;E/site (&#955;=8 &#8722; &#955;=4) = {dE:+.2e} at &#967;={b1["chi"]} '
                   f'(one &#967; point per pattern &#8212; no extrapolation yet)')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-30}" text-anchor="middle" font-size="11.5" fill="{vcol}" font-weight="bold">{verdict}</text>')
s.append(f'<text x="{(PX0+PX1)/2:.0f}" y="{H-12}" text-anchor="middle" font-size="9.5" fill="#777" font-style="italic">'
         'A&#8324;/A&#8328; = Fourier amplitude of the bulk hole density at wavelength 4 / 8. '
         'Both patterns are variational states of the same Hamiltonian; the lower energy wins only if the gap exceeds the extrapolation spread.</text>')
s.append('</svg>')

out = os.path.join(HERE, "..", "docs", "hubbard_stripe_duel.svg")
open(out, "w", encoding="utf-8").write("\n".join(s))
print("wrote", os.path.normpath(out))
for r in sorted(runs, key=lambda r: (r["lam"], r["chi"])):
    print(f'  lam={r["lam"]} chi={r["chi"]:>5}  E0/site={r["E0site"]:.8f}  '
          f'trunc={r["trunc"]:.3e}  A4={r["a4"]:.4f} A8={r["a8"]:.4f}  ({r["wall"]:.0f}s)')
for lam, (a, b) in sorted(ext.items()):
    if a is not None:
        print(f'  lam={lam} extrapolated E0/site = {a:.8f}')
