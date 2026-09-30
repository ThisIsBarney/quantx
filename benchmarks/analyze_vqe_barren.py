#!/usr/bin/env python3
"""Analysis for part F: is the width penalty a barren plateau, or a constant factor?

A barren plateau is Var[dC/dtheta] ~ e^{-a n}. At a SINGLE size a constant factor and
e^{-a n} evaluated there are the same number, so the depth scan alone cannot answer the
question it raises -- only the size axis can, and only where the box could afford one.

This script exists so the distinction is drawn by arithmetic rather than by assertion:
it fits log(var) against n per width and reports the decay rate WITH its uncertainty,
refuses to fit fewer than three points, and separates the even-n series (2Sz = 0) from
the odd-n one (2Sz = 1) because half filling at odd n sits in a different charge sector
and that is a difference between points, not a size effect.

Error bars come from the scan's own fourth moment (se_frac), not from s^2*sqrt(2/N).
That familiar form assumes a Gaussian gradient distribution, and where the distribution
is heavy-tailed it understates the error -- the direction that turns noise into a
significant difference.
"""
import csv, math, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DOCS = os.path.join(os.path.dirname(HERE), "docs")
NAME = {1: "chain", 2: "ladder", 3: "cylinder"}


def load(name):
    path = os.path.join(DOCS, name)
    if not os.path.exists(path):
        sys.exit(f"missing docs/{name} -- run QX_VQE_ONLY=F qx_vqe_showcase")
    return list(csv.DictReader(open(path)))


def local_rates(pts):
    """per-site decay rate between CONSECUTIVE points: a_i = ln(v_i/v_{i+1})/(n_{i+1}-n_i).

    An exponential has these constant. They are the model check that has to run BEFORE a
    rate is quoted: a straight line through a bending curve still returns a slope and a
    respectable-looking standard error, and reporting that as a barren-plateau exponent
    would be a fabricated measurement.
    """
    return [math.log(pts[i][1] / pts[i + 1][1]) / (pts[i + 1][0] - pts[i][0])
            for i in range(len(pts) - 1)]


def fit_loglin(xs, ys):
    """least squares log(y) = c - a x; returns (a, stderr_a, n) or None if under-determined"""
    m = len(xs)
    if m < 3:
        return None
    lx = list(xs)
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / m, sum(ly) / m
    sxx = sum((x - mx) ** 2 for x in lx)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(lx, ly)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(lx, ly)]
    dof = m - 2
    se = math.sqrt(sum(r * r for r in resid) / dof / sxx) if dof > 0 else float("nan")
    return -slope, se, m


# ---- depth scan -------------------------------------------------------------
D = load("vqe_barren.csv")
Dk = {(int(r["W"]), int(r["p"])): r for r in D}
ws = sorted({int(r["W"]) for r in D})
ps = sorted({int(r["p"]) for r in D})
N = int(D[0]["nsamp"])
gauss = math.sqrt(2.0 / N)                        # what a Gaussian WOULD give
meas = [float(r["se_frac"]) / float(r["var_frac"]) for r in D]

print(f"DEPTH SCAN  ({N} samples/point)")
print(f"  relative SE per variance: measured {min(meas):.1%}-{max(meas):.1%} "
      f"(Gaussian assumption would say {gauss:.1%})")
if max(meas) > 1.5 * gauss:
    print(f"  -> the gradient distribution is heavy-tailed here; the Gaussian error bar "
          f"would have been up to {max(meas)/gauss:.1f}x too small")
gsz = {int(r["W"]): int(r["nb_g0"]) for r in D}
print(f"  differentiated generator: group 0 = {gsz} bonds per width")
if len(set(gsz.values())) == 1:
    print("  -> identical generator norm at every width (a colouring group is a MATCHING,")
    print("     so H_g is a sum of that many disjoint commuting 2-site terms): the")
    print("     cross-width comparison is not a normalisation artefact")
else:
    print("  -> !! generator sizes DIFFER; the cross-width comparison is confounded")
print()
print("    p |" + "".join(f"  W={w:<8d}" for w in ws) + "|" + "".join(
      f"  W{w}/W1" for w in ws[1:]))
for p in ps:
    v = [float(Dk[(w, p)]["var_frac"]) for w in ws]
    e = [float(Dk[(w, p)]["se_frac"]) for w in ws]
    line = f"  {p:3d} |" + "".join(f"  {x:8.4f}" for x in v) + "|"
    for i in range(1, len(ws)):
        r = v[i] / v[0]
        # SE of a ratio of independent estimates, first order
        rse = r * math.sqrt((e[i] / v[i]) ** 2 + (e[0] / v[0]) ** 2)
        sig = "*" if abs(r - 1.0) > 2 * rse else " "
        line += f"  {r:5.2f}+/-{rse:.2f}{sig}"
    print(line)
print("  (* = differs from the chain by more than 2 measured sigma)")
print()
for w in ws:
    v = [float(Dk[(w, p)]["var_frac"]) for p in ps]
    print(f"  W={w} {NAME[w]:9s} p={ps[0]}->{ps[-1]}: {v[0]:.3f} -> {v[-1]:.3f}  "
          f"(x{v[-1]/v[0]:.3f})")
snr_min = min(float(r["snr"]) for r in D)
print(f"\n  worst signal-to-noise over all rows: {snr_min:.3g}x the finite-difference floor")

# ---- size axis --------------------------------------------------------------
print("\n\nSIZE AXIS  (the only thing that can distinguish a plateau from a factor)")
S = load("vqe_barren_size.csv")
Ns = int(S[0]["nsamp"])
print(f"  {Ns} samples/point, {math.sqrt(2.0/Ns):.1%} SE per variance\n")
print("    W    n  qubits   chi    Var[dfrac/dtheta]   sector")
for r in sorted(S, key=lambda r: (int(r["W"]), int(r["n"]))):
    n = int(r["n"])
    print(f"  {int(r['W']):3d} {n:4d} {int(r['qubits']):7d} {int(r['chi']):5d}   "
          f"{float(r['var_frac']):16.5f}   2Sz={n % 2}")

print("\n  decay rate a in Var ~ exp(-a n), even n only (all in the 2Sz=0 sector):")
any_fit = False
for w in ws:
    pts = sorted((int(r["n"]), float(r["var_frac"])) for r in S
                 if int(r["W"]) == w and int(r["n"]) % 2 == 0)
    f = fit_loglin([n for n, _ in pts], [v for _, v in pts])
    if f is None:
        have = ", ".join(str(n) for n, _ in pts) or "none"
        print(f"    W={w} {NAME[w]:9s}: NOT FITTABLE -- only {len(pts)} even-n point(s) "
              f"(n = {have}); a rate needs at least 3")
        continue
    a, se, m = f
    lr = local_rates(pts)
    spread = max(lr) / min(lr) if min(lr) > 0 else float("inf")
    rates = " ".join(f"{r:+.2f}" for r in lr)
    print(f"    W={w} {NAME[w]:9s}: local rates {rates}  (n = "
          f"{', '.join(str(n) for n, _ in pts)})")
    if spread > 3.0 or min(lr) <= 0:
        print(f"      -> NOT a single exponential: the local rate varies by "
              f"x{spread:.1f} along the series, so no barren-plateau exponent is")
        print(f"         quoted. A straight-line fit would return a = {a:+.3f} +/- "
              f"{se:.3f}, which would be a fabricated measurement.")
    else:
        any_fit = True
        print(f"      -> consistent with a single exponential: a = {a:+.4f} +/- "
              f"{se:.4f} per site ({m} points)")
if not any_fit:
    print()
    print("  NO width supports a barren-plateau exponent over the accessible sizes.")
    print("  The variance does fall with n, but the rate itself collapses along the")
    print("  series, which is a finite-size crossover and not the exponential decay the")
    print("  term denotes. n <= 10 is where this box stops (chi = 4^(n/2) lossless).")

odd = sorted((int(r["W"]), int(r["n"]), float(r["var_frac"])) for r in S
             if int(r["n"]) % 2 == 1)
if odd:
    print("\n  odd-n points sit in the 2Sz=1 sector and are NOT part of the fits above:")
    for w, n, v in odd:
        print(f"    W={w} n={n:2d}: Var = {v:.5f}")
