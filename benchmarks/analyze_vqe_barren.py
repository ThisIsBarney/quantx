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

# The derivative is taken with respect to group 0's angle, and group 0 drives nb_g0
# bonds -- 2, 3, 4, 5 at n = 4..10 on the chain. So the generator grows with n, and the
# variance carries a size-dependent generator-norm factor that a polynomial can turn
# into exactly the collapsing local rates this test looks for (a review's point). Every
# series is therefore checked three ways: raw, with the generator's Hilbert-Schmidt
# norm squared divided out (/ nb_g0), and with its operator norm squared (/ nb_g0^2).
NORMS = ((0, "raw"), (1, "/ nb_g0"), (2, "/ nb_g0^2"))


def size_series(w, parity, k):
    return sorted((int(r["n"]), float(r["var_frac"]) / int(r["nb_g0"]) ** k) for r in S
                  if int(r["W"]) == w and int(r["n"]) % 2 == parity)


def judge(w, parity):
    """Local rates under each normalisation, and the verdict they support together."""
    label = f"W={w} {NAME[w]}" + ("" if parity == 0 else " (odd n, 2Sz=1)")
    pts0 = size_series(w, parity, 0)
    if len(pts0) < 3:
        have = ", ".join(str(n) for n, _ in pts0) or "none"
        print(f"    {label:28s}: NOT FITTABLE -- {len(pts0)} point(s) (n = {have}); "
              f"a rate needs at least 3")
        return None
    spreads = []
    for k, name in NORMS:
        lr = local_rates(size_series(w, parity, k))
        spread = max(lr) / min(lr) if min(lr) > 0 else float("inf")
        spreads.append(spread)
        print(f"    {label:28s} {name:9s}: local rates "
              f"{' '.join(f'{r:+.2f}' for r in lr)}   (spread x{spread:.1f})")
    if all(sp > 3.0 for sp in spreads):
        print(f"      -> NOT a single exponential under any normalisation: the rate "
              f"varies by x{min(spreads):.1f} even at best.")
        return False
    print(f"      -> INCONCLUSIVE: under at least one normalisation the rates vary by "
          f"only x{min(spreads):.1f}, so this series neither supports nor rules out an "
          f"exponential.")
    return True


print("\n  decay rate a in Var ~ exp(-a n), per width and charge sector:")
verdicts = {}
for w in ws:
    for parity in (0, 1):
        v = judge(w, parity)
        if v is not None:
            verdicts[(w, parity)] = v
pts0 = size_series(1, 0, 0)
if len(pts0) >= 3:
    f = fit_loglin([n for n, _ in pts0], [v for _, v in pts0])
    if f:
        print(f"\n  A straight-line fit through the raw chain series would return "
              f"a = {f[0]:+.3f} +/- {f[1]:.3f} per site.")
        print("  It is not quoted as an exponent: the local rates above are not constant.")
print()
print("  The variance falls with n at every width that has a size axis. For the chain,")
print("  the local rate collapses under every normalisation, so its decay over n <= 10 is")
print("  not a single exponential -- consistent with a finite-size crossover. A")
print("  derivative with respect to a fixed single bond would remove the generator-norm")
print("  confound entirely; it was not run. The cylinder -- the only width with the")
print("  persistent suppression -- has one even-n point, so for it the question stays")
print("  open. n <= 10 is where this box stops (chi = 4^(n/2) lossless).")
