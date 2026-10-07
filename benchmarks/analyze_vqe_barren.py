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
print("\n\nSIZE AXIS  (and why it settles nothing -- read the normalisation note)")
S = load("vqe_barren_size.csv")
Ns = int(S[0]["nsamp"])
smeas = [float(r["se_frac"]) / float(r["var_frac"]) for r in S]
print(f"  {Ns} samples/point, measured SE {min(smeas):.1%}-{max(smeas):.1%} per "
      f"variance (a Gaussian would claim {math.sqrt(2.0/Ns):.1%})")
print()
print("    W    n  qubits   chi    Var[dfrac/dtheta]   sector")
for r in sorted(S, key=lambda r: (int(r["W"]), int(r["n"]))):
    n = int(r["n"])
    print(f"  {int(r['W']):3d} {n:4d} {int(r['qubits']):7d} {int(r['chi']):5d}   "
          f"{float(r['var_frac']):16.5f}   2Sz={n % 2}")

# TWO size-dependent factors sit between var_frac and anything a reader would call
# "the gradient variance of this ansatz at size n", and BOTH have to come out before a
# decay rate means anything.
#
#  (a) The cost. var_frac = var_raw / corr^2 with corr = E_psi0 - E_exact, the
#      correlation energy -- an EXTENSIVE quantity. corr/n is 0.370, 0.351, 0.340, 0.333
#      at n = 4, 6, 8, 10, i.e. flat, so corr^2 grows like n^2 and var_frac carries a
#      built-in 1/n^2. That normalisation is what makes the three WIDTHS comparable at
#      fixed n -- each geometry's own correlation energy -- and it is exactly what must
#      not be carried ACROSS n. Using it there manufactures a decay.
#  (b) The generator. theta_g drives group 0, which holds nb_g0 bonds: 2, 3, 4, 5 at
#      n = 4..10. So ||H_g|| grows like n/2 on top of (a).
#
# var_raw is the un-normalised quantity and it is in the same CSV. Reported first, and
# it does not decay: the chain is flat from n=6 and RISES at n=10. Everything downstream
# of that is a statement about the normalisation, not about trainability.
NORMS = ((None, 0, "raw dE/dtheta"), (None, 1, "raw / nb_g0"),
         ("frac", 0, "frac (1/n^2 in it)"), ("frac", 1, "frac / nb_g0"),
         ("frac", 2, "frac / nb_g0^2"))


def size_series(w, parity, k, cost=None):
    col = "var_frac" if cost == "frac" else "var_raw"
    return sorted((int(r["n"]), float(r[col]) / int(r["nb_g0"]) ** k) for r in S
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
    raw_rates = None
    for cost, k, name in NORMS:
        lr = local_rates(size_series(w, parity, k, cost))
        if cost is None and k == 0:
            raw_rates = lr
        spread = (max(lr) / min(lr)) if min(lr) > 0 else float("inf")
        tag = "" if min(lr) > 0 else "   <-- NOT monotone: a rate <= 0 means the variance RISES"
        print(f"    {label:28s} {name:19s}: local rates "
              f"{' '.join(f'{r:+.2f}' for r in lr)}   (spread x{spread:.1f}){tag}")
    # The verdict rests on the RAW series. A decay that only appears after dividing by an
    # extensive cost is a property of the divisor.
    if min(raw_rates) <= 0:
        print(f"      -> NO DECAY TO CHARACTERISE: the raw gradient variance does not "
              f"fall monotonically with n (a local rate is {min(raw_rates):+.2f}), so "
              f"this series")
        print(f"         supports no statement about exponential decay in either "
              f"direction. The fall seen in `frac` is the 1/n^2 the cost carries.")
        return "no-decay"
    spread = max(raw_rates) / min(raw_rates)
    if spread > 3.0:
        print(f"      -> raw variance falls, but NOT as a single exponential: the rate "
              f"varies by x{spread:.1f}.")
        return "not-exponential"
    print(f"      -> raw variance falls with a roughly constant rate (spread "
          f"x{spread:.1f}); an exponent would be defensible here.")
    return "exponential"


print("\n  decay rate a in Var ~ exp(-a n), per width and charge sector:")
verdicts = {}
for w in ws:
    for parity in (0, 1):
        v = judge(w, parity)
        if v is not None:
            verdicts[(w, parity)] = v
pts0 = size_series(1, 0, 0, "frac")
if len(pts0) >= 3:
    f = fit_loglin([n for n, _ in pts0], [v for _, v in pts0])
    if f:
        print(f"\n  A straight-line fit through the frac-normalised chain series would "
              f"return a = {f[0]:+.3f} +/- {f[1]:.3f} per site. It is not an exponent:")
        print(f"  the series it fits carries the cost's own 1/n^2.")

# The summary is DERIVED. An earlier version ended in eight hard-coded sentences
# asserting the study's conclusions while the computed verdicts went unread -- the exact
# stale-claim failure mode check_vqe_doc.py exists to prevent, reintroduced in the
# script whose stated purpose is that the distinction be drawn by arithmetic.
print()
tally = {}
for v in verdicts.values():
    tally[v] = tally.get(v, 0) + 1
print(f"  SIZE-AXIS VERDICT over {len(verdicts)} fittable series: " +
      ", ".join(f"{k} x{v}" for k, v in sorted(tally.items())))
if tally.get("no-decay"):
    print(f"  {tally['no-decay']} of them show NO monotone decay in the raw gradient")
    print("  variance, so this size axis cannot be used to claim a barren plateau OR to")
    print("  rule one out. Any decay quoted from `frac` is the extensive cost's 1/n^2.")
nofit = [f"W={w}" + ("" if par == 0 else " odd") for w in ws for par in (0, 1)
         if (w, par) not in verdicts and
         any(int(r["W"]) == w and int(r["n"]) % 2 == par for r in S)]
if nofit:
    print(f"  Series with too few points to fit at all: {', '.join(nofit)}.")
print("  What would fix it: differentiate a FIXED single bond (removes the generator "
      "factor)")
print("  and report dE/dtheta per bond (removes the cost factor), at sizes a lossless "
      "chi = 4^(n/2)")
print("  can still reach -- n <= 10 on this box.")
