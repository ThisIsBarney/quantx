"""Audit every CSV-DERIVED VQE figure quoted in docs/universal_mpo.md.

Figures that come from test stdout (the exactness numbers vs the state-vector
reference) are out of scope -- they are not in any CSV.

Three review rounds in a row found numbers in the prose that no longer matched the
data -- stale after a re-run, or simply mistyped (one cell was literally another
row's value). Hand-checking a table of thirty derived quantities does not work, so
this script derives each one from the committed CSVs and asserts the exact string is
present in the doc. Run it after regenerating any VQE data, before committing.

Run: python benchmarks/check_vqe_doc.py   (exit 1 on any mismatch)
"""
import csv, math, os, sys

HERE = os.path.dirname(__file__)
DOCS = os.path.join(HERE, "..", "docs")

def load(name, key):
    path = os.path.join(DOCS, name)
    if not os.path.exists(path):
        sys.exit(f"missing {name}")
    return {tuple(int(r[k]) for k in key): r for r in csv.DictReader(open(path))}

A = load("vqe_hubbard.csv", ("n", "p"))            # part A: depth convergence
B = load("vqe_entanglement.csv", ("n", "p"))       # part B: entropy vs size
L = load("vqe_ladder.csv", ("Lx", "p"))            # part D: ladder at its own optimum
C = list(csv.DictReader(open(os.path.join(DOCS, "vqe_transfer_check.csv"))))
doc = open(os.path.join(DOCS, "universal_mpo.md"), encoding="utf-8",
           errors="surrogateescape").read()

fails = []


def fit_loglin_f(xs, ys):
    """log(y) = c - a x by least squares; returns (a, stderr_a) -- part F only quotes
    this to say it is NOT a legitimate measurement, so the checker has to reproduce the
    same number the prose disowns."""
    m = len(xs)
    if m < 3:
        return None
    ly = [math.log(y) for y in ys]
    mx, my = sum(xs) / m, sum(ly) / m
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ly)) / sxx
    resid = [y - (my + slope * (x - mx)) for x, y in zip(xs, ly)]
    return -slope, math.sqrt(sum(r * r for r in resid) / (m - 2) / sxx)

def expect(label, text):
    if text not in doc:
        fails.append(f"{label}\n    expected in doc: {text}")

def S(n, p):                                        # part B first, part A as fallback
    r = B.get((n, p)) or A[(n, p)]
    return float(r["S_max"])

def rel_err(row):
    return abs(float(row["E_vqe"]) - float(row["E_exact"])) / abs(float(row["E_exact"]))

def frac_chain(n, p):
    """(E_psi0 - E_circ)/(E_psi0 - E_exact); |psi_0> is the p=1 identity circuit."""
    e0, ec = float(A[(n, 1)]["E_vqe"]), float(A[(n, p)]["E_vqe"])
    return (e0 - ec) / (e0 - float(A[(n, p)]["E_exact"]))

# ---- part B: entropy table, S/n row, logarithmic fit ----
NS = [8, 12, 16, 24, 32]
Sv = [S(n, 8) for n in NS]
expect("part B entropy row",
       "| largest bond entropy S (p=8) | " +
       " | ".join(f"{v:.3f}" for v in Sv[:-1]) + f" | **{Sv[-1]:.3f}** |")
expect("part B S/n row",
       "| S / n (n = sites) | " +
       " | ".join(f"{Sv[i]/n:.3f}" for i, n in enumerate(NS[:-1])) +
       f" | **{Sv[-1]/NS[-1]:.3f}** |")
xs = [math.log(n) for n in NS]
k = len(xs)
b = (k*sum(x*y for x, y in zip(xs, Sv)) - sum(xs)*sum(Sv)) / (k*sum(x*x for x in xs) - sum(xs)**2)
a = (sum(Sv) - b*sum(xs)) / k
dev = max(abs(v - (a + b*x)) for x, v in zip(xs, Sv))
expect("part B log fit", f"`S ≈ {b:.3f} ln n + {a:.3f}` reproduces every point to **{dev:.3f}**")

# ---- part A: the one row the doc tabulates in full ----
MINUS, ARROW, EMDASH = "−", "→", "—"   # the doc's typography

def sci(v):                    # doc convention: 1.1e-2, not python's 1.1e-02
    return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e+")

def num(v, nd):                # minus sign, not hyphen
    return f"{v:.{nd}f}".replace("-", MINUS)
def row_in_table(header_marker, prefix):
    """Cells of the row starting with `prefix`, searched only inside the markdown table
    whose header line contains `header_marker`. Part F reuses the same row labels in
    three different tables (variance vs depth, variance vs size, local decay rate), so a
    whole-document search would silently read the wrong one."""
    lines = doc.split(chr(10))
    for i, line in enumerate(lines):
        if header_marker not in line or not line.lstrip().startswith("|"):
            continue
        for row in lines[i + 1:]:
            if not row.lstrip().startswith("|"):
                break
            if row.startswith(prefix):
                return [c.strip().replace("*", "")
                        for c in row.strip().strip("|").split("|")]
    return None


def row_cells(prefix):
    """The doc's table cells for the row starting with `prefix`, bold markers and
    whitespace stripped -- comparing cell by cell survives **bolding** of whichever
    cell an author decided to emphasise."""
    for line in doc.split(chr(10)):
        if line.startswith(prefix):
            return [c.strip().replace("*", "") for c in line.strip().strip("|").split("|")]
    return None

for n in (4, 6, 8):            # the doc tabulates all three, not just n=8
    want = [sci(float(A[(n, p)]["err_site"])) for p in (2, 4, 6, 8)]
    got = row_cells(f"| n={n} ({2*n} qubits) |")
    if got is None:
        fails.append(f"part A n={n} row: not found in the doc")
    elif got[1:] != want:
        fails.append(f"part A n={n} row: doc {got[1:]} vs csv {want}")

# part B's correlation-energy row is fully derivable too
expect("part B frac row",
       "| correlation energy captured | " +
       " | ".join(f"{float(B[(n, 8)]['frac']):.0%}" for n in NS[:-1]).replace("%", " %") +
       f" | **{float(B[(NS[-1], 8)]['frac']):.0%}** |".replace("%", " %"))

# ---- part D / geometry: matched pairs at each pair's deepest available depth ----
pairs = []
for n, lx in ((6, 3), (8, 4)):
    pm = max(p for (l, p) in L if l == lx)
    pairs.append(dict(n=n, lx=lx, p=pm,
                      ce=rel_err(A[(n, pm)]), le=rel_err(L[(lx, pm)]),
                      cf=frac_chain(n, pm), lf=float(L[(lx, pm)]["frac"]),
                      cs=S(n, pm), ls=float(L[(lx, pm)]["S_max"])))
q = pairs
expect("geometry error row",
       f"| relative energy error | {q[0]['ce']:.2%} | **{q[0]['le']:.2%}** | "
       f"{q[1]['ce']:.2%} | **{q[1]['le']:.1%}** |".replace("%", " %"))
expect("geometry frac row",
       f"| correlation energy captured | {q[0]['cf']:.1%} | **{q[0]['lf']:.1%}** | "
       f"{q[1]['cf']:.1%} | **{q[1]['lf']:.1%}** |".replace("%", " %"))
expect("geometry entropy row",
       f"| largest bond entropy S | {q[0]['cs']:.3f} | {q[0]['ls']:.3f} | "
       f"{q[1]['cs']:.3f} | {q[1]['ls']:.3f} |")
expect("geometry ratio row",
       f"| ratio | | **×{q[0]['le']/q[0]['ce']:.1f} error**, ×{q[0]['ls']/q[0]['cs']:.2f} S | | "
       f"**×{q[1]['le']/q[1]['ce']:.1f} error**, ×{q[1]['ls']/q[1]['cs']:.2f} S |")

# ---- the two caveats on the entropy claim ----
r1 = [float(L[(d['lx'], 1)]["S_max"]) / S(d['n'], 1) for d in q]
expect("p=1 identity ratios", f"×{r1[0]:.2f} and ×{r1[1]:.2f}")
# the doc frames this as a RATIO excess, so the check must use the same reading;
# the absolute-difference reading (S_L-S_C at p=1 over the same at p_final) gives a
# LARGER share, so this formula understates the caveat rather than inflating it
share = [100*(r1[i] - 1) / ((q[i]['ls']/q[i]['cs']) - 1) for i in (0, 1)]
expect("state-prep share",
       f"**{min(share):.0f}–{max(share):.0f} % of the excess entropy")

# ---- part C transfer check ----
for r in C:
    # the whole row, not just S_transfer: a mutation harness showed the reopt and
    # energy columns were passing unchecked
    p_ = int(r["p"])
    if p_ in (2, 4):           # deeper rows are the "Adam found nothing" placeholders
        dS = float(r["dS"])
        expect(f"part C p={p_}",
               f"| {p_} | {float(r['S_transfer']):.3f} | {float(r['S_reopt']):.3f} | "
               f"{'+' if dS >= 0 else MINUS}{abs(dS):.3f} | "
               f"{num(float(r['E_transfer']), 4)} {ARROW} {num(float(r['E_reopt']), 4)} |")
    else:
        expect(f"part C p={p_}",
               f"| {p_} | {float(r['S_transfer']):.3f} | {EMDASH} | {EMDASH} |")
    # and the prose figure for the re-optimisation energy shift
    if p_ == 2:
        shift = abs(float(r["E_reopt"]) - float(r["E_transfer"])) / abs(float(r["E_transfer"]))
        expect("part C prose shift", f"moves the energy by {shift:.2%}".replace("%", " %"))

# ---- remaining CSV-derivable prose figures ----
dS_max = max(abs(float(r["dS"])) for r in C)
expect("part C dS bound", f"S by ≤{dS_max:.3f}")
p4 = next(r for r in C if int(r["p"]) == 4)
shift4 = abs(float(p4["E_reopt"]) - float(p4["E_transfer"])) / abs(float(p4["E_transfer"]))
expect("part C p=4 shift", f"and {shift4:.2%}".replace("%", " %"))
worst = float(B[(32, 8)]["trunc_err"])
expect("part B discarded weight", f"discarded weight of **{worst:.1e}**".replace("e-", "e-"))
expect("part D chain frac", f"optimum already captures {float(B[(8, 8)]['frac']):.0%}".replace("%", " %"))
# depth-stability ranges quoted in the geometry prose
for d, lo_hi in ((q[0], "12 qubits"), (q[1], "16 qubits")):
    rr = [rel_err(L[(d["lx"], pp)]) / rel_err(A[(d["n"], pp)])
          for pp in sorted(pp for (l, pp) in L if l == d["lx"] and pp >= 2)]
    expect(f"error-ratio range {lo_hi}", f"×{min(rr):.1f}–{max(rr):.1f} at {lo_hi}"
           if lo_hi == "12 qubits" else f"×{min(rr):.1f}–{max(rr):.1f} at 16")
expect("ln 16 cap", f"ln 16 = {math.log(16):.4f}")

# ---- part E: the width scan, all three geometries in one representation ----
Wd = load("vqe_width.csv", ("W", "p"))
ws = sorted({w for (w, _) in Wd})
ps = sorted({p for (_, p) in Wd})
pw = max(ps)
# Part E writes its CSV row-by-row over a ~45-minute run, so a truncated file is the
# normal failure mode, not an exotic one. Report the hole as a mismatch like any other
# rather than dying in a traceback -- the caller cannot tell a stale doc from a partial
# scan if the script crashes.
_missing = set()
def row(w, p):
    r = Wd.get((w, p))
    if r is None and (w, p) not in _missing:
        _missing.add((w, p))
        fails.append(f"part E: docs/vqe_width.csv has no row for W={w} p={p}; "
                     f"the doc quotes the full {len(ws)}x{len(ps)} scan")
    return r
def wcells(fmt):
    return " | ".join(fmt(row(w, pw)) for w in ws if row(w, pw))

def pct2(r): return format(rel_err(r), ".2%").replace("%", " %")
def pct1(r): return format(float(r["frac"]), ".1%").replace("%", " %")
def smid3(r): return format(float(r["S_mid"]), ".3f")
def nang(r): return str(int(r["nangle"]) // pw)

expect("width error row", f"| relative energy error | {wcells(pct2)} |")
expect("width frac row", f"| correlation energy captured | {wcells(pct1)} |")
expect("width S_mid row", f"| entropy across the balanced cut | {wcells(smid3)} |")
expect("width angle row", f"| angles per layer | {wcells(nang)} |")
b1, bw = row(ws[0], pw), row(ws[-1], pw)
if b1 and bw:
    expect("width headline ratio", f"×{rel_err(bw)/rel_err(b1):.1f} the chain's error")
expect("width bond row", f"| bonds | {wcells(lambda r: r['nbond'])} |")
# The entropy claim is that the balanced-cut ordering is NOT the width ordering, so
# check the shape of the data, not a ratio: the peak must be at the ladder and it must
# hold at every depth, or the prose is wrong and must be rewritten rather than retuned.
for p_ in ps:
    have = [w for w in ws if row(w, p_)]
    if have and max(have, key=lambda w: float(row(w, p_)["S_mid"])) != 2:
        fails.append(f"part E: S_mid does not peak at the ladder at p={p_}; "
                     f"the doc claims it does at every depth")
if row(2, pw) and row(3, pw):
    expect("width entropy peak",
           f"peaks at the *ladder* ({float(row(2, pw)['S_mid']):.3f}), "
           f"not the cylinder ({float(row(3, pw)['S_mid']):.3f})")
    expect("cylinder smax vs smid",
           f"its S_max ({float(row(3, pw)['S_max']):.3f}) is well above its S_mid "
           f"({float(row(3, pw)['S_mid']):.3f})")
# the energy ordering is the surviving claim -- assert it at every depth the doc claims
for p_ in ps:
    if p_ < 2 or not all(row(w, p_) for w in ws):
        continue
    fr = [float(row(w, p_)["frac"]) for w in ws]
    if not all(fr[i] > fr[i+1] for i in range(len(fr)-1)):
        fails.append(f"part E: frac is not monotone in W at p={p_} ({fr}); "
                     f"the doc claims the ordering holds at every depth p>=2")
# The p=1 claim is the one that already went wrong once: the prose said the optimiser
# returns the identity at all three widths, but the cylinder lands slightly below it.
# Quote both numbers from the data so the distinction cannot rot back.
if all(row(w, 1) for w in ws):
    id_w = [w for w in ws if float(row(w, 1)["frac"]) == 0.0]
    nz_w = [w for w in ws if float(row(w, 1)["frac"]) != 0.0]
    if id_w != [1, 2] or nz_w != [3]:
        fails.append(f"part E: at p=1 the exact-identity widths are {id_w} and the "
                     f"non-identity {nz_w}; the doc says chain+ladder and cylinder")
    else:
        expect("p=1 identity", f"the identity exactly\n  (frac = {float(row(1,1)['frac']):.4f})")
        expect("p=1 cylinder", f"the cylinder reaches frac = {float(row(3,1)['frac']):.4f}")
# ratio ranges and the seeding-scatter figures the prose quotes
for w, lbl in ((2, "ladder-to-chain"), (3, "cylinder-to-chain")):
    rr = [rel_err(row(w, p_)) / rel_err(row(1, p_))
          for p_ in ps if p_ >= 2 and row(w, p_) and row(1, p_)]
    if rr:
        expect(f"part E {lbl} ratio range", f"×{min(rr):.1f}–{max(rr):.1f}")
# The seeding-scatter claim compares this scan's chain point against part A's at the
# same size and depth. vqe_hubbard.csv stores no frac, so recover the shared |psi0>
# energy from the width row (which has both frac and the energies) and apply it to part
# A. That the two runs agree on E_exact bit-for-bit is itself the check that they are
# the same system -- without it, comparing their fracs would be meaningless.
scan_chain, partA_chain = row(1, pw), A.get((6, 8))
if scan_chain and partA_chain:
    f_scan, e_scan, x_scan = (float(scan_chain[k]) for k in ("frac", "E_vqe", "E_exact"))
    e_a, x_a = float(partA_chain["E_vqe"]), float(partA_chain["E_exact"])
    if abs(x_a - x_scan) > 1e-12:
        fails.append(f"part E: the scan's chain E_exact ({x_scan}) and part A's ({x_a}) "
                     f"differ, so the doc's seeding-scatter comparison is not like-for-like")
    else:
        e0 = (e_scan - f_scan * x_scan) / (1 - f_scan)
        f_a = (e_a - e0) / (x_a - e0)
        expect("part E scan chain frac", f"({f_scan:.1%} at p=8)".replace("%", " %"))
        expect("part E part-A chain frac", f"({f_a:.1%})".replace("%", " %"))
        expect("part E seeding scatter", f"differ by {abs(f_scan - f_a) * 100:.1f} pp")
nfevs = [int(r["nfev"]) for r in Wd.values()]
expect("part E nfev range", f"spent {min(nfevs)}–{max(nfevs)} circuit evaluations")
# the whole point of part E is that nothing but the graph differs -- pin the
# invariants the comparison rests on, so a re-run with a changed size fails loudly
for w in ws:
    r = row(w, pw)
    if not r:
        continue
    # trunc is exact zero or denormal-small here; 1e-14 is "no state was discarded",
    # not a tolerance on the result
    if int(r["qubits"]) != 12 or int(r["chi"]) > 64 or float(r["trunc"]) > 1e-14:
        fails.append(f"part E W={w}: not the matched lossless setup "
                     f"(qubits={r['qubits']} chi={r['chi']} trunc={r['trunc']})")

# ---- part F: the barren-plateau diagnostic ----------------------------------
# Part E's own new prose slipped a claim past this script once (p=1 "frac = 0.0000 at
# all three widths" while the CSV said 0.0002 for the cylinder). The lesson was not
# "write a checker" -- it was "extend it to every new claim" -- so part F's table, its
# ratios, its calibration row, its local decay rates and the exponent it REFUSES to
# quote are all asserted here.
F = load("vqe_barren.csv", ("W", "p"))
FS = load("vqe_barren_size.csv", ("W", "n"))
fws = sorted({w for (w, _) in F})
fps = sorted({p for (_, p) in F})


def fcells(fmt, p_):
    return " | ".join(fmt(F[(w, p_)]) for w in fws if (w, p_) in F)


def frow(label, w, fmt):
    want = [fmt(F[(w, p_)]) for p_ in fps if (w, p_) in F]
    got = row_in_table("Var[&#8706;frac", f"| {label} |") or           row_in_table("Var[∂frac", f"| {label} |")
    if got is None:
        fails.append(f"part F {label} row: row not found in the depth table")
    elif got[1:] != want:
        fails.append(f"part F {label} row: doc {got[1:]} vs csv {want}")


frow("W=1 chain", 1, lambda r: format(float(r["var_frac"]), ".3f"))
frow("W=2 ladder", 2, lambda r: format(float(r["var_frac"]), ".3f"))
frow("W=3 cylinder", 3, lambda r: format(float(r["var_frac"]), ".3f"))
ratios = " | ".join(
    format(float(F[(3, p_)]["var_frac"]) / float(F[(1, p_)]["var_frac"]), ".2f")
    for p_ in fps if (3, p_) in F and (1, p_) in F)
expect("part F ratio row", f"| ratio W3/W1 | {ratios} |")

# the instrument: floor, worst signal-to-noise, and the group-size confound
floor = float(F[(fws[0], fps[0])]["floor"])
expect("part F noise floor", f"and **{floor:.1e}** is the floor".replace("e-0", "e-"))
snr_min = min(float(r["snr"]) for r in F.values())
expect("part F worst snr", f"at least **{snr_min:.1e} ×**".replace("e+0", "e"))
g0 = {int(r["nb_g0"]) for r in F.values()}
if len(g0) != 1:
    fails.append(f"part F: group-0 bond counts differ across widths ({sorted(g0)}), so "
                 f"the doc's claim that the generator norm is identical is false")
else:
    expect("part F group size", f"holds **{g0.pop()} bonds at every width**")

# the cylinder's suppression must be significant at EVERY depth, or the prose is wrong
for p_ in fps:
    if (3, p_) not in F or (1, p_) not in F:
        continue
    v3, v1 = float(F[(3, p_)]["var_frac"]), float(F[(1, p_)]["var_frac"])
    e3, e1 = float(F[(3, p_)]["se_frac"]), float(F[(1, p_)]["se_frac"])
    r = v3 / v1
    rse = r * math.sqrt((e3 / v3) ** 2 + (e1 / v1) ** 2)
    if abs(r - 1.0) <= 2 * rse:
        fails.append(f"part F: the cylinder/chain ratio at p={p_} is {r:.2f}+/-{rse:.2f}, "
                     f"not significant; the doc claims six independent confirmations")
    if not (0.28 <= r <= 0.37):
        fails.append(f"part F: the cylinder/chain ratio at p={p_} is {r:.2f}, outside the "
                     f"0.30-0.35 band the doc quotes")

# the size axis: the local rates the doc tabulates, and the exponent it refuses to quote
def rates_of(w, parity):
    pts = sorted((n, float(r["var_frac"])) for (ww, n), r in FS.items()
                 if ww == w and n % 2 == parity)
    return pts, [math.log(pts[i][1] / pts[i + 1][1]) / (pts[i + 1][0] - pts[i][0])
                 for i in range(len(pts) - 1)]


for lbl, w, par in (("W=1 chain", 1, 0), ("W=1 chain (odd n, 2Sz=1)", 1, 1),
                    ("W=2 ladder", 2, 0)):
    pts, lr = rates_of(w, par)
    if len(lr) != 3:
        fails.append(f"part F: {lbl} has {len(pts)} size points, the doc tabulates 4")
        continue
    # compare cell by cell with bold markers stripped: which cell an author emphasises
    # is a presentation choice, and asserting it would make the checker enforce prose
    # style instead of data
    got = row_in_table("local rate a", f"| {lbl} |")
    want = [f"{x:+.2f}" for x in lr]
    if got is None:
        fails.append(f"part F local rates {lbl}: row not found in the doc")
    elif got[1:] != want:
        fails.append(f"part F local rates {lbl}: doc {got[1:]} vs csv {want}")

pts0, lr0 = rates_of(1, 0)
if lr0:
    expect("part F rate collapse", f"falls by **×{max(lr0)/min(lr0):.1f}**")
    f = fit_loglin_f([n for n, _ in pts0], [v for _, v in pts0])
    if f:
        expect("part F refused exponent", f"`a = {f[0]:+.3f} ± {f[1]:.3f}` per site")
# the even-n variance table
for lbl, w in (("W=1 chain", 1), ("W=2 ladder", 2)):
    want = [format(float(FS[(w, n)]["var_frac"]), ".3f")
            for n in sorted(n for (ww, n) in FS if ww == w and n % 2 == 0)]
    got = row_in_table("n (even, 2Sz=0)", f"| {lbl} |")
    if got is None:
        fails.append(f"part F size row {lbl}: row not found in the size table")
    elif got[1:] != want:
        fails.append(f"part F size row {lbl}: doc {got[1:]} vs csv {want}")
# and the cylinder must still have exactly one even-n point, or the "unanswered" stands wrong
cyl_even = [n for (w, n) in FS if w == 3 and n % 2 == 0]
if len(cyl_even) != 1:
    fails.append(f"part F: the cylinder now has {len(cyl_even)} even-n size points "
                 f"({sorted(cyl_even)}); the doc says one, so the 'unanswered' framing "
                 f"and the 'what would settle it' paragraph need rewriting")

if fails:
    print(f"{len(fails)} MISMATCH(es) between docs/universal_mpo.md and the CSVs:\n")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all quoted VQE figures in docs/universal_mpo.md reproduce from the CSVs")
