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
import csv, glob, math, os, re, subprocess, sys

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
# The public Results page repeats CSV-derived figures AND claims this script protects
# them. That claim was false for the page itself: only universal_mpo.md was ever read, so
# regenerating a CSV could leave the published page stale with qx_vqe_doc_figures still
# green. A page that advertises enforcement has to be inside it.
RES_PATH = os.path.join(DOCS, "source", "results.rst")
res = (open(RES_PATH, encoding="utf-8", errors="surrogateescape").read()
       if os.path.exists(RES_PATH) else None)

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


def expect_res(label, text):
    """Assert against docs/source/results.rst -- the page the docs site publishes."""
    if res is None:
        # once, not once per call: 13 identical lines would bury the real report
        msg = "docs/source/results.rst is missing, but the docs site publishes it"
        if msg not in fails:
            fails.append(msg)
    elif text not in res:
        fails.append(f"results page: {label}\n    expected in results.rst: {text}")

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
# Part E's prose once slipped a claim past this script, and part F then slipped a bigger
# one past a first review: the size-axis "decay" was the 1/n^2 of an extensive cost
# normalisation. So the assertions below pin the things that went wrong, not only the
# things that looked checkable -- the raw series next to the normalised one, the cost
# denominator's extensivity, both CSVs' signal-to-noise, and the 2-sigma the prose quotes.
F = load("vqe_barren.csv", ("W", "p"))
FS = load("vqe_barren_size.csv", ("W", "n"))
fws = sorted({w for (w, _) in F})
fps = sorted({p for (_, p) in F})


def frow(label, w, fmt, table):
    want = [fmt(F[(w, p_)]) for p_ in fps if (w, p_) in F]
    got = row_in_table(table, f"| {label} |")
    if got is None:
        fails.append(f"part F {label} row: not found in the table matching {table!r}")
    elif [norm(c) for c in got[1:]] != [norm(c) for c in want]:
        fails.append(f"part F {label} row: doc {got[1:]} vs csv {want}")


DEPTH_TBL = "Var[\u2202frac/\u2202\u03b8] | p=2"


def norm(c):
    """The doc sets minus as U+2212 while we format with an ASCII hyphen, and bold
    markers are presentation. Normalise both before comparing cells, or the checker
    enforces typography instead of numbers."""
    return c.replace("\u2212", "-").replace("*", "").strip()

frow("W=1 chain", 1, lambda r: format(float(r["var_frac"]), ".3f"), DEPTH_TBL)
frow("W=2 ladder", 2, lambda r: format(float(r["var_frac"]), ".3f"), DEPTH_TBL)
frow("W=3 cylinder", 3, lambda r: format(float(r["var_frac"]), ".3f"), DEPTH_TBL)
for lbl, col in (("ratio W3/W1", "var_frac"),
                 ("ratio W3/W1, raw \u2202E/\u2202\u03b8", "var_raw")):
    cells = " | ".join(format(float(F[(3, p_)][col]) / float(F[(1, p_)][col]), ".2f")
                       for p_ in fps if (3, p_) in F and (1, p_) in F)
    expect(f"part F {lbl} row", f"| {lbl} | {cells} |")

# the instrument -- floor, and the worst SNR of EACH csv (the old check used the depth
# csv only, and certified a "5.4e8" bound that a committed size row violated at 1.8e8)
floor = float(F[(fws[0], fps[0])]["floor"])
expect("part F noise floor", f"**{floor:.1e}** \u2014 is the one".replace("e-0", "e-"))
for lbl, tbl in (("depth", F), ("size", FS)):
    m = min(float(r["snr"]) for r in tbl.values())
    expect(f"part F worst snr ({lbl})", f"**{m:.1e} \u00d7**".replace("e+0", "e"))

# the Richardson calibration rows are stdout-only, so parse the committed log rather
# than leaving the one table in this section that nothing verified
cal = {}
log = os.path.join(HERE, "vqe_barren.log")
if os.path.exists(log):
    for line in open(log, encoding="utf-8", errors="replace"):
        if line.startswith("CAL "):
            _, w_, h_, _g, err = line.split()
            cal.setdefault(int(w_), []).append((float(h_), float(err)))
    for w_, name in ((1, "W=1 chain"), (3, "W=3 cylinder")):
        if w_ not in cal:
            continue
        want = [f"{e:.1e}".replace("e-0", "e-") for _h, e in cal[w_]]
        got = row_in_table("| h |", f"| {name} |")
        if got is None:
            fails.append(f"part F calibration row {name}: not found")
        elif got[1:] != want:
            fails.append(f"part F calibration row {name}: doc {got[1:]} vs log {want}")
else:
    fails.append("part F: benchmarks/vqe_barren.log is missing, so the calibration "
                 "table in the doc is unverifiable")

# the cylinder's suppression: significant at every depth, and the 2-SIGMA the doc quotes
# (the first version printed 1 sigma and labelled it 2)
sig2 = []
for p_ in fps:
    if (3, p_) not in F or (1, p_) not in F:
        continue
    v3, v1 = float(F[(3, p_)]["var_frac"]), float(F[(1, p_)]["var_frac"])
    e3, e1 = float(F[(3, p_)]["se_frac"]), float(F[(1, p_)]["se_frac"])
    r = v3 / v1
    rse = r * math.sqrt((e3 / v3) ** 2 + (e1 / v1) ** 2)
    sig2.append(2 * rse)
    if abs(r - 1.0) <= 2 * rse:
        fails.append(f"part F: cylinder/chain at p={p_} is {r:.2f}+/-{2*rse:.2f} (2s), "
                     f"not significant; the doc claims six confirmations")
    if not (0.30 <= round(r, 2) <= 0.35):
        fails.append(f"part F: cylinder/chain at p={p_} is {r:.2f}, outside the "
                     f"0.30-0.35 the doc quotes")
if sig2:
    expect("part F 2-sigma", f"measured 2\u03c3 of \u00b1{min(sig2):.2f}\u2013"
                             f"{max(sig2):.2f} at every depth")

# the group-size confound, stated as a range over the size axis
g0 = sorted({(int(r["n"]), int(r["nb_g0"])) for r in FS.values() if int(r["W"]) == 1})
if g0:
    ev = [b for n, b in g0 if n % 2 == 0]
    expect("part F generator growth",
           "**" + ", ".join(str(b) for b in ev) + " bonds at n = 4\u20269**".replace("9", "10"))

# THE retraction: corr must be extensive, and the raw series must not be monotone
chain_even = sorted((n, r) for (w, n), r in FS.items() if w == 1 and n % 2 == 0)
if len(chain_even) >= 3:
    corr_over_n = []
    for n, r in chain_even:
        c = math.sqrt(float(r["var_raw"]) / float(r["var_frac"]))
        corr_over_n.append(c / n)
    expect("part F corr extensive",
           "`corr/n` = " + ", ".join(f"{c:.3f}" for c in corr_over_n))
    raw = [float(r["var_raw"]) for _n, r in chain_even]
    frac = [float(r["var_frac"]) for _n, r in chain_even]
    got = row_in_table("W=1 chain, even n", "| raw Var")
    want = [f"{v:.3f}" for v in raw[:-1]] + [f"**{raw[-1]:.3f}**"]
    if got is None:
        fails.append("part F raw size row: not found")
    elif [norm(c) for c in got[1:]] != [norm(c) for c in want]:
        fails.append(f"part F raw size row: doc {got[1:]} vs csv {want}")
    got = row_in_table("W=1 chain, even n", "| Var[\u2202frac")
    if got and got[1:] != [f"{v:.3f}" for v in frac]:
        fails.append(f"part F frac size row: doc {got[1:]} vs csv "
                     f"{[f'{v:.3f}' for v in frac]}")

    def lrates(vals):
        ns = [n for n, _ in chain_even]
        return [math.log(vals[i] / vals[i + 1]) / (ns[i + 1] - ns[i])
                for i in range(len(vals) - 1)]

    for lbl, vals in (("local rate (raw)", raw), ("local rate (frac)", frac)):
        got = row_in_table("W=1 chain, even n", f"| {lbl} |")
        want = [""] + [f"{x:+.2f}" for x in lrates(vals)]
        if got is None:
            fails.append(f"part F {lbl} row: not found")
        elif [norm(c) for c in got[1:]] != [norm(c) for c in want]:
            fails.append(f"part F {lbl} row: doc {got[1:]} vs csv {want}")
    # the claim the whole retraction rests on
    if min(lrates(raw)) > 0:
        fails.append("part F: the raw chain series IS monotone decreasing, so the doc's "
                     "retraction ('flat from n=6 and rises at n=10') is wrong and the "
                     "section needs rewriting back")

# EVERY committed row must be reproducible by the documented default command. The n cap
# the default chi cap implies is 10 (4^5 = 1024); a row above it can only have come from
# a non-default invocation, and one such row -- n=11 -- had been supplying the final
# negative odd-sector rate behind a claim in the prose. Data the committed protocol
# cannot regenerate is not data the doc may lean on.
DEFAULT_N_CAP = 10
# ...and at the default sample counts. The committed artifacts were made at 900 (depth)
# and max(50, 900/3) = 300 (size); if the source default drifts from these, the
# documented command silently produces different variances and error bars from the ones
# the prose quotes. Read the default out of the source rather than restating it here.
_src = open(os.path.join(HERE, "..", "src", "vqe_showcase.cpp"), encoding="utf-8",
            errors="replace").read()
_m = re.search(r"nsamp = bp_nsamp > 0 \? bp_nsamp : \(quick \? \d+ : (\d+)\)", _src)
if not _m:
    fails.append("part F: cannot find part F's default sample count in "
                 "src/vqe_showcase.cpp, so the committed CSVs' reproducibility is "
                 "unverifiable")
else:
    _def = int(_m.group(1))
    for lbl, tbl, want in (("depth", F, _def), ("size", FS, max(50, _def // 3))):
        got = {int(r["nsamp"]) for r in tbl.values()}
        if got != {want}:
            fails.append(f"part F: the committed {lbl} CSV has nsamp={sorted(got)} but "
                         f"the source default yields {want}; the documented command "
                         f"would not reproduce it")
stray = sorted(n for (w, n) in FS if n > DEFAULT_N_CAP)
if stray:
    fails.append(f"part F: docs/vqe_barren_size.csv has rows at n = {stray}, above the "
                 f"n <= {DEFAULT_N_CAP} the default QX_VQE_BPCHI admits; the committed "
                 f"scan cannot reproduce them")

# The prose now quotes rates WITH errors, significances, and the drop across the series.
# Every one of those is derivable, and leaving them unasserted is how three earlier
# versions of this section kept a wrong number: the checker has to cover the sentence
# that carries the conclusion, not only the table beside it.
if len(chain_even) >= 3:
    ns_e = [n for n, _ in chain_even]
    vr_e = [float(r["var_raw"]) for _n, r in chain_even]
    # se_frac is quoted on var_frac; the same RELATIVE error applies to var_raw
    er_e = [float(r["se_frac"]) / float(r["var_frac"]) * float(r["var_raw"])
            for _n, r in chain_even]
    rates, rerrs = [], []
    for i in range(len(ns_e) - 1):
        dn = ns_e[i + 1] - ns_e[i]
        rates.append(math.log(vr_e[i] / vr_e[i + 1]) / dn)
        rerrs.append(math.sqrt((er_e[i] / vr_e[i]) ** 2 +
                               (er_e[i + 1] / vr_e[i + 1]) ** 2) / dn)
    expect("part F raw rates with errors",
           ", ".join(f"{r:+.2f} \u00b1 {e:.2f}".replace("-", "\u2212", 1)
                     if r < 0 else f"{r:+.2f} \u00b1 {e:.2f}"
                     for r, e in zip(rates, rerrs)).replace("+", "+"))
    # the two significances the first bullet rests on, and the unresolved last step
    expect("part F first-rate sigma", f"{rates[0]:+.2f} is {rates[0]/rerrs[0]:.1f}\u03c3")
    expect("part F second-rate sigma", f"{rates[1]:+.2f} is {rates[1]/rerrs[1]:.1f}\u03c3")
    expect("part F last-rate sigma",
           f"is **{abs(rates[-1]/rerrs[-1]):.1f}\u03c3** from zero")
    # and the drop that carries the "not a single exponential" conclusion
    drop = rates[0] - rates[-1]
    dse = math.sqrt(rerrs[0] ** 2 + rerrs[-1] ** 2)
    expect("part F rate drop",
           f"drops by {drop:.2f} \u00b1 {dse:.2f} \u2014 {drop/dse:.1f}\u03c3")
    if abs(rates[-1]) > 2 * rerrs[-1]:
        fails.append(f"part F: the last raw local rate is {rates[-1]:+.3f}+/-"
                     f"{rerrs[-1]:.3f}, now RESOLVED from zero; the doc says it is not, "
                     f"so the retraction's wording needs revisiting")
    if drop <= 2 * dse:
        fails.append(f"part F: the raw rate drop is {drop:.2f}+/-{dse:.2f}, no longer "
                     f"significant; the doc's 'not a single exponential' rests on it")
    # the frac-minus-raw gap the prose attributes to the 1/n^2
    vf_e = [float(r["var_frac"]) for _n, r in chain_even]
    fr = [math.log(vf_e[i] / vf_e[i + 1]) / (ns_e[i + 1] - ns_e[i])
          for i in range(len(ns_e) - 1)]
    expect("part F frac-minus-raw",
           "(" + ", ".join(f"{f - r:+.2f}" for f, r in zip(fr, rates)) + ")")

chain_odd = sorted((n, r) for (w, n), r in FS.items() if w == 1 and n % 2 == 1)
# ...and the rates are asserted for HOWEVER many points there are. Gating this on
# len >= 4 meant that dropping n=11 silently skipped the check instead of flagging the
# prose that still quoted its rate -- the third time in this PR that the checker stepped
# over precisely the thing that had gone wrong.
if len(chain_odd) >= 3:
    rv = [float(r["var_raw"]) for _n, r in chain_odd]
    ns = [n for n, _ in chain_odd]
    lr = [math.log(rv[i] / rv[i + 1]) / (ns[i + 1] - ns[i]) for i in range(len(rv) - 1)]
    expect("part F odd raw rates",
           "raw local rates " + ", ".join(f"{x:+.2f}".replace("-", "\u2212") for x in lr))
    expect("part F odd point count",
           f"it has three points (n = {', '.join(str(n) for n in ns)})")
elif chain_odd:
    fails.append(f"part F: the odd-n chain series has only {len(chain_odd)} point(s), too "
                 f"few for the local rates the doc tabulates")

# the exponent the prose disowns, and the measured-vs-Gaussian SE on both scans
if len(chain_even) >= 3:
    f_ = fit_loglin_f([n for n, _ in chain_even],
                      [float(r["var_frac"]) for _n, r in chain_even])
    if f_:
        expect("part F disowned exponent",
               f"`a = {f_[0]:+.3f} \u00b1 {f_[1]:.3f}`")
for lbl, tbl in (("depth", F), ("size", FS)):
    rel = [float(r["se_frac"]) / float(r["var_frac"]) for r in tbl.values()]
    n0 = int(next(iter(tbl.values()))["nsamp"])
    expect(f"part F measured SE ({lbl})",
           f"{min(rel)*100:.1f}\u2013{max(rel)*100:.1f} %")
    expect(f"part F gaussian SE ({lbl})", f"{math.sqrt(2.0/n0):.1%}".replace("%", " %"))

# and the cylinder still has exactly one even-n point, or the framing changes
cyl_even = [n for (w, n) in FS if w == 3 and n % 2 == 0]
if len(cyl_even) != 1:
    fails.append(f"part F: the cylinder now has {len(cyl_even)} even-n size points "
                 f"({sorted(cyl_even)}); the doc says one, so its framing needs rewriting")

# ---- the published Results page ---------------------------------------------
# Only values this script can DERIVE are asserted. The page also quotes the GPU margins,
# which live in docs/gpu_acceleration_report.md rather than a CSV, and the Haldane and
# Lieb-Wu reference values, which are external constants -- those are checked below only
# where a CSV can reproduce them.
if res is None:
    # NOT a skip. The block below is wrapped in `res is not None`, so a missing page made
    # every check vanish and the script still printed success -- the same silently-skipping
    # guard this project has been bitten by before. The docs site publishes this page; if
    # it is gone, that is a failure, not a reason to check less.
    fails.append("docs/source/results.rst is missing, but docs.yml publishes it and this "
                 "script is what keeps its figures honest")
else:
    # part E: the width scan's correlation-energy row, as the page states it
    pw_ = max(p for (_, p) in Wd)
    for w_ in sorted({w for (w, _) in Wd}):
        r_ = Wd.get((w_, pw_))
        if r_:
            expect_res(f"W={w_} correlation energy at p={pw_}",
                       format(float(r_["frac"]), ".1%").replace("%", " %"))
    # part F: the cylinder/chain gradient-variance band, both normalisations
    if F:
        pf_ = sorted({p for (_, p) in F})
        rr = [float(F[(3, p_)]["var_frac"]) / float(F[(1, p_)]["var_frac"])
              for p_ in pf_ if (3, p_) in F and (1, p_) in F]
        rw = [float(F[(3, p_)]["var_raw"]) / float(F[(1, p_)]["var_raw"])
              for p_ in pf_ if (3, p_) in F and (1, p_) in F]
        if rr:
            expect_res("cylinder/chain band, dimensionless cost",
                       f"\u00d7{min(rr):.2f}\u2013{max(rr):.2f}")
        if rw:
            expect_res("cylinder/chain band, raw gradient",
                       f"\u00d7{min(rw):.2f}\u2013{max(rw):.2f}")
        floor_ = float(next(iter(F.values()))["floor"])
        expect_res("noise floor", f"{floor_:.1e}".replace("e-0", "e-"))
        snr_ = min(float(r_["snr"]) for r_ in F.values())
        expect_res("worst snr", f"{snr_:.1e} \u00d7".replace("e+0", "e"))
    # the Lieb-Wu table: engine, exact and the difference, per U
    LW = {}
    lw_path = os.path.join(DOCS, "hubbard_lieb_wu.csv")
    if os.path.exists(lw_path):
        for r_ in csv.DictReader(open(lw_path)):
            LW.setdefault(float(r_["U"]), []).append(r_)
        for U_ in sorted(LW):
            big = max(LW[U_], key=lambda r_: int(r_["N"]))
            eng, ex = float(big["e_bulk_incremental"]), float(big["e_liebwu"])
            expect_res(f"Lieb-Wu U={U_:g} engine", f"{eng:.8f}".replace("-", "\u2212"))
            expect_res(f"Lieb-Wu U={U_:g} exact", f"{ex:.8f}".replace("-", "\u2212"))
            expect_res(f"Lieb-Wu U={U_:g} difference",
                       f"{abs(eng - ex):.1e}".replace("e-0", "e-"))
        # The engine column is a finite difference of the two longest chains, and the page
        # must name which two: it called the N = 64 row an "extrapolation" until Codex read
        # the showcase and found (E_N - E_prev)/(N - N_prev).
        _sub = str.maketrans("0123456789", "₀₁₂₃₄"
                                           "₅₆₇₈₉")
        for U_ in sorted(LW):
            n1, n2 = sorted(int(r_["N"]) for r_ in LW[U_])[-2:]
            expect_res(f"Lieb-Wu U={U_:g} difference formula",
                       f"(E{str(n2).translate(_sub)} − E{str(n1).translate(_sub)})"
                       f"/{n2 - n1}")
        # "mainly finite-size error ... a 1/N term ... shrinks as U grows ... fitting it out
        # removes most of the residual": E_N = N e + f + a/N, solved exactly through the three
        # longest chains, must leave less than half the residual at every U, and |a| must
        # fall as U grows.
        def _det3(M):
            return (M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1])
                    - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0])
                    + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]))
        _a = []
        for U_ in sorted(LW):
            R_ = {int(r_["N"]): r_ for r_ in LW[U_]}
            n3 = sorted(R_)[-3:]
            A_ = [[N_, 1.0, 1.0 / N_] for N_ in n3]
            b_ = [float(R_[N_]["E_per_site"]) * N_ for N_ in n3]
            e_, _, a_ = [_det3([[b_[i] if j == k else A_[i][j] for j in range(3)]
                                for i in range(3)]) / _det3(A_) for k in range(3)]
            ex_ = float(R_[n3[-1]]["e_liebwu"])
            inc_ = float(R_[n3[-1]]["e_bulk_incremental"])
            if not abs(e_ - ex_) < 0.5 * abs(inc_ - ex_):
                fails.append(f"results page: at U={U_:g} the 1/N fit leaves {abs(e_ - ex_):.1e} "
                             f"of the {abs(inc_ - ex_):.1e} residual, so 'mainly finite-size' "
                             f"and 'removes most of the residual' no longer hold")
            _a.append(abs(a_))
        if any(lo_ >= hi_ for hi_, lo_ in zip(_a, _a[1:])):
            fails.append(f"results page: the 1/N coefficient |a| = "
                         f"{', '.join(f'{v:.3f}' for v in _a)} does not fall as U grows, so "
                         f"'that term shrinks as U grows' no longer holds")
    # the J1-J2 truncation-sensitivity row the page tabulates
    jj_path = os.path.join(DOCS, "j1j2_dimerization.csv")
    if os.path.exists(jj_path):
        J = {f"{float(r_['j2']):.2f}": r_ for r_ in csv.DictReader(open(jj_path))}
        for key in ("0.00", "0.45", "0.49", "0.51", "0.55"):
            r_ = J.get(key)
            if r_:
                expect_res(f"J1-J2 gap at j2={key}",
                           f"{float(r_['gap_per_site']):.1e}".replace("e-0", "e-"))
        mg = J.get("0.49")
        if mg:
            expect_res("J1-J2 energy at the MG straddle",
                       f"{float(mg['E_per_site']):.5f}".replace("-", "\u2212"))
        gaps = [float(r_["gap_per_site"]) for r_ in J.values()]
        expect_res("J1-J2 collapse factor", f"\u00d7{max(gaps)/min(gaps):.0f}")
    # the cylinder/chain 2-sigma on the Results page. Codex found the page quoting a single
    # +/-0.05 while the data spans +/-0.049..0.061 -- the page rounded the error bar DOWN,
    # i.e. claimed more precision than measured, and nothing here checked it.
    if F:
        sig2r = []
        for p_ in sorted({p for (_, p) in F}):
            if (3, p_) in F and (1, p_) in F:
                v3, v1 = float(F[(3, p_)]["var_frac"]), float(F[(1, p_)]["var_frac"])
                e3, e1 = float(F[(3, p_)]["se_frac"]), float(F[(1, p_)]["se_frac"])
                r_ = v3 / v1
                sig2r.append(2 * r_ * math.sqrt((e3 / v3) ** 2 + (e1 / v1) ** 2))
        if sig2r:
            expect_res("cylinder/chain 2-sigma",
                       f"2σ of ±{min(sig2r):.2f}–{max(sig2r):.2f}")
    # The J1-J2 energy the page quotes is the J2 = 0.49 row, and it must say so: the CSV
    # skips the exact Majumdar-Ghosh point, and attributing -1.50021 to MG made the engine
    # look wrong exactly where chi = 8 is exact by construction.
    jj2 = os.path.join(DOCS, "j1j2_dimerization.csv")
    if os.path.exists(jj2):
        J2 = {f"{float(r_['j2']):.2f}": r_ for r_ in csv.DictReader(open(jj2))}
        if "0.50" in J2:
            fails.append("results page: j1j2 CSV now has a J2 = 0.50 row, so the page's "
                         "'the scan samples the two points straddling it' is stale")
        elif "0.49" in J2:
            expect_res("J1-J2 straddling energy attributed to J2 = 0.49/0.51",
                       "J2 = 0.49 and 0.51,\nwhose energies are "
                       + f"{float(J2['0.49']['E_per_site']):.5f}".replace("-", "−"))
    # Every figure on the page names where its numbers come from -- the page promises it in
    # its first paragraph, and the README repeats the promise. The GPU figure broke it, so
    # the promise is now structural: each `.. image::` must be followed, before the next
    # image or section heading, by a line starting "Data:".
    _lines = res.split("\n")
    for _i, _ln in enumerate(_lines):
        if not _ln.startswith(".. image:: "):
            continue
        _fig = _ln.split("/")[-1]
        _ok = False
        # grouped figures may share one Data: line (the two Haldane figures do), so
        # scan to the next SECTION, not the next image
        for _nx in _lines[_i + 1:]:
            if _nx.startswith("----"):
                break
            if _nx.startswith("Data:"):
                _ok = True
                break
        if not _ok:
            fails.append(f"results page: figure {_fig} has no 'Data:' line naming its "
                         f"source, though the page promises one for every figure")
    # The HVA-vs-state-vector bound the page states must not be tighter than the bound the
    # README states and CI enforces. The page said 1e-12 against a measured 2.7e-12.
    _rm = open(os.path.join(HERE, "..", "README.md"), encoding="utf-8").read()
    import re as _re2
    _pg = _re2.search(r"independent state-vector simulator to \*\*(1e-\d+)\*\*", res)
    _rd = _re2.search(r"state-vector simulation: energy and entropies to (1e-\d+)", _rm)
    if _pg and _rd and float(_pg.group(1)) < float(_rd.group(1)):
        fails.append(f"results page: claims HVA agreement to {_pg.group(1)}, tighter than "
                     f"the {_rd.group(1)} the README states and CI enforces")
    elif not _pg:
        fails.append("results page: the HVA-vs-state-vector precision sentence is gone or "
                     "reworded, so its bound can no longer be checked against the README")

    # The stripe demonstration. Its data line says the CSV is what plot_stripes.py writes
    # from the run's log, so the two must agree row for row; the page called a run the
    # showcase makes periodic in y a "strip", so the geometry is read from the showcase's
    # own MPO call; and the pattern the prose describes is re-derived from the profile.
    st_path = os.path.join(DOCS, "hubbard_stripes.csv")
    st_log = os.path.join(HERE, "stripes_16x4_chi1024.log")
    if os.path.exists(st_path) and os.path.exists(st_log):
        S_ = list(csv.DictReader(open(st_path)))
        _hd, _prof = None, []
        for _ln in open(st_log):
            _m = re.match(r"STRIPES Lx=(\d+) W=(\d+) chi=(\d+) U=(-?[\d.]+) holes=(\d+)", _ln)
            _hd = _hd or (_m and _m.groups())
            _m = re.match(r"PROF (\d+) (\d+) (\d+) (-?[\d.]+) (-?[\d.]+)$", _ln.strip())
            if _m:
                _prof.append(list(_m.groups()))
        _csv = [[r_["i"], r_["x"], r_["y"], r_["n"], r_["two_sz"]] for r_ in S_]
        _logrows = [p_[:3] + [f"{float(p_[3]):.8f}", f"{float(p_[4]):.8f}"] for p_ in _prof]
        if not _hd or _csv != sorted(_logrows, key=lambda p_: int(p_[0])):
            fails.append("results page: docs/hubbard_stripes.csv is not what plot_stripes.py "
                         "writes from benchmarks/stripes_16x4_chi1024.log, as the page says")
        nx = max(int(r_["x"]) for r_ in S_) + 1
        ny = max(int(r_["y"]) for r_ in S_) + 1
        hole, stag = [0.0] * nx, [0.0] * nx
        for r_ in S_:
            x_, y_ = int(r_["x"]), int(r_["y"])
            hole[x_] += (1.0 - float(r_["n"])) / ny
            stag[x_] += (-1) ** (x_ + y_) * float(r_["two_sz"]) / (2 * ny)
        nholes = round(sum(hole) * ny)
        _src = open(os.path.join(HERE, "..", "src", "hubbard_showcase.cpp"),
                    encoding="utf-8").read()
        _m = re.search(r"QX_STRIPES.*?build_hubbard_lattice_split_mpo\(([^;]*?)\);", _src, re.S)
        _yp = [a_.strip() for a_ in _m.group(1).split(",")][5] if _m else None
        if _yp not in ("true", "false") or not _hd:
            fails.append("results page: cannot read the stripe run's geometry from "
                         "src/hubbard_showcase.cpp and its log, so the stripe prose is unchecked")
        else:
            expect_res("stripe geometry, U and doping",
                       f"{nx}\u00d7{ny} {'cylinder' if _yp == 'true' else 'strip'} at "
                       f"U = {float(_hd[3]):g} and 1/{(nx * ny) // nholes} hole")
            expect_res("stripe size and chi", f"{nx * ny} sites at \u03c7 = {_hd[2]}")
        # hole-rich rings with the AF order flipping sign across each one
        peaks = [x_ for x_ in range(1, nx - 1) if hole[x_] > max(hole[x_ - 1], hole[x_ + 1])]
        flips = [x_ + 0.5 for x_ in range(nx - 1) if stag[x_] * stag[x_ + 1] < 0]
        if (len(flips) < 2 or len(peaks) != len(flips)
                or any(min(abs(f_ - p_) for p_ in peaks) > 1 for f_ in flips)):
            fails.append(f"results page: the AF sign flips at x = {flips} no longer sit on the "
                         f"hole-rich rings at x = {peaks}, as the stripe prose says")
        else:
            expect_res("stripe spacing",
                       f"about every {(flips[-1] - flips[0]) / (len(flips) - 1):.0f} sites")
            # holes per site along a ring: 1/2 is "half-filled", and filled stripes at the
            # same doping sit 1/(1/2) = twice as far apart
            if nholes / len(flips) / ny != 0.5:
                fails.append(f"results page: {nholes} holes in {len(flips)} rings of {ny} is "
                             f"not half-filled, so 'half-filled ... twice as far apart' is stale")
            else:
                expect_res("stripe filling", "The half-filled pattern here")
                expect_res("filled-stripe spacing", "*filled*, twice as far apart")
    else:
        fails.append("results page: the stripe CSV or the log it is written from is missing")
    # and every figure the page embeds must be a file that exists, so a fresh checkout
    # cannot deploy a broken image -- the failure that reached review on this very page
    import re as _re
    for m in _re.findall(r"\.\./([A-Za-z0-9_.]+\.svg)", res):
        if not os.path.exists(os.path.join(DOCS, m)):
            fails.append(f"results page: embeds ../{m}, which is not in docs/")
    # ... and every data file it cites must be COMMITTED: an untracked file passes
    # os.path.exists on the one machine that made it and fails on every fresh checkout.
    try:
        _tracked = set(subprocess.run(["git", "ls-files"], cwd=os.path.join(HERE, ".."),
                                      capture_output=True, text=True,
                                      check=True).stdout.splitlines())
    except (OSError, subprocess.CalledProcessError):
        _tracked = None                     # no git (an unpacked sdist): existence only

    def _have(rel):
        if _tracked is not None:
            return rel in _tracked
        return os.path.exists(os.path.join(HERE, "..", rel))

    for m in _re.findall(r"``((?:docs|benchmarks)/[A-Za-z0-9_.+-]+\.(?:csv|log))``", res):
        if not _have(m):
            fails.append(f"results page: cites {m}, which is not committed")
    # A figure is reproducible only if the script that draws it runs on a fresh checkout.
    # Codex found the stripe renderer reading a log that was never committed -- the CSV the
    # page named was the script's OUTPUT -- so every .csv/.log a figure's renderer names
    # must be committed as well.
    for svg in _re.findall(r"\.\./([A-Za-z0-9_.]+\.svg)", res):
        for py in sorted(glob.glob(os.path.join(HERE, "plot_*.py"))):
            body = open(py, encoding="utf-8").read()
            if svg not in body:
                continue
            for lit in sorted(set(_re.findall(r"[\"']([A-Za-z0-9_.+-]+\.(?:csv|log))[\"']",
                                              body))):
                if not (_have("benchmarks/" + lit) or _have("docs/" + lit)):
                    fails.append(f"results page: {svg} is drawn by benchmarks/"
                                 f"{os.path.basename(py)}, which reads {lit}, and that file "
                                 f"is not committed")

if fails:
    # The doc uses -, x, sigma and superscripts, so the report itself contains non-ASCII.
    # On a GBK console `print` then raises UnicodeEncodeError and the operator sees a
    # traceback instead of the mismatches -- a reporter that cannot report its findings.
    out = sys.stdout
    enc = (getattr(out, "encoding", None) or "ascii").lower()

    def say(line):
        try:
            print(line)
        except UnicodeEncodeError:
            print(line.encode(enc, "backslashreplace").decode(enc, "replace"))

    say(f"{len(fails)} MISMATCH(es) between the docs and the CSVs:\n")
    for f in fails:
        say("  " + f)
    sys.exit(1)
print("all quoted figures in docs/universal_mpo.md and docs/source/results.rst reproduce from the CSVs")
