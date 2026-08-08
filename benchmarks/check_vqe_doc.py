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
def row_cells(prefix):
    """The doc's table cells for the row starting with `prefix`, bold markers and
    whitespace stripped -- comparing cell by cell survives **bolding** of whichever
    cell an author decided to emphasise."""
    for line in doc.split(chr(10)):
        if line.startswith(prefix):
            return [c.strip().replace("*", "") for c in line.strip("|").split("|")]
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

if fails:
    print(f"{len(fails)} MISMATCH(es) between docs/universal_mpo.md and the CSVs:\n")
    for f in fails:
        print("  " + f)
    sys.exit(1)
print("all quoted VQE figures in docs/universal_mpo.md reproduce from the CSVs")
