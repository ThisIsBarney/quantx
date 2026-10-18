"""The Hubbard chain's finite-temperature current correlator, from the public Python API.

Writes <J(t) J(0)>_beta of the half-filled Hubbard chain (U = 4t, mu = U/2, which pins
<n> = 1 at every temperature) at beta = 0.5, 1 and 2, in the CSV format that
benchmarks/plot_conductivity.py reads (columns t, ReC_b<beta>, ImC_b<beta>, ...). J is
the particle current; quantx_engine.thermal states its convention.

The method is gated against a full-space exact diagonalisation in
python/quantx_engine/tests/test_thermal_ed.py. This chain is far beyond ED, so the run
is a demonstration, checked against itself at a smaller max_bond. The committed data
(docs/cond_hubbard_correlator.csv) is at max_bond 1024, about an hour per temperature on
the reference desktop. It is converged only to t ~ 1.2-1.5: max_bond 512 and 1024 differ
by more than 1% of C(0) after that, and by up to 5%.

Run:  python benchmarks/hubbard_finite_t_conductivity.py [L] [max_bond] [out.csv]
      python benchmarks/plot_conductivity.py docs/cond_hubbard_correlator.csv hubbard
"""
import csv
import os
import sys
import time

from quantx_engine import thermal
from quantx_engine.models import HubbardChain

L = int(sys.argv[1]) if len(sys.argv) > 1 else 16
MAX_BOND = int(sys.argv[2]) if len(sys.argv) > 2 else 1024
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(__file__), "..", "docs", "cond_hubbard_correlator.csv")
BETAS = (0.5, 1.0, 2.0)
U = 4.0

model = HubbardChain(L=L, U=U, mu=U / 2)
runs = []
for beta in BETAS:
    t0 = time.time()
    cc = thermal.current_correlation(model, beta=beta, t_max=3.0, dt=0.1, dbeta=0.025,
                                     max_bond=MAX_BOND, cutoff=1e-10)
    print(f"beta={beta}: C(0)={cc.correlation[0].real:.6f}  max_bond={cc.max_bond}  "
          f"{time.time() - t0:.0f} s", flush=True)
    runs.append(cc)

with open(OUT, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["t"] + [c for b in BETAS for c in (f"ReC_b{b:.2f}", f"ImC_b{b:.2f}")])
    for i, t in enumerate(runs[0].times):
        row = [f"{t:.4f}"]
        for cc in runs:
            row += [f"{cc.correlation[i].real:.6f}", f"{cc.correlation[i].imag:.6f}"]
        w.writerow(row)
print("wrote", OUT)
