"""What the adjoint gradient of an HVA circuit costs, against the number of angles (M13).

For circuits of 1 to 32 layers on one lattice, this times an energy evaluation, the energy
with its adjoint gradient (``hva.circuit(..., gradient=True)``), and central differences,
which take two circuits per angle. Both cost more with depth, as the circuit does, but
only central differences pay per angle: their cost over one energy evaluation grows
with the angles. The adjoint gradient's is set by bond dimensions instead. The backward
pass carries H|psi> as well as psi, and H|psi> needs up to w times psi's bond dimension
(w the Hamiltonian's MPO bond dimension, 6 for the chain), up to max_bond. So the ratio
is largest while psi's bonds are well below max_bond, and settles once the circuit
fills them. Each row also gives what truncation cost: the circuit's truncation_error
and the gradient's gradient_truncation_error. From 16 layers on, max_bond 128 truncates
the chain hard, so those rows time an approximate gradient.

    python benchmarks/hva_gradient_cost.py              # a 16-site chain, chi = 128
    python benchmarks/hva_gradient_cost.py --quick      # a 6-site chain, a few seconds

The times are the best of three runs, so a busy machine moves them, but not the trend.
Central differences are timed directly up to 4 layers and estimated from the energy's
time beyond, as two circuits per angle.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))

from quantx_engine import hva  # noqa: E402
from quantx_engine.models import HubbardLattice  # noqa: E402


def best_of(f, reps=3):
    best, out = float("inf"), None
    for _ in range(reps):
        t0 = time.perf_counter()
        out = f()
        best = min(best, time.perf_counter() - t0)
    return best, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--quick", action="store_true", help="a 6-site chain and up to 4 layers")
    args = ap.parse_args()
    if args.quick:
        model, chi, depths = HubbardLattice(Lx=6, W=1, U=4.0), 64, (1, 2, 4)
    else:
        model, chi, depths = HubbardLattice(Lx=16, W=1, U=4.0), 128, (1, 2, 4, 8, 16, 32)
    lay = hva.layout(model)
    kw = dict(init="product", max_bond=chi, cutoff=1e-12, entropy=False)
    rng = np.random.default_rng(1)
    print(f"{model.L}-site chain, U = {model.U:g}, max_bond {chi}, product start; "
          f"{lay.angles_per_layer} angles per layer")
    print(f"{'layers':>6} {'angles':>6} {'energy s':>9} {'adjoint s':>10} {'ratio':>6} "
          f"{'central s':>10} {'ratio':>6} {'trunc':>8} {'grad trunc':>10}")
    ratios = []
    for p in depths:
        angles = rng.uniform(-0.6, 0.6, size=(p, lay.angles_per_layer))
        t_e, r = best_of(lambda: hva.circuit(model, angles.tolist(), **kw))
        t_g, rg = best_of(lambda: hva.circuit(model, angles.tolist(), gradient=True, **kw))
        assert rg.energy == r.energy
        n = angles.size
        if p <= 4:
            def central():
                h = 1e-4
                for idx in np.ndindex(*angles.shape):
                    for m in (-1, 1):
                        a = angles.copy()
                        a[idx] += m * h
                        hva.circuit(model, a.tolist(), **kw)
            t_c, _ = best_of(central, reps=1)
            mark = ""
        else:
            t_c, mark = 2 * n * t_e, "*"
        ratios.append(t_g / t_e)
        print(f"{p:>6} {n:>6} {t_e:>9.3f} {t_g:>10.3f} {t_g / t_e:>6.1f} "
              f"{t_c:>9.2f}{mark:1s} {t_c / t_e:>6.1f} {r.truncation_error:>8.1e} "
              f"{rg.gradient_truncation_error:>10.1e}")
    print("* estimated as two circuits per angle")
    print("trunc: the circuit's largest discarded weight; grad trunc: the gradient's, in "
          "H|psi>'s compression and the backward pass")
    print(f"adjoint / energy: {min(ratios):.1f} to {max(ratios):.1f} over "
          f"{depths[0]} to {depths[-1]} layers")
    # the claim: no cost per angle. At the deepest circuit the adjoint gradient costs a few
    # energy evaluations, where central differences cost two per angle.
    n_deep = depths[-1] * lay.angles_per_layer
    assert ratios[-1] < 0.25 * 2 * n_deep, (ratios, n_deep)


if __name__ == "__main__":
    main()
