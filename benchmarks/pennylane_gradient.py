"""The adjoint gradient of a PennyLane circuit on quantx.statevector, against PennyLane's own.

StronglyEntanglingLayers with a ZZ observable: a forward and a backward pass, from a QNode, so
the times include PennyLane's preprocessing. default.qubit runs in NumPy, lightning.qubit on
its AVX kernels. Every device gets OMP_NUM_THREADS threads; set it before running.

    OMP_NUM_THREADS=2 python benchmarks/pennylane_gradient.py
    OMP_NUM_THREADS=2 python benchmarks/pennylane_gradient.py --quick

Needs pennylane>=0.45 (which brings lightning.qubit) and the native engine. Prints the best of
three runs of each, and checks that the three gradients agree.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pennylane as qml

SIZES = ((12, 4), (16, 4), (20, 2))
QUICK = ((10, 2),)
SKIP_DEFAULT_ABOVE = 16         # NumPy takes minutes per gradient from here on


def gradient(device, n, layers):
    def circuit(w):
        qml.StronglyEntanglingLayers(w, wires=range(n))
        return qml.expval(qml.Z(0) @ qml.Z(n - 1))

    w = qml.numpy.array(np.random.default_rng(1).uniform(-1, 1, size=(layers, n, 3)),
                        requires_grad=True)
    node = qml.QNode(circuit, qml.device(device, wires=n), diff_method="adjoint")
    grad = qml.grad(node)
    grad(w)                     # the first call builds what later calls reuse
    best = float("inf")
    for _ in range(3):
        t0 = time.perf_counter()
        out = grad(w)
        best = min(best, time.perf_counter() - t0)
    return best, np.asarray(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="one small size, to check it runs")
    args = parser.parse_args()
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
          f"PennyLane {qml.__version__}")
    print(f"{'n':>3} {'layers':>6} {'params':>6} {'quantx.statevector':>19} "
          f"{'lightning.qubit':>16} {'default.qubit':>14}  largest difference")
    for n, layers in QUICK if args.quick else SIZES:
        qx, g = gradient("quantx.statevector", n, layers)
        lit, g_lit = gradient("lightning.qubit", n, layers)
        worst = float(np.max(np.abs(g - g_lit)))
        cell = "-"
        if n <= SKIP_DEFAULT_ABOVE:
            dq, g_dq = gradient("default.qubit", n, layers)
            worst = max(worst, float(np.max(np.abs(g - g_dq))))
            cell = f"{dq * 1e3:.1f} ms"
        print(f"{n:>3} {layers:>6} {3 * n * layers:>6} {qx * 1e3:>16.1f} ms "
              f"{lit * 1e3:>13.1f} ms {cell:>14}  {worst:.1e}")
        assert worst < 1e-10


if __name__ == "__main__":
    main()
