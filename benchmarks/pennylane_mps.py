"""quantx.mps against default.tensor, PennyLane's MPS device (on quimb).

Three circuits. "demo" is the 50-qubit circuit of PennyLane's demo on simulating circuits with
tensor networks. It entangles little, so no bond cap binds, and its time is mostly the Python
that decomposes its gates. "brick" and "hop" entangle enough for the cap to bind on both
devices, which is what makes them comparable: below the cap the two differ in what they drop
(default.tensor keeps a discarded weight of machine epsilon, quantx.mps Schmidt values down to
1e-14 of the norm). "brick" does not conserve anything. "hop", a Trotterised XY chain from a
Neel state, conserves the number of |1>s, so quantx.mps runs it in charge blocks by itself, and
default.tensor has none to use. Each device gets OMP_NUM_THREADS threads; set it before running.

    OMP_NUM_THREADS=2 python benchmarks/pennylane_mps.py
    OMP_NUM_THREADS=2 python benchmarks/pennylane_mps.py --quick

Needs pennylane>=0.45, quimb and the native engine. Prints the best of two runs, for quantx.mps
with both of its SVDs, the bond the state reached and whether quantx.mps used the charge, and
the largest difference between the two devices' expectation values. At a binding cap that
difference is truncation, which goes differently in the two, not a bug.
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np
import pennylane as qml

PLAN = (("demo", 50, (64,)), ("brick", 40, (32, 64, 128)), ("hop", 48, (64, 128, 256)))
QUICK = (("demo", 12, (16,)), ("brick", 12, (16,)), ("hop", 12, (16,)))
LAYERS = 32


def demo(n):
    theta, phi = 0.5, 0.1
    for q in range(n - 4):
        qml.X(wires=q)
        qml.RX(theta, wires=q + 1)
        qml.CNOT(wires=[q, q + 1])
        qml.DoubleExcitation(phi, wires=[q, q + 1, q + 3, q + 4])
        qml.CSWAP(wires=[q + 1, q + 3, q + 4])
        qml.RY(theta, wires=q + 1)
        qml.Toffoli(wires=[q + 1, q + 3, q + 4])
    return [qml.expval(qml.Z(0)), qml.expval(qml.Hamiltonian([0.1, 0.2], [qml.Z(0), qml.Y(1)])),
            qml.expval(qml.Y(n - 1))]


def brick(n):
    for layer in range(LAYERS):
        for w in range(layer % 2, n - 1, 2):
            qml.IsingZZ(0.6, wires=[w, w + 1])
            qml.RX(0.7, wires=w)
            qml.RY(0.4, wires=w + 1)
    return [qml.expval(qml.Z(n // 2)), qml.expval(qml.X(n // 2) @ qml.X(n // 2 + 1))]


def hop(n):
    qml.BasisState(np.array([1, 0] * (n // 2)), wires=range(n))
    for layer in range(LAYERS):
        for w in range(layer % 2, n - 1, 2):
            qml.IsingXY(0.5, wires=[w, w + 1])
    return [qml.expval(qml.Z(n // 2)), qml.expval(qml.Z(n // 2) @ qml.Z(n // 2 + 1))]


def timed(device, circuit, n):
    node = qml.QNode(lambda: circuit(n), device)
    best = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        out = node()
        best = min(best, time.perf_counter() - t0)
    return best, np.array(out, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--quick", action="store_true", help="small sizes, to check it runs")
    args = parser.parse_args()
    print(f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
          f"PennyLane {qml.__version__}, {LAYERS} layers")
    print(f"{'circuit':>8} {'wires':>5} {'cap':>5} {'bond':>5} {'charge':>6} {'direct':>9} "
          f"{'gram':>9} {'default.tensor':>15}  largest difference")
    for name, n, bonds in QUICK if args.quick else PLAN:
        circuit = {"demo": demo, "brick": brick, "hop": hop}[name]
        for chi in bonds:
            dev = qml.device("quantx.mps", wires=n, max_bond=chi)
            with qml.Tracker(dev) as tracker:
                qx, a = timed(dev, circuit, n)
            bond = tracker.history["max_bond"][-1]
            charge = "yes" if tracker.history["conserved"][-1] else "no"
            gram, _ = timed(qml.device("quantx.mps", wires=n, max_bond=chi, svd="gram"), circuit, n)
            dt, b = timed(qml.device("default.tensor", method="mps", wires=n, max_bond_dim=chi,
                                     contract="auto-mps"), circuit, n)
            print(f"{name:>8} {n:>5} {chi:>5} {bond:>5} {charge:>6} {qx:>7.2f} s {gram:>7.2f} s "
                  f"{dt:>13.2f} s  {float(np.max(np.abs(a - b))):.1e}", flush=True)


if __name__ == "__main__":
    main()
