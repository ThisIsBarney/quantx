"""The CUDA-Q target quantx on a kernel that conserves the number of |1>s, with and without the
charge.

A Trotterised XXZ chain from a Neel state, written as CUDA-Q writes such things: each step is an
exp_pauli of XX, YY and ZZ on every bond, which CUDA-Q hands a simulator as basis changes, CNOTs
and an Rz. None of those conserves the number of |1>s by itself; fused bond by bond, they do,
and the target runs the state in charge blocks. QUANTX_CONSERVE=0 runs the same kernel on the
same engine, with the same fusion, and no charge, so the ratio is what the charge buys. At the
smallest size qpp-cpu, CUDA-Q's state vector, checks the energy. Set OMP_NUM_THREADS before
running; the bond cap binds at the larger sizes, and both runs then truncate.

    OMP_NUM_THREADS=2 python benchmarks/cudaq_target.py
    OMP_NUM_THREADS=2 python benchmarks/cudaq_target.py --quick

Needs CUDA-Q 0.17 with the quantx target installed (docs/source/api/interfaces.rst). Prints the
best of two runs, the bond each reached and the weight its truncations discarded.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time

import cudaq
from cudaq import spin

PLAN = ((16, (256,)), (48, (32, 64, 128)), (96, (32, 64)))
QUICK = ((12, (64,)), (24, (16,)))
STEPS, DT, DELTA = 20, 0.1, 0.5


@cudaq.kernel
def trotter(n: int, steps: int, dt: float, words: list[cudaq.pauli_word], coeffs: list[float]):
    q = cudaq.qvector(n)
    for i in range(0, n, 2):
        x(q[i])
    for _ in range(steps):
        for k in range(len(words)):
            exp_pauli(-coeffs[k] * dt, q, words[k])


def chain(n):
    words, coeffs = [], []
    h = 0.0 * spin.i(0)
    for i in range(n - 1):
        for p, c in (("X", 1.0), ("Y", 1.0), ("Z", DELTA)):
            words.append(cudaq.pauli_word("I" * i + p + p + "I" * (n - i - 2)))
            coeffs.append(c)
        h += spin.x(i) * spin.x(i + 1) + spin.y(i) * spin.y(i + 1) + DELTA * spin.z(i) * spin.z(i + 1)
    return words, coeffs, h


class StderrOf:
    """What C++ writes to stderr inside the block (the target's QUANTX_VERBOSE report)."""

    def __enter__(self):
        sys.stderr.flush()
        self.file = tempfile.TemporaryFile()
        self.saved = os.dup(2)
        os.dup2(self.file.fileno(), 2)
        return self

    def __exit__(self, *exc):
        os.dup2(self.saved, 2)
        os.close(self.saved)
        self.file.seek(0)
        self.text = self.file.read().decode(errors="replace")
        self.file.close()


def run(n, cap, conserve):
    words, coeffs, h = chain(n)
    os.environ.update(QUANTX_MAX_BOND=str(cap), QUANTX_CONSERVE=conserve, QUANTX_VERBOSE="1")
    best, energy, report = float("inf"), None, None
    for _ in range(2):
        with StderrOf() as err:
            t = time.perf_counter()
            energy = cudaq.observe(trotter, h, n, STEPS, DT, words, coeffs).expectation()
            best = min(best, time.perf_counter() - t)
        report = re.findall(r"the charge (used|not used), bond (\d+), truncation (\S+)", err.text)[-1]
    return best, energy, report


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="small sizes, to check the script")
    args = parser.parse_args()
    cudaq.set_target("quantx")
    print(f"XXZ chain (Delta = {DELTA}), {STEPS} Trotter steps of {DT}, "
          f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}")
    print(f"{'qubits':>6} {'cap':>5} | {'charge':>8} {'bond':>5} {'discarded':>10} | "
          f"{'no charge':>9} {'bond':>5} {'discarded':>10} | {'ratio':>5} {'dE':>8}")
    for n, caps in QUICK if args.quick else PLAN:
        for cap in caps:
            t1, e1, r1 = run(n, cap, "auto")
            t0, e0, r0 = run(n, cap, "0")
            assert r1[0] == "used" and r0[0] == "not used", (r1, r0)
            print(f"{n:>6} {cap:>5} | {t1:>7.2f}s {r1[1]:>5} {float(r1[2]):>10.1e} | "
                  f"{t0:>8.2f}s {r0[1]:>5} {float(r0[2]):>10.1e} | {t0 / t1:>5.1f} "
                  f"{abs(e1 - e0):>8.1e}")
        if n == (QUICK if args.quick else PLAN)[0][0]:
            words, coeffs, h = chain(n)
            cudaq.set_target("qpp-cpu")
            exact = cudaq.observe(trotter, h, n, STEPS, DT, words, coeffs).expectation()
            cudaq.set_target("quantx")
            print(f"{'':>13}  qpp-cpu, the state vector: |E(charge) - E| = {abs(e1 - exact):.1e}")


if __name__ == "__main__":
    main()
