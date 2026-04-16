#!/usr/bin/env python
"""GPU benchmark for QuantX — run on a machine with CuPy/CUDA installed.

Usage:
    python benchmark_gpu.py
    python benchmark_gpu.py --quick    # smaller sizes, faster run
    python benchmark_gpu.py --full     # larger sizes, ~5-10 min

Output: formatted table of timings and GPU/CPU speedup ratios.
"""
from __future__ import annotations


def bench_statevector(n_qubits: int, n_gates: int = 50, backends=("numpy",)):
    import time
    import random
    import math
    from quantx_engine.gpu import GpuStatevector

    results = {}
    gate_names = ["H", "X", "RX", "RY", "RZ"]

    # Precompute a fixed gate sequence so every backend runs the same workload
    rng = random.Random(42)
    gate_seq = []
    for _ in range(n_gates):
        name = rng.choice(gate_names)
        q = rng.randrange(n_qubits)
        params = (rng.uniform(0, 2 * math.pi),) if name in ("RX", "RY", "RZ") else ()
        gate_seq.append((name, q, params))

    for backend in backends:
        try:
            sv = GpuStatevector(n_qubits=n_qubits, backend=backend)
            # Warmup
            sv.apply_gate("H", 0)
            sv.reset()

            t0 = time.perf_counter()
            for name, q, params in gate_seq:
                sv.apply_gate(name, q, params)
            # Force synchronization for GPU
            _ = sv.norm()
            t1 = time.perf_counter()
            results[backend] = t1 - t0
        except Exception as e:
            results[backend] = f"ERROR: {e}"
    return results


def bench_batched(n_qubits: int, n_circuits: int, backends=("numpy",)):
    import time
    import numpy as np
    from quantx_engine.gpu import GpuBatchedCircuit

    results = {}

    for backend in backends:
        try:
            bc = GpuBatchedCircuit(n_qubits=n_qubits, n_circuits=n_circuits, backend=backend)
            # Warmup
            bc.apply_gate("H", 0)
            bc.reset()

            rng = np.random.default_rng(42)
            angles = rng.uniform(0, 2 * np.pi, n_circuits)

            t0 = time.perf_counter()
            for q in range(n_qubits):
                bc.apply_gate("H", q)
            bc.apply_gate("RX", 0, params=angles)  # per-circuit
            for q in range(0, n_qubits - 1, 2):
                bc.apply_cnot(q, q + 1)
            _ = bc.probabilities()
            t1 = time.perf_counter()
            results[backend] = t1 - t0
        except Exception as e:
            results[backend] = f"ERROR: {e}"
    return results


def bench_lanczos(n_sites: int, k: int = 4, backends=("numpy",)):
    import time
    from quantx_engine.gpu import GpuLanczos
    from quantx_engine.lanczos import SparseHamiltonian

    results = {}
    H = SparseHamiltonian.ising_1d(n_sites=n_sites, J=1.0, h=0.5)

    for backend in backends:
        try:
            solver = GpuLanczos(k=k, max_iter=200, backend=backend, seed=42)
            # Warmup
            small_H = SparseHamiltonian.ising_1d(n_sites=4)
            GpuLanczos(k=2, backend=backend, seed=0).solve(small_H)

            t0 = time.perf_counter()
            result = solver.solve(H)
            _ = result.ground_state_energy()
            t1 = time.perf_counter()
            results[backend] = {"time": t1 - t0, "energy": result.ground_state_energy()}
        except Exception as e:
            results[backend] = {"time": None, "error": str(e)}
    return results


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="QuantX GPU benchmark")
    parser.add_argument("--quick", action="store_true", help="Small sizes only")
    parser.add_argument("--full",  action="store_true", help="Large sizes (slow)")
    args = parser.parse_args()

    try:
        from quantx_engine.gpu import GpuStatevector, GpuBatchedCircuit, GpuLanczos, gpu_info
    except ImportError:
        print("ERROR: quantx_engine not installed. Run from ~/coding/quantx/python/")
        sys.exit(1)

    info = gpu_info()
    print("=" * 70)
    print("QuantX GPU Benchmark")
    print("=" * 70)
    print(f"CuPy available : {info['cupy_available']}")
    if info['cupy_available']:
        print(f"CuPy version   : {info.get('cupy_version', 'unknown')}")
        print(f"Device         : {info.get('device_name', 'unknown')}")
        print(f"GPU memory     : {info.get('free_memory_gb', '?')}/{info.get('total_memory_gb', '?')} GB free/total")
    print(f"NumPy version  : {info['numpy_version']}")
    print()

    backends = ["numpy", "cupy"] if info['cupy_available'] else ["numpy"]

    # --- Statevector benchmark ---
    if args.quick:
        sv_sizes = [10, 14, 18]
    elif args.full:
        sv_sizes = [10, 14, 18, 22, 26]
    else:
        sv_sizes = [10, 14, 18, 22]

    print("─" * 70)
    print("GpuStatevector (50 gates)")
    print(f"{'n_qubits':>10}  {'dim':>10}  {'numpy (s)':>12}  {'cupy (s)':>12}  {'speedup':>10}")
    print("─" * 70)
    for n in sv_sizes:
        r = bench_statevector(n, n_gates=50, backends=backends)
        t_np = r.get("numpy", "N/A")
        t_cp = r.get("cupy", "N/A")
        speedup = f"{t_np/t_cp:.1f}x" if isinstance(t_np, float) and isinstance(t_cp, float) and t_cp > 0 else "N/A"
        t_np_s = f"{t_np:.4f}" if isinstance(t_np, float) else str(t_np)
        t_cp_s = f"{t_cp:.4f}" if isinstance(t_cp, float) else str(t_cp)
        print(f"{n:>10}  {2**n:>10,}  {t_np_s:>12}  {t_cp_s:>12}  {speedup:>10}")
    print()

    # --- Batched circuit benchmark ---
    if args.quick:
        batch_cases = [(4, 100), (4, 1000), (8, 100), (8, 1000)]
    elif args.full:
        batch_cases = [(4, 100), (4, 1000), (4, 10000), (8, 100), (8, 1000), (8, 10000), (10, 100), (10, 1000)]
    else:
        batch_cases = [(4, 100), (4, 1000), (4, 10000), (8, 100), (8, 1000), (10, 100)]

    print("─" * 70)
    print("GpuBatchedCircuit (H + RX_per_circuit + CNOT chain)")
    print(f"{'n_qubits':>10}  {'n_circuits':>12}  {'numpy (s)':>12}  {'cupy (s)':>12}  {'speedup':>10}")
    print("─" * 70)
    for (nq, nb) in batch_cases:
        r = bench_batched(nq, nb, backends=backends)
        t_np = r.get("numpy", "N/A")
        t_cp = r.get("cupy", "N/A")
        speedup = f"{t_np/t_cp:.1f}x" if isinstance(t_np, float) and isinstance(t_cp, float) and t_cp > 0 else "N/A"
        t_np_s = f"{t_np:.4f}" if isinstance(t_np, float) else str(t_np)
        t_cp_s = f"{t_cp:.4f}" if isinstance(t_cp, float) else str(t_cp)
        print(f"{nq:>10}  {nb:>12,}  {t_np_s:>12}  {t_cp_s:>12}  {speedup:>10}")
    print()

    # --- Lanczos benchmark ---
    if args.quick:
        lz_sizes = [10, 12]
    elif args.full:
        lz_sizes = [10, 12, 14, 16]
    else:
        lz_sizes = [10, 12, 14]

    print("─" * 70)
    print("GpuLanczos (Ising 1D, k=4 lowest eigenvalues)")
    print(f"{'n_sites':>10}  {'dim':>8}  {'numpy (s)':>12}  {'cupy (s)':>12}  {'speedup':>10}  {'E_0':>12}")
    print("─" * 70)
    for n in lz_sizes:
        r = bench_lanczos(n, k=4, backends=backends)
        np_r = r.get("numpy", {})
        cp_r = r.get("cupy", {})
        t_np = np_r.get("time") if isinstance(np_r, dict) else None
        t_cp = cp_r.get("time") if isinstance(cp_r, dict) else None
        e0 = np_r.get("energy", "?") if isinstance(np_r, dict) else "?"
        speedup = f"{t_np/t_cp:.1f}x" if t_np and t_cp and t_cp > 0 else "N/A"
        t_np_s = f"{t_np:.4f}" if t_np else "ERROR"
        t_cp_s = f"{t_cp:.4f}" if t_cp else "N/A"
        e0_s = f"{e0:.6f}" if isinstance(e0, float) else str(e0)
        print(f"{n:>10}  {2**n:>8,}  {t_np_s:>12}  {t_cp_s:>12}  {speedup:>10}  {e0_s:>12}")
    print()
    print("=" * 70)
    if not info['cupy_available']:
        print("NOTE: CuPy not installed -- only numpy baseline shown.")
        print("      Install CuPy matching your CUDA version, e.g.:")
        print("      pip install cupy-cuda12x")


if __name__ == "__main__":
    main()
