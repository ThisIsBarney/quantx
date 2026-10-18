#!/usr/bin/env python
"""GPU benchmark for QuantX — run on a machine with CuPy/CUDA installed.

Usage:
    python benchmark_gpu.py
    python benchmark_gpu.py --quick    # smaller sizes, faster run
    python benchmark_gpu.py --full     # larger sizes, ~5-10 min

Output: formatted table of timings and GPU/CPU speedup ratios.
         Includes both fp64 (complex128) and fp32 (complex64) GPU columns.
"""
from __future__ import annotations


def bench_statevector(n_qubits: int, n_gates: int = 50, backends=("numpy",)):
    import time
    import random
    import math
    from quantx_engine.contrib.gpu import GpuStatevector

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

    # Run both fp64 and fp32 for GPU; only fp64 for numpy
    run_keys = []
    for backend in backends:
        run_keys.append((backend, "complex128"))
        if backend == "cupy":
            run_keys.append((backend, "complex64"))

    for backend, dtype in run_keys:
        key = backend if dtype == "complex128" else f"{backend}_fp32"
        try:
            sv = GpuStatevector(n_qubits=n_qubits, backend=backend, dtype=dtype)
            # Warmup (2 passes to prime GPU code paths)
            for _ in range(2):
                sv.apply_gate("H", 0)
                sv.reset()

            t0 = time.perf_counter()
            for name, q, params in gate_seq:
                sv.apply_gate(name, q, params)
            _ = sv.norm()   # force GPU synchronization
            t1 = time.perf_counter()
            results[key] = t1 - t0
        except Exception as e:
            results[key] = f"ERROR: {e}"
    return results


def bench_batched(n_qubits: int, n_circuits: int, backends=("numpy",)):
    import time
    import numpy as np
    from quantx_engine.contrib.gpu import GpuBatchedCircuit

    rng = np.random.default_rng(42)
    angles = rng.uniform(0, 2 * np.pi, n_circuits)

    run_keys = []
    for backend in backends:
        run_keys.append((backend, "complex128"))
        if backend == "cupy":
            run_keys.append((backend, "complex64"))

    results = {}
    for backend, dtype in run_keys:
        key = backend if dtype == "complex128" else f"{backend}_fp32"
        try:
            bc = GpuBatchedCircuit(
                n_qubits=n_qubits, n_circuits=n_circuits, backend=backend, dtype=dtype
            )
            # Warmup
            bc.apply_gate("H", 0)
            bc.reset()

            t0 = time.perf_counter()
            for q in range(n_qubits):
                bc.apply_gate("H", q)
            bc.apply_gate("RX", 0, params=angles)   # per-circuit angles
            for q in range(0, n_qubits - 1, 2):
                bc.apply_cnot(q, q + 1)
            _ = bc.probabilities()
            t1 = time.perf_counter()
            results[key] = t1 - t0
        except Exception as e:
            results[key] = f"ERROR: {e}"
    return results


def bench_lanczos(n_sites: int, k: int = 4, backends=("numpy",)):
    import time
    from quantx_engine.contrib.gpu import GpuLanczos
    from quantx_engine.contrib.lanczos import SparseHamiltonian

    results = {}
    H = SparseHamiltonian.ising_1d(n_sites=n_sites, J=1.0, h=0.5)

    for backend in backends:
        try:
            solver = GpuLanczos(k=k, max_iter=200, backend=backend, seed=42)
            # Warmup with n_sites=8 (dim=256) to fully prime cuSPARSE kernel selection
            warmup_H = SparseHamiltonian.ising_1d(n_sites=8)
            GpuLanczos(k=2, backend=backend, seed=0).solve(warmup_H)

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
        from quantx_engine.contrib.gpu import GpuStatevector, GpuBatchedCircuit, GpuLanczos, gpu_info
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

    W = 90  # table width

    def fmt_t(v):
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    def sp(t_np, t_gpu):
        if isinstance(t_np, float) and isinstance(t_gpu, float) and t_gpu > 0:
            return f"{t_np/t_gpu:.1f}x"
        return "N/A"

    has_gpu = info['cupy_available']

    # --- Statevector benchmark ---
    if args.quick:
        sv_sizes = [10, 14, 18]
    elif args.full:
        sv_sizes = [10, 14, 18, 22, 26]
    else:
        sv_sizes = [10, 14, 18, 22]

    print("─" * W)
    print("GpuStatevector (50 gates)")
    if has_gpu:
        print(f"{'n_qubits':>10}  {'dim':>10}  {'numpy (s)':>12}  {'cupy-f64 (s)':>14}  {'sp-f64':>8}  {'cupy-f32 (s)':>14}  {'sp-f32':>8}")
    else:
        print(f"{'n_qubits':>10}  {'dim':>10}  {'numpy (s)':>12}")
    print("─" * W)
    for n in sv_sizes:
        r = bench_statevector(n, n_gates=50, backends=backends)
        t_np  = r.get("numpy", "N/A")
        t_f64 = r.get("cupy", "N/A")
        t_f32 = r.get("cupy_fp32", "N/A")
        if has_gpu:
            print(f"{n:>10}  {2**n:>10,}  {fmt_t(t_np):>12}  {fmt_t(t_f64):>14}  {sp(t_np,t_f64):>8}  {fmt_t(t_f32):>14}  {sp(t_np,t_f32):>8}")
        else:
            print(f"{n:>10}  {2**n:>10,}  {fmt_t(t_np):>12}")
    print()

    # --- Batched circuit benchmark ---
    if args.quick:
        batch_cases = [(4, 100), (4, 1000), (8, 100), (8, 1000)]
    elif args.full:
        batch_cases = [(4, 100), (4, 1000), (4, 10000), (8, 100), (8, 1000), (8, 10000), (10, 100), (10, 1000)]
    else:
        batch_cases = [(4, 100), (4, 1000), (4, 10000), (8, 100), (8, 1000), (10, 100)]

    print("─" * W)
    print("GpuBatchedCircuit (H + RX_per_circuit + CNOT chain)")
    if has_gpu:
        print(f"{'n_qubits':>10}  {'n_circuits':>12}  {'numpy (s)':>12}  {'cupy-f64 (s)':>14}  {'sp-f64':>8}  {'cupy-f32 (s)':>14}  {'sp-f32':>8}")
    else:
        print(f"{'n_qubits':>10}  {'n_circuits':>12}  {'numpy (s)':>12}")
    print("─" * W)
    for (nq, nb) in batch_cases:
        r = bench_batched(nq, nb, backends=backends)
        t_np  = r.get("numpy", "N/A")
        t_f64 = r.get("cupy", "N/A")
        t_f32 = r.get("cupy_fp32", "N/A")
        if has_gpu:
            print(f"{nq:>10}  {nb:>12,}  {fmt_t(t_np):>12}  {fmt_t(t_f64):>14}  {sp(t_np,t_f64):>8}  {fmt_t(t_f32):>14}  {sp(t_np,t_f32):>8}")
        else:
            print(f"{nq:>10}  {nb:>12,}  {fmt_t(t_np):>12}")
    print()

    # --- Lanczos benchmark ---
    if args.quick:
        lz_sizes = [10, 12]
    elif args.full:
        lz_sizes = [10, 12, 14, 16]
    else:
        lz_sizes = [10, 12, 14]

    print("─" * W)
    print("GpuLanczos (Ising 1D, k=4 lowest eigenvalues)  [fp64 only — sparse solver]")
    print(f"{'n_sites':>10}  {'dim':>8}  {'numpy (s)':>12}  {'cupy (s)':>12}  {'speedup':>10}  {'E_0':>12}")
    print("─" * W)
    for n in lz_sizes:
        r = bench_lanczos(n, k=4, backends=backends)
        np_r = r.get("numpy", {})
        cp_r = r.get("cupy", {})
        t_np = np_r.get("time") if isinstance(np_r, dict) else None
        t_cp = cp_r.get("time") if isinstance(cp_r, dict) else None
        e0   = np_r.get("energy", "?") if isinstance(np_r, dict) else "?"
        t_np_s = f"{t_np:.4f}" if t_np else "ERROR"
        t_cp_s = f"{t_cp:.4f}" if t_cp else "N/A"
        e0_s   = f"{e0:.6f}" if isinstance(e0, float) else str(e0)
        print(f"{n:>10}  {2**n:>8,}  {t_np_s:>12}  {t_cp_s:>12}  {sp(t_np,t_cp):>10}  {e0_s:>12}")
    print()
    print("=" * W)
    if not has_gpu:
        print("NOTE: CuPy not installed -- only numpy baseline shown.")
        print("      Install CuPy matching your CUDA version, e.g.:")
        print("      pip install cupy-cuda12x")
    else:
        print("NOTE: fp32 speedup uses complex64 internal dtype — ~64x more FLOPS on consumer GPUs.")
        print("      Lanczos uses fp64 throughout (cuSPARSE sparse solver stays in fp64).")


if __name__ == "__main__":
    main()
